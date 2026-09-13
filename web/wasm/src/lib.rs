use wasm_bindgen::prelude::*;

const BLOCK_HEADER_BYTES: usize = 20;
const UINT16_MAX: u32 = u16::MAX as u32;

fn error(message: &str) -> JsValue {
    JsValue::from_str(message)
}

fn read_u16(bytes: &[u8], offset: usize) -> u32 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]]) as u32
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn checked_add(left: i32, right: i32) -> Result<i32, &'static str> {
    if right > 0 && left > i32::MAX - right {
        return Err("fixed-point addition exceeded signed int32 range");
    }
    if right < 0 && left < i32::MIN - right {
        return Err("fixed-point addition exceeded signed int32 range");
    }
    Ok(left + right)
}

fn checked_sub(left: i32, right: i32) -> Result<i32, &'static str> {
    if right > 0 && left < i32::MIN + right {
        return Err("fixed-point subtraction exceeded signed int32 range");
    }
    if right < 0 && left > i32::MAX + right {
        return Err("fixed-point subtraction exceeded signed int32 range");
    }
    Ok(left - right)
}

#[wasm_bindgen]
pub fn runtime_version() -> String {
    "0.1.0".to_owned()
}

fn validate_config_inner(
    threshold: i32,
    reset: i32,
    decay_shifts: &[u32],
    refractory_steps: u32,
) -> Result<(), &'static str> {
    let _ = threshold;
    let _ = reset;
    if decay_shifts.is_empty() {
        return Err("decay_shifts must not be empty");
    }
    if decay_shifts.len() > 8 {
        return Err("WASM WebGPU contract supports at most 8 decay shifts");
    }
    if decay_shifts.iter().any(|shift| *shift > 31) {
        return Err("decay shifts must be between zero and 31");
    }
    if refractory_steps > UINT16_MAX {
        return Err("refractory steps are outside uint16 range");
    }
    Ok(())
}

#[wasm_bindgen]
pub fn validate_config(
    threshold: i32,
    reset: i32,
    decay_shifts: &[u32],
    refractory_steps: u32,
) -> Result<(), JsValue> {
    validate_config_inner(threshold, reset, decay_shifts, refractory_steps).map_err(error)
}

#[derive(Clone, Copy)]
struct BlockLayout {
    source_start: u32,
    source_count: u32,
    edge_count: u32,
    offsets_start: usize,
    targets_start: usize,
    weights_start: usize,
    delays_start: usize,
}

fn parse_block_layout(bytes: &[u8], n_neurons: u32) -> Result<BlockLayout, &'static str> {
    if bytes.len() < BLOCK_HEADER_BYTES {
        return Err("block is shorter than its header");
    }
    if &bytes[0..4] != b"SCB1" {
        return Err("invalid block magic");
    }
    if read_u32(bytes, 4) != 1 {
        return Err("unsupported block version");
    }

    let source_start = read_u32(bytes, 8);
    let source_count = read_u32(bytes, 12);
    let edge_count = read_u32(bytes, 16);
    if source_start
        .checked_add(source_count)
        .is_none_or(|end| end > n_neurons)
    {
        return Err("block source range is outside the graph");
    }

    let source_count_usize =
        usize::try_from(source_count).map_err(|_| "source count does not fit host usize")?;
    let edge_count_usize =
        usize::try_from(edge_count).map_err(|_| "edge count does not fit host usize")?;
    let offsets_count = source_count_usize
        .checked_add(1)
        .ok_or("CSR offset count overflow")?;
    let offsets_bytes = offsets_count
        .checked_mul(4)
        .ok_or("CSR offset byte size overflow")?;
    let edges_bytes = edge_count_usize
        .checked_mul(8)
        .ok_or("edge byte size overflow")?;
    let expected_size = BLOCK_HEADER_BYTES
        .checked_add(offsets_bytes)
        .and_then(|size| size.checked_add(edges_bytes))
        .ok_or("block byte size overflow")?;
    if bytes.len() != expected_size {
        return Err("block byte size does not match header");
    }

    let offsets_start = BLOCK_HEADER_BYTES;
    let mut offset = offsets_start;
    let mut previous = 0;
    for index in 0..=source_count_usize {
        let current = read_u32(bytes, offset);
        if index == 0 && current != 0 {
            return Err("CSR offsets must start at zero");
        }
        if current < previous {
            return Err("CSR offsets are not monotonic");
        }
        previous = current;
        offset += 4;
    }
    if previous != edge_count {
        return Err("CSR offsets do not cover the edge array");
    }

    let targets_start = offset;
    for _ in 0..edge_count_usize {
        if read_u32(bytes, offset) >= n_neurons {
            return Err("edge target is outside the graph");
        }
        offset += 4;
    }
    let weights_start = offset;
    offset += edge_count_usize * 2;
    let delays_start = offset;
    for _ in 0..edge_count_usize {
        if read_u16(bytes, offset) == 0 {
            return Err("edge delay must be positive");
        }
        offset += 2;
    }
    Ok(BlockLayout {
        source_start,
        source_count,
        edge_count,
        offsets_start,
        targets_start,
        weights_start,
        delays_start,
    })
}

fn validate_scb_block_inner(bytes: &[u8], n_neurons: u32) -> Result<u32, &'static str> {
    Ok(parse_block_layout(bytes, n_neurons)?.edge_count)
}

#[wasm_bindgen]
pub fn validate_scb_block(bytes: &[u8], n_neurons: u32) -> Result<u32, JsValue> {
    validate_scb_block_inner(bytes, n_neurons).map_err(error)
}

#[wasm_bindgen]
pub struct ConnectomeRuntime {
    n_neurons: usize,
    bucket_count: usize,
    threshold: i32,
    reset: i32,
    decay_shifts: Vec<u32>,
    refractory_steps: u32,
    potentials: Vec<i32>,
    refractory: Vec<u32>,
    arrivals: Vec<i32>,
    spikes: Vec<u8>,
    cursor: usize,
    expected_source: u32,
    step_open: bool,
}

impl ConnectomeRuntime {
    fn new_inner(
        n_neurons: u32,
        max_delay: u32,
        threshold: i32,
        reset: i32,
        decay_shifts: &[u32],
        refractory_steps: u32,
    ) -> Result<Self, &'static str> {
        validate_config_inner(threshold, reset, decay_shifts, refractory_steps)?;
        if max_delay > UINT16_MAX {
            return Err("maximum delay is outside uint16 range");
        }
        let n_neurons =
            usize::try_from(n_neurons).map_err(|_| "neuron count does not fit host usize")?;
        let bucket_count = usize::try_from(max_delay)
            .map_err(|_| "maximum delay does not fit host usize")?
            .checked_add(1)
            .ok_or("delay-line bucket count overflow")?;
        let arrival_len = bucket_count
            .checked_mul(n_neurons)
            .ok_or("delay-line size overflow")?;
        Ok(Self {
            n_neurons,
            bucket_count,
            threshold,
            reset,
            decay_shifts: decay_shifts.to_vec(),
            refractory_steps,
            potentials: vec![0; n_neurons],
            refractory: vec![0; n_neurons],
            arrivals: vec![0; arrival_len],
            spikes: vec![0; n_neurons],
            cursor: 0,
            expected_source: 0,
            step_open: false,
        })
    }

    fn set_initial_state_inner(
        &mut self,
        potentials: &[i32],
        refractory: &[u32],
    ) -> Result<(), &'static str> {
        if self.step_open {
            return Err("cannot reset state during an open timestep");
        }
        if potentials.len() != self.n_neurons || refractory.len() != self.n_neurons {
            return Err("initial state length does not match graph");
        }
        if refractory.iter().any(|value| *value > UINT16_MAX) {
            return Err("initial refractory counter is outside uint16 range");
        }
        self.potentials.copy_from_slice(potentials);
        self.refractory.copy_from_slice(refractory);
        Ok(())
    }

    fn begin_step_inner(&mut self) -> Result<Vec<u8>, &'static str> {
        if self.step_open {
            return Err("a timestep is already open");
        }
        let bucket_start = self
            .cursor
            .checked_mul(self.n_neurons)
            .ok_or("delay-line index overflow")?;
        for index in 0..self.n_neurons {
            let arrival_index = bucket_start + index;
            let input_current = self.arrivals[arrival_index];
            self.arrivals[arrival_index] = 0;
            if self.refractory[index] != 0 {
                self.potentials[index] = self.reset;
                self.refractory[index] -= 1;
                self.spikes[index] = 0;
                continue;
            }

            let mut potential = checked_add(self.potentials[index], input_current)?;
            for shift in &self.decay_shifts {
                potential = checked_sub(potential, potential >> shift)?;
            }
            if potential >= self.threshold {
                self.potentials[index] = self.reset;
                self.refractory[index] = self.refractory_steps;
                self.spikes[index] = 1;
            } else {
                self.potentials[index] = potential;
                self.refractory[index] = 0;
                self.spikes[index] = 0;
            }
        }
        self.expected_source = 0;
        self.step_open = true;
        Ok(self.spikes.clone())
    }

    fn schedule_block_inner(&mut self, bytes: &[u8]) -> Result<(), &'static str> {
        if !self.step_open {
            return Err("schedule_block requires an open timestep");
        }
        let n_neurons =
            u32::try_from(self.n_neurons).map_err(|_| "neuron count does not fit uint32")?;
        let layout = parse_block_layout(bytes, n_neurons)?;
        if layout.source_start != self.expected_source {
            return Err("blocks must be supplied in contiguous source order");
        }
        self.expected_source = self
            .expected_source
            .checked_add(layout.source_count)
            .ok_or("block source range overflow")?;

        for row in 0..layout.source_count as usize {
            let source = layout.source_start as usize + row;
            if self.spikes[source] == 0 {
                continue;
            }
            let row_offset = layout.offsets_start + row * 4;
            let start = read_u32(bytes, row_offset) as usize;
            let end = read_u32(bytes, row_offset + 4) as usize;
            for edge in start..end {
                let target_offset = layout.targets_start + edge * 4;
                let target = read_u32(bytes, target_offset) as usize;
                let weight_offset = layout.weights_start + edge * 2;
                let weight =
                    i16::from_le_bytes([bytes[weight_offset], bytes[weight_offset + 1]]) as i32;
                let delay = read_u16(bytes, layout.delays_start + edge * 2) as usize;
                let bucket = (self.cursor + delay) % self.bucket_count;
                let arrival_index = bucket * self.n_neurons + target;
                self.arrivals[arrival_index] = checked_add(self.arrivals[arrival_index], weight)?;
            }
        }
        Ok(())
    }

    fn finish_step_inner(&mut self) -> Result<(), &'static str> {
        if !self.step_open {
            return Err("finish_step requires an open timestep");
        }
        let n_neurons =
            u32::try_from(self.n_neurons).map_err(|_| "neuron count does not fit uint32")?;
        if self.expected_source != n_neurons {
            return Err("blocks do not cover every source neuron");
        }
        self.cursor = (self.cursor + 1) % self.bucket_count;
        self.expected_source = 0;
        self.step_open = false;
        Ok(())
    }
}

#[wasm_bindgen]
impl ConnectomeRuntime {
    #[wasm_bindgen(constructor)]
    pub fn new(
        n_neurons: u32,
        max_delay: u32,
        threshold: i32,
        reset: i32,
        decay_shifts: &[u32],
        refractory_steps: u32,
    ) -> Result<ConnectomeRuntime, JsValue> {
        Self::new_inner(
            n_neurons,
            max_delay,
            threshold,
            reset,
            decay_shifts,
            refractory_steps,
        )
        .map_err(error)
    }

    pub fn set_initial_state(
        &mut self,
        potentials: &[i32],
        refractory: &[u32],
    ) -> Result<(), JsValue> {
        self.set_initial_state_inner(potentials, refractory)
            .map_err(error)
    }

    pub fn begin_step(&mut self) -> Result<Vec<u8>, JsValue> {
        self.begin_step_inner().map_err(error)
    }

    pub fn schedule_block(&mut self, bytes: &[u8]) -> Result<(), JsValue> {
        self.schedule_block_inner(bytes).map_err(error)
    }

    pub fn finish_step(&mut self) -> Result<(), JsValue> {
        self.finish_step_inner().map_err(error)
    }

    pub fn final_potentials(&self) -> Vec<i32> {
        self.potentials.clone()
    }

    pub fn final_refractory(&self) -> Vec<u32> {
        self.refractory.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block() -> Vec<u8> {
        let mut bytes = vec![0_u8; 20 + 3 * 4 + 2 * 4 + 2 * 2 + 2 * 2];
        bytes[0..4].copy_from_slice(b"SCB1");
        bytes[4..8].copy_from_slice(&1_u32.to_le_bytes());
        bytes[8..12].copy_from_slice(&4_u32.to_le_bytes());
        bytes[12..16].copy_from_slice(&2_u32.to_le_bytes());
        bytes[16..20].copy_from_slice(&2_u32.to_le_bytes());
        let mut offset = 20;
        for value in [0_u32, 1, 2] {
            bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            offset += 4;
        }
        for value in [7_u32, 6] {
            bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            offset += 4;
        }
        for value in [1_i16, -2] {
            bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
            offset += 2;
        }
        for value in [1_u16, 2] {
            bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
            offset += 2;
        }
        bytes
    }

    fn runtime_block() -> Vec<u8> {
        let mut bytes = vec![0_u8; 20 + 3 * 4 + 4 + 2 + 2];
        bytes[0..4].copy_from_slice(b"SCB1");
        bytes[4..8].copy_from_slice(&1_u32.to_le_bytes());
        bytes[8..12].copy_from_slice(&0_u32.to_le_bytes());
        bytes[12..16].copy_from_slice(&2_u32.to_le_bytes());
        bytes[16..20].copy_from_slice(&1_u32.to_le_bytes());
        let mut offset = 20;
        for value in [0_u32, 1, 1] {
            bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            offset += 4;
        }
        bytes[offset..offset + 4].copy_from_slice(&1_u32.to_le_bytes());
        offset += 4;
        bytes[offset..offset + 2].copy_from_slice(&8_i16.to_le_bytes());
        offset += 2;
        bytes[offset..offset + 2].copy_from_slice(&1_u16.to_le_bytes());
        bytes
    }

    #[test]
    fn accepts_valid_block() {
        assert_eq!(validate_scb_block_inner(&block(), 8).unwrap(), 2);
    }

    #[test]
    fn rejects_zero_delay() {
        let mut bytes = block();
        let last_delay = bytes.len() - 2;
        bytes[last_delay..].copy_from_slice(&0_u16.to_le_bytes());
        assert!(validate_scb_block_inner(&bytes, 8).is_err());
    }

    #[test]
    fn validates_config_bounds() {
        assert!(validate_config_inner(0, 0, &[2], 2).is_ok());
        assert!(validate_config_inner(0, 0, &[], 2).is_err());
        assert!(validate_config_inner(0, 0, &[32], 2).is_err());
        assert!(validate_config_inner(0, 0, &[2], 65_536).is_err());
    }

    #[test]
    fn streamed_runtime_matches_delayed_fixed_point_transition() {
        let mut runtime = ConnectomeRuntime::new_inner(2, 1, 4, 0, &[1], 0).unwrap();
        runtime.set_initial_state_inner(&[10, 0], &[0, 0]).unwrap();
        assert_eq!(runtime.begin_step_inner().unwrap(), vec![1, 0]);
        runtime.schedule_block_inner(&runtime_block()).unwrap();
        runtime.finish_step_inner().unwrap();

        assert_eq!(runtime.begin_step_inner().unwrap(), vec![0, 1]);
        runtime.schedule_block_inner(&runtime_block()).unwrap();
        runtime.finish_step_inner().unwrap();
        assert_eq!(runtime.final_potentials(), vec![0, 0]);
        assert_eq!(runtime.final_refractory(), vec![0, 0]);
    }
}
