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

fn validate_scb_block_inner(bytes: &[u8], n_neurons: u32) -> Result<u32, &'static str> {
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
    let offsets_bytes = (source_count_usize + 1)
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

    let mut offset = BLOCK_HEADER_BYTES;
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

    for _ in 0..edge_count_usize {
        if read_u32(bytes, offset) >= n_neurons {
            return Err("edge target is outside the graph");
        }
        offset += 4;
    }
    offset += edge_count_usize * 2;
    for _ in 0..edge_count_usize {
        if read_u16(bytes, offset) == 0 {
            return Err("edge delay must be positive");
        }
        offset += 2;
    }
    Ok(edge_count)
}

#[wasm_bindgen]
pub fn validate_scb_block(bytes: &[u8], n_neurons: u32) -> Result<u32, JsValue> {
    validate_scb_block_inner(bytes, n_neurons).map_err(error)
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
}
