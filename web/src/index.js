const INT32_MIN = -(2 ** 31);
const INT32_MAX = 2 ** 31 - 1;
const UINT16_MAX = 2 ** 16 - 1;
const UINT32_MAX = 2 ** 32 - 1;
const WORKGROUP_SIZE = 64;
const MAX_DECAY_SHIFTS = 8;
const CONFIG_WORDS = 16;
const BLOCK_HEADER_BYTES = 20;

// Keep this source byte-for-byte aligned with the Python WebGPU backend.
export const WGSL_LIF_SHADER = /* wgsl */ `
const I32_MAX: i32 = 2147483647;
const I32_MIN: i32 = -2147483647 - 1;
const MAX_DECAY_SHIFTS: u32 = 8u;

@group(0) @binding(0) var<storage, read_write> potentials: array<i32>;
@group(0) @binding(1) var<storage, read_write> refractory: array<u32>;
@group(0) @binding(2) var<storage, read_write> spikes: array<u32>;
@group(0) @binding(3) var<storage, read_write> arrivals: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read_write> config: array<i32>;
@group(0) @binding(5) var<storage, read_write> errors: array<atomic<i32>>;

fn checked_add(left: i32, right: i32) -> i32 {
  if (right > 0 && left > I32_MAX - right) {
    atomicStore(&errors[0], 1);
    return 0;
  }
  if (right < 0 && left < I32_MIN - right) {
    atomicStore(&errors[0], 1);
    return 0;
  }
  return left + right;
}

fn checked_sub(left: i32, right: i32) -> i32 {
  if (right > 0 && left < I32_MIN + right) {
    atomicStore(&errors[0], 1);
    return 0;
  }
  if (right < 0 && left > I32_MAX + right) {
    atomicStore(&errors[0], 1);
    return 0;
  }
  return left - right;
}

@compute @workgroup_size(64)
fn lif_step(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let index = global_id.x;
  let neuron_count = u32(config[0]);
  if (index >= neuron_count) { return; }

  let current_bucket = u32(config[1]);
  let bucket_count = u32(config[2]);
  let input_index = current_bucket * neuron_count + index;
  let input_current = atomicExchange(&arrivals[input_index], 0);
  let refractory_count = refractory[index];

  if (refractory_count != 0u) {
    potentials[index] = config[4];
    refractory[index] = refractory_count - 1u;
    spikes[index] = 0u;
    return;
  }

  var potential = checked_add(potentials[index], input_current);
  for (var shift_index: u32 = 0u; shift_index < MAX_DECAY_SHIFTS; shift_index++) {
    if (shift_index >= u32(config[6])) { break; }
    let shift = u32(config[8u + shift_index]);
    potential = checked_sub(potential, potential >> shift);
  }

  if (potential >= config[3]) {
    potentials[index] = config[4];
    refractory[index] = u32(config[5]);
    spikes[index] = 1u;
  } else {
    potentials[index] = potential;
    refractory[index] = 0u;
    spikes[index] = 0u;
  }
  _ = bucket_count;
}

@group(1) @binding(0) var<storage, read_write> edge_sources: array<u32>;
@group(1) @binding(1) var<storage, read_write> edge_targets: array<u32>;
@group(1) @binding(2) var<storage, read_write> edge_weights: array<i32>;
@group(1) @binding(3) var<storage, read_write> edge_delays: array<u32>;
@group(1) @binding(4) var<storage, read_write> edge_spikes: array<u32>;
@group(1) @binding(5) var<storage, read_write> edge_arrivals: array<atomic<i32>>;
@group(1) @binding(6) var<storage, read_write> edge_config: array<i32>;
@group(1) @binding(7) var<storage, read_write> edge_errors: array<atomic<i32>>;

fn atomic_checked_add(index: u32, delta: i32) {
  var old_value = atomicLoad(&edge_arrivals[index]);
  loop {
    if (delta > 0 && old_value > I32_MAX - delta) {
      atomicStore(&edge_errors[0], 1);
      return;
    }
    if (delta < 0 && old_value < I32_MIN - delta) {
      atomicStore(&edge_errors[0], 1);
      return;
    }
    let exchange = atomicCompareExchangeWeak(
      &edge_arrivals[index], old_value, old_value + delta
    );
    if (exchange.exchanged) { return; }
    old_value = exchange.old_value;
  }
}

@compute @workgroup_size(64)
fn schedule_edges(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let edge_index = global_id.x;
  let edge_count = u32(edge_config[7]);
  if (edge_index >= edge_count) { return; }
  let source = edge_sources[edge_index];
  if (edge_spikes[source] == 0u) { return; }

  let neuron_count = u32(edge_config[0]);
  let bucket_count = u32(edge_config[2]);
  let bucket = (u32(edge_config[1]) + edge_delays[edge_index]) % bucket_count;
  let arrival_index = bucket * neuron_count + edge_targets[edge_index];
  atomic_checked_add(arrival_index, edge_weights[edge_index]);
}
`;

function asDataView(data) {
  if (data instanceof ArrayBuffer) return new DataView(data);
  return new DataView(data.buffer, data.byteOffset, data.byteLength);
}

function bytesOf(value) {
  if (value instanceof Uint8Array) return value;
  if (ArrayBuffer.isView(value)) {
    return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
  }
  return new Uint8Array(value);
}

function requireInteger(value, label, minimum, maximum) {
  if (!Number.isInteger(value) || value < minimum || value > maximum) {
    throw new Error(`${label} is outside its declared integer range`);
  }
}

export function parseManifest(manifest) {
  if (!manifest || manifest.format_id !== "scx" || manifest.format_version !== 1) {
    throw new Error("unsupported connectome artifact manifest");
  }
  requireInteger(manifest.n_neurons, "manifest neuron count", 0, UINT32_MAX);
  requireInteger(manifest.n_edges, "manifest edge count", 0, UINT32_MAX);
  if (!Array.isArray(manifest.blocks)) throw new Error("manifest blocks must be an array");
  let expectedSource = 0;
  let expectedEdges = 0;
  for (const descriptor of manifest.blocks) {
    if (typeof descriptor.filename !== "string" || descriptor.filename.includes("..")) {
      throw new Error("manifest block filename is unsafe");
    }
    requireInteger(descriptor.source_start, "block source start", 0, UINT32_MAX);
    requireInteger(descriptor.source_count, "block source count", 0, UINT32_MAX);
    requireInteger(descriptor.edge_count, "block edge count", 0, UINT32_MAX);
    requireInteger(descriptor.byte_size, "block byte size", 0, Number.MAX_SAFE_INTEGER);
    if (!/^[0-9a-f]{64}$/.test(descriptor.sha256)) {
      throw new Error("block checksum is not a lowercase SHA-256 digest");
    }
    if (descriptor.source_start !== expectedSource) {
      throw new Error("manifest blocks are not contiguous");
    }
    expectedSource += descriptor.source_count;
    expectedEdges += descriptor.edge_count;
    if (expectedSource > manifest.n_neurons) {
      throw new Error("manifest blocks exceed the neuron count");
    }
  }
  if (expectedSource !== manifest.n_neurons) throw new Error("manifest blocks do not cover graph");
  if (expectedEdges !== manifest.n_edges) throw new Error("manifest edge count does not match blocks");
  return manifest;
}

export function parseBlock(data, { nNeurons } = {}) {
  const view = asDataView(data);
  if (view.byteLength < BLOCK_HEADER_BYTES) throw new Error("block is shorter than its header");
  const magic = String.fromCharCode(...new Uint8Array(view.buffer, view.byteOffset, 4));
  if (magic !== "SCB1") throw new Error("invalid block magic");
  if (view.getUint32(4, true) !== 1) throw new Error("unsupported block version");
  const sourceStart = view.getUint32(8, true);
  const sourceCount = view.getUint32(12, true);
  const edgeCount = view.getUint32(16, true);
  const expectedSize = BLOCK_HEADER_BYTES + (sourceCount + 1) * 4 + edgeCount * 8;
  if (view.byteLength !== expectedSize) throw new Error("block byte size does not match header");
  if (sourceStart + sourceCount > (nNeurons ?? UINT32_MAX + 1)) {
    throw new Error("block source range is outside the graph");
  }

  let offset = BLOCK_HEADER_BYTES;
  const rowOffsets = new Uint32Array(sourceCount + 1);
  for (let index = 0; index < rowOffsets.length; index += 1) {
    rowOffsets[index] = view.getUint32(offset, true);
    offset += 4;
  }
  if (rowOffsets[0] !== 0 || rowOffsets[sourceCount] !== edgeCount) {
    throw new Error("CSR offsets do not cover the edge array");
  }
  for (let index = 1; index < rowOffsets.length; index += 1) {
    if (rowOffsets[index - 1] > rowOffsets[index]) throw new Error("CSR offsets are not monotonic");
  }

  const targets = new Uint32Array(edgeCount);
  for (let index = 0; index < edgeCount; index += 1) {
    targets[index] = view.getUint32(offset, true);
    offset += 4;
    if (nNeurons !== undefined && targets[index] >= nNeurons) {
      throw new Error("edge target is outside the graph");
    }
  }
  const weights = new Int32Array(edgeCount);
  for (let index = 0; index < edgeCount; index += 1) {
    weights[index] = view.getInt16(offset, true);
    offset += 2;
  }
  const delays = new Uint32Array(edgeCount);
  for (let index = 0; index < edgeCount; index += 1) {
    delays[index] = view.getUint16(offset, true);
    offset += 2;
    if (delays[index] < 1) throw new Error("edge delay must be positive");
  }
  const sources = new Uint32Array(edgeCount);
  for (let row = 0; row < sourceCount; row += 1) {
    for (let index = rowOffsets[row]; index < rowOffsets[row + 1]; index += 1) {
      sources[index] = sourceStart + row;
    }
  }
  return { sourceStart, sourceCount, edgeCount, sources, targets, weights, delays };
}

async function sha256Hex(bytes) {
  if (!globalThis.crypto?.subtle) throw new Error("WebCrypto SHA-256 is required for checksum verification");
  const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, "0")).join("");
}

export async function openArtifact(baseUrl, { fetchImpl = globalThis.fetch, verifyChecksums = true } = {}) {
  if (typeof fetchImpl !== "function") throw new Error("a fetch implementation is required");
  const base = new URL(baseUrl, globalThis.location?.href ?? "http://localhost/");
  if (!base.href.endsWith("/")) base.pathname += "/";
  const response = await fetchImpl(new URL("manifest.json", base));
  if (!response.ok) throw new Error(`cannot fetch manifest: ${response.status}`);
  const manifest = parseManifest(await response.json());

  async function fetchBlock(descriptor) {
    const response = await fetchImpl(new URL(descriptor.filename, base));
    if (!response.ok) throw new Error(`cannot fetch block: ${response.status}`);
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (bytes.byteLength !== descriptor.byte_size) throw new Error("block byte size mismatch");
    if (verifyChecksums && (await sha256Hex(bytes)) !== descriptor.sha256) {
      throw new Error(`block checksum mismatch: ${descriptor.filename}`);
    }
    const block = parseBlock(bytes, { nNeurons: manifest.n_neurons });
    if (
      block.sourceStart !== descriptor.source_start ||
      block.sourceCount !== descriptor.source_count ||
      block.edgeCount !== descriptor.edge_count
    ) {
      throw new Error(`block descriptor does not match file: ${descriptor.filename}`);
    }
    return { ...block, bytes };
  }

  return {
    manifest,
    async maxDelay() {
      if (Number.isInteger(manifest.max_delay)) return manifest.max_delay;
      let maximum = 0;
      for (const descriptor of manifest.blocks) {
        const block = await fetchBlock(descriptor);
        for (const delay of block.delays) maximum = Math.max(maximum, delay);
      }
      return maximum;
    },
    async *iterBlocks() {
      for (const descriptor of manifest.blocks) yield await fetchBlock(descriptor);
    },
  };
}

function flattenActiveBlock(block, spikeFlags) {
  const sources = [];
  const targets = [];
  const weights = [];
  const delays = [];
  for (let index = 0; index < block.edgeCount; index += 1) {
    if (!spikeFlags[block.sources[index]]) continue;
    sources.push(block.sources[index]);
    targets.push(block.targets[index]);
    weights.push(block.weights[index]);
    delays.push(block.delays[index]);
  }
  return {
    sources: Uint32Array.from(sources),
    targets: Uint32Array.from(targets),
    weights: Int32Array.from(weights),
    delays: Uint32Array.from(delays),
  };
}

function validateConfig(config) {
  requireInteger(config.threshold, "threshold", INT32_MIN, INT32_MAX);
  requireInteger(config.reset, "reset", INT32_MIN, INT32_MAX);
  requireInteger(config.refractory_steps, "refractory steps", 0, UINT16_MAX);
  if (!Array.isArray(config.decay_shifts) || config.decay_shifts.length < 1) {
    throw new Error("decay_shifts must not be empty");
  }
  if (config.decay_shifts.length > MAX_DECAY_SHIFTS) {
    throw new Error(`WebGPU supports at most ${MAX_DECAY_SHIFTS} decay shifts`);
  }
  for (const shift of config.decay_shifts) requireInteger(shift, "decay shift", 0, 31);
}

function configWords(nNeurons, bucketCount, config, currentBucket, edgeCount) {
  const words = new Int32Array(CONFIG_WORDS);
  words[0] = nNeurons;
  words[1] = currentBucket % bucketCount;
  words[2] = bucketCount;
  words[3] = config.threshold;
  words[4] = config.reset;
  words[5] = config.refractory_steps;
  words[6] = config.decay_shifts.length;
  words[7] = edgeCount;
  words.set(config.decay_shifts, 8);
  return words;
}

function browserGPUConstants() {
  if (!globalThis.GPUBufferUsage || !globalThis.GPUShaderStage || !globalThis.GPUMapMode) {
    throw new Error("WebGPU browser globals are unavailable");
  }
  return {
    buffer: globalThis.GPUBufferUsage,
    shaderStage: globalThis.GPUShaderStage,
    mapMode: globalThis.GPUMapMode,
  };
}

function createBuffer(device, bytes, usage) {
  const input = bytesOf(bytes);
  const buffer = device.createBuffer({
    size: Math.max(4, input.byteLength),
    usage,
    mappedAtCreation: true,
  });
  new Uint8Array(buffer.getMappedRange()).set(input);
  buffer.unmap();
  return buffer;
}

function createBindGroup(device, layout, buffers) {
  return device.createBindGroup({
    layout,
    entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
}

function dispatch(device, pipeline, bindGroups, workgroups) {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  bindGroups.forEach((bindGroup, index) => pass.setBindGroup(index, bindGroup));
  pass.dispatchWorkgroups(Math.max(1, workgroups));
  pass.end();
  device.queue.submit([encoder.finish()]);
}

async function readBuffer(device, source, size, constants) {
  const staging = device.createBuffer({
    size: Math.max(4, size),
    usage: constants.buffer.COPY_DST | constants.buffer.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, staging, 0, size);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(constants.mapMode.READ);
  const result = new Uint8Array(staging.getMappedRange(0, size)).slice();
  staging.unmap();
  staging.destroy();
  return result;
}

function readInt32(bytes) {
  return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getInt32(0, true);
}

async function createPipelines(device, constants) {
  const shader = device.createShaderModule({ code: WGSL_LIF_SHADER });
  const layout = (count, group) => device.createBindGroupLayout({
    label: `soup-connectome-webgpu-group-${group}`,
    entries: Array.from({ length: count }, (_, binding) => ({
      binding,
      visibility: constants.shaderStage.COMPUTE,
      buffer: { type: "storage" },
    })),
  });
  const stepLayout = layout(6, 0);
  const scatterLayout = layout(8, 1);
  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [stepLayout, scatterLayout],
  });
  return {
    stepLayout,
    scatterLayout,
    stepPipeline: device.createComputePipeline({
      layout: pipelineLayout,
      compute: { module: shader, entryPoint: "lif_step" },
    }),
    scatterPipeline: device.createComputePipeline({
      layout: pipelineLayout,
      compute: { module: shader, entryPoint: "schedule_edges" },
    }),
  };
}

export async function requestWebGPUDevice() {
  if (!globalThis.navigator?.gpu) throw new Error("WebGPU is unavailable in this browser");
  const adapter = await globalThis.navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("WebGPU adapter request failed");
  return adapter.requestDevice();
}

export async function runStreamed({ device, artifact, config, timesteps, initialPotentials, initialRefractory }) {
  if (!device) throw new Error("a GPUDevice is required");
  if (!artifact?.manifest || typeof artifact.iterBlocks !== "function") {
    throw new Error("a loaded connectome artifact is required");
  }
  requireInteger(timesteps, "timesteps", 0, Number.MAX_SAFE_INTEGER);
  validateConfig(config);
  const constants = browserGPUConstants();
  const nNeurons = artifact.manifest.n_neurons;
  const maxDelay = await artifact.maxDelay();
  requireInteger(maxDelay, "maximum delay", 0, UINT16_MAX);
  const bucketCount = maxDelay + 1;
  const potentials = initialPotentials === undefined
    ? new Int32Array(nNeurons)
    : Int32Array.from(initialPotentials);
  const refractory = initialRefractory === undefined
    ? new Uint32Array(nNeurons)
    : Uint32Array.from(initialRefractory);
  if (potentials.length !== nNeurons || refractory.length !== nNeurons) {
    throw new Error("initial state length does not match graph");
  }
  for (const value of potentials) requireInteger(value, "initial potential", INT32_MIN, INT32_MAX);
  for (const value of refractory) requireInteger(value, "initial refractory", 0, UINT16_MAX);

  const storage = constants.buffer.STORAGE | constants.buffer.COPY_SRC;
  const copyStorage = storage | constants.buffer.COPY_DST;
  const potentialBuffer = createBuffer(device, potentials, storage);
  const refractoryBuffer = createBuffer(device, refractory, storage);
  const spikeBuffer = createBuffer(device, new Uint32Array(nNeurons), storage);
  const arrivalBuffer = createBuffer(device, new Int32Array(bucketCount * nNeurons), storage);
  const configBuffer = createBuffer(device, new Uint8Array(CONFIG_WORDS * 4), copyStorage);
  const errorBuffer = createBuffer(device, new Int32Array(1), copyStorage);
  const pipelines = await createPipelines(device, constants);
  const stepGroup = createBindGroup(device, pipelines.stepLayout, [
    potentialBuffer, refractoryBuffer, spikeBuffer, arrivalBuffer, configBuffer, errorBuffer,
  ]);
  const dummyEdge = createBuffer(device, new Uint32Array(1), storage);
  const dummyWeight = createBuffer(device, new Int32Array(1), storage);
  const dummyConfig = createBuffer(device, new Uint8Array(CONFIG_WORDS * 4), storage);
  const dummyScatterGroup = createBindGroup(device, pipelines.scatterLayout, [
    dummyEdge, dummyEdge, dummyWeight, dummyEdge, spikeBuffer, arrivalBuffer, dummyConfig, errorBuffer,
  ]);

  const spikes = [];
  for (let currentBucket = 0; currentBucket < timesteps; currentBucket += 1) {
    device.queue.writeBuffer(configBuffer, 0, configWords(nNeurons, bucketCount, config, currentBucket, 0));
    device.queue.writeBuffer(errorBuffer, 0, new Int32Array(1));
    dispatch(device, pipelines.stepPipeline, [stepGroup, dummyScatterGroup],
      Math.ceil(nNeurons / WORKGROUP_SIZE));
    if (readInt32(await readBuffer(device, errorBuffer, 4, constants))) {
      throw new Error("webgpu fixed-point operation exceeded int32 range");
    }
    const spikeBytes = await readBuffer(device, spikeBuffer, nNeurons * 4, constants);
    const spikeView = new DataView(spikeBytes.buffer, spikeBytes.byteOffset, spikeBytes.byteLength);
    const spikeFlags = Array.from({ length: nNeurons }, (_, index) => spikeView.getUint32(index * 4, true) !== 0);
    spikes.push(spikeFlags);

    for await (const block of artifact.iterBlocks()) {
      const active = flattenActiveBlock(block, spikeFlags);
      if (active.sources.length === 0) continue;
      const edgeCount = active.sources.length;
      const edgeSources = createBuffer(device, active.sources, storage);
      const edgeTargets = createBuffer(device, active.targets, storage);
      const edgeWeights = createBuffer(device, active.weights, storage);
      const edgeDelays = createBuffer(device, active.delays, storage);
      const blockConfig = createBuffer(
        device,
        configWords(nNeurons, bucketCount, config, currentBucket, edgeCount),
        storage,
      );
      const scatterGroup = createBindGroup(device, pipelines.scatterLayout, [
        edgeSources, edgeTargets, edgeWeights, edgeDelays,
        spikeBuffer, arrivalBuffer, blockConfig, errorBuffer,
      ]);
      device.queue.writeBuffer(errorBuffer, 0, new Int32Array(1));
      dispatch(device, pipelines.scatterPipeline, [stepGroup, scatterGroup],
        Math.ceil(edgeCount / WORKGROUP_SIZE));
      const error = readInt32(await readBuffer(device, errorBuffer, 4, constants));
      edgeSources.destroy();
      edgeTargets.destroy();
      edgeWeights.destroy();
      edgeDelays.destroy();
      blockConfig.destroy();
      if (error) throw new Error("webgpu fixed-point operation exceeded int32 range");
    }
  }

  const potentialBytes = await readBuffer(device, potentialBuffer, nNeurons * 4, constants);
  const refractoryBytes = await readBuffer(device, refractoryBuffer, nNeurons * 4, constants);
  const potentialView = new DataView(potentialBytes.buffer, potentialBytes.byteOffset, potentialBytes.byteLength);
  const refractoryView = new DataView(refractoryBytes.buffer, refractoryBytes.byteOffset, refractoryBytes.byteLength);
  const finalPotentials = Array.from({ length: nNeurons }, (_, index) => potentialView.getInt32(index * 4, true));
  const finalRefractory = Array.from({ length: nNeurons }, (_, index) => refractoryView.getUint32(index * 4, true));
  [potentialBuffer, refractoryBuffer, spikeBuffer, arrivalBuffer, configBuffer, errorBuffer,
    dummyEdge, dummyWeight, dummyConfig].forEach((buffer) => buffer.destroy());
  return { spikes, finalPotentials, finalRefractory };
}

export async function runWasmStreamed({
  Runtime,
  artifact,
  config,
  timesteps,
  initialPotentials,
  initialRefractory,
}) {
  if (typeof Runtime !== "function") throw new Error("a wasm Runtime constructor is required");
  if (!artifact?.manifest || typeof artifact.iterBlocks !== "function") {
    throw new Error("a loaded connectome artifact is required");
  }
  requireInteger(timesteps, "timesteps", 0, Number.MAX_SAFE_INTEGER);
  validateConfig(config);
  const nNeurons = artifact.manifest.n_neurons;
  const maxDelay = await artifact.maxDelay();
  requireInteger(maxDelay, "maximum delay", 0, UINT16_MAX);
  const potentials = initialPotentials === undefined
    ? new Int32Array(nNeurons)
    : Int32Array.from(initialPotentials);
  const refractory = initialRefractory === undefined
    ? new Uint32Array(nNeurons)
    : Uint32Array.from(initialRefractory);
  if (potentials.length !== nNeurons || refractory.length !== nNeurons) {
    throw new Error("initial state length does not match graph");
  }
  const runtime = new Runtime(
    nNeurons,
    maxDelay,
    config.threshold,
    config.reset,
    Uint32Array.from(config.decay_shifts),
    config.refractory_steps,
  );
  runtime.set_initial_state(potentials, refractory);
  const spikes = [];
  for (let timestep = 0; timestep < timesteps; timestep += 1) {
    spikes.push([...runtime.begin_step()].map((value) => value !== 0));
    for await (const block of artifact.iterBlocks()) runtime.schedule_block(block.bytes);
    runtime.finish_step();
  }
  return {
    spikes,
    finalPotentials: [...runtime.final_potentials()],
    finalRefractory: [...runtime.final_refractory()],
  };
}
