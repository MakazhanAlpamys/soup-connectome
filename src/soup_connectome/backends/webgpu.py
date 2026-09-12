"""Optional WebGPU backend with canonical WGSL integer kernels.

The Python adapter uses ``wgpu`` when available. The WGSL source is deliberately
kept as a public constant so the same kernels can be embedded by a browser or
WASM host later without changing the fixed-point contract.
"""

from __future__ import annotations

import struct
from typing import Any

from soup_connectome.config import Residency, SimulationConfig
from soup_connectome.errors import (
    BackendNotImplementedError,
    BackendUnavailableError,
    ConfigurationError,
    FixedPointOverflowError,
    GraphValidationError,
)
from soup_connectome.graph.format import ConnectomeGraph
from soup_connectome.sim.runtime import GraphSource, SimulationResult

INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1
UINT16_MAX = 2**16 - 1
WORKGROUP_SIZE = 64
MAX_DECAY_SHIFTS = 8
CONFIG_WORDS = 16

WGSL_LIF_SHADER = r"""
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
"""


def availability() -> tuple[bool, str]:
    """Report whether a wgpu device can be created on this host."""

    try:
        import wgpu.utils

        device = wgpu.utils.get_default_device()
        info = device.adapter.info
    except ImportError:
        return False, "WebGPU backend requires the optional wgpu dependency"
    except Exception as exc:  # pragma: no cover - depends on local adapter state
        return False, f"WebGPU device initialization failed: {exc}"
    return True, f"WebGPU device is available: {info.get('description', 'unknown adapter')}"


def _require_wgpu() -> tuple[Any, Any]:
    try:
        import wgpu
        import wgpu.utils
    except ImportError as exc:
        raise BackendUnavailableError(
            "WebGPU backend requires the optional wgpu dependency"
        ) from exc
    return wgpu, wgpu.utils


def _pack_i32(values: list[int]) -> bytes:
    return struct.pack(f"<{len(values)}i", *values)


def _pack_u32(values: list[int]) -> bytes:
    return struct.pack(f"<{len(values)}I", *values)


def _buffer(device: Any, wgpu: Any, data: bytes, usage: Any) -> Any:
    return device.create_buffer_with_data(data=data or b"\0\0\0\0", usage=usage)


def _flatten_graph(graph: ConnectomeGraph) -> tuple[list[int], list[int], list[int], list[int]]:
    sources: list[int] = []
    targets: list[int] = []
    weights: list[int] = []
    delays: list[int] = []
    for block in graph.blocks:
        for local_source, row in enumerate(block.rows):
            source = block.source_start + local_source
            for edge in row:
                sources.append(source)
                targets.append(edge.target)
                weights.append(edge.weight)
                delays.append(edge.delay)
    return sources, targets, weights, delays


def _layout(
    device: Any,
    wgpu: Any,
    entries: int,
    group: int,
    read_only_bindings: set[int],
) -> Any:
    return device.create_bind_group_layout(
        label=f"soup-connectome-webgpu-group-{group}",
        entries=[
            {
                "binding": binding,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": "read-only-storage" if binding in read_only_bindings else "storage"
                },
            }
            for binding in range(entries)
        ],
    )


def _bind_group(device: Any, layout: Any, buffers: list[Any]) -> Any:
    return device.create_bind_group(
        layout=layout,
        entries=[
            {"binding": binding, "resource": {"buffer": buffer}}
            for binding, buffer in enumerate(buffers)
        ],
    )


def _dispatch(device: Any, pipeline: Any, bind_groups: list[Any], workgroups: int) -> None:
    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    for index, bind_group in enumerate(bind_groups):
        compute_pass.set_bind_group(index, bind_group)
    compute_pass.dispatch_workgroups(max(1, workgroups))
    compute_pass.end()
    device.queue.submit([encoder.finish()])


def _run_resident(
    wgpu: Any,
    device: Any,
    graph: ConnectomeGraph,
    config: SimulationConfig,
    *,
    timesteps: int,
    initial_potentials: tuple[int, ...] | None,
    initial_refractory: tuple[int, ...] | None,
) -> SimulationResult:
    potentials = initial_potentials or (0,) * graph.n_neurons
    refractory = initial_refractory or (0,) * graph.n_neurons
    if len(potentials) != graph.n_neurons or len(refractory) != graph.n_neurons:
        raise GraphValidationError("initial state length does not match graph")
    if any(not INT32_MIN <= value <= INT32_MAX for value in potentials):
        raise FixedPointOverflowError("initial potential exceeds signed int32 range")
    if any(value < 0 or value > UINT16_MAX for value in refractory):
        raise ValueError("initial refractory counter is outside uint16 range")
    if len(config.decay_shifts) > MAX_DECAY_SHIFTS:
        raise ConfigurationError(f"WebGPU supports at most {MAX_DECAY_SHIFTS} decay shifts")

    sources, targets, weights, delays = _flatten_graph(graph)
    neuron_count = graph.n_neurons
    edge_count = len(sources)
    if neuron_count > INT32_MAX or edge_count > INT32_MAX:
        raise ConfigurationError("WebGPU configuration counts must fit signed int32")
    bucket_count = graph.max_delay + 1
    usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
    copy_usage = usage | wgpu.BufferUsage.COPY_DST
    potential_buffer = _buffer(device, wgpu, _pack_i32(list(potentials)), usage)
    refractory_buffer = _buffer(device, wgpu, _pack_u32(list(refractory)), usage)
    spike_buffer = _buffer(device, wgpu, _pack_u32([0] * neuron_count), usage)
    arrival_buffer = _buffer(
        device,
        wgpu,
        _pack_i32([0] * (bucket_count * neuron_count)),
        wgpu.BufferUsage.STORAGE,
    )
    config_buffer = _buffer(device, wgpu, b"\0" * (CONFIG_WORDS * 4), copy_usage)
    error_buffer = _buffer(device, wgpu, _pack_i32([0]), copy_usage)
    source_buffer = _buffer(device, wgpu, _pack_u32(sources), usage)
    target_buffer = _buffer(device, wgpu, _pack_u32(targets), usage)
    weight_buffer = _buffer(device, wgpu, _pack_i32(weights), usage)
    delay_buffer = _buffer(device, wgpu, _pack_u32(delays), usage)

    shader = device.create_shader_module(code=WGSL_LIF_SHADER)
    step_layout = _layout(device, wgpu, 6, 0, set())
    scatter_layout = _layout(device, wgpu, 8, 1, set())
    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[step_layout, scatter_layout]
    )
    step_pipeline = device.create_compute_pipeline(
        layout=pipeline_layout,
        compute={"module": shader, "entry_point": "lif_step"},
    )
    scatter_pipeline = device.create_compute_pipeline(
        layout=pipeline_layout,
        compute={"module": shader, "entry_point": "schedule_edges"},
    )
    step_group = _bind_group(
        device,
        step_layout,
        [
            potential_buffer,
            refractory_buffer,
            spike_buffer,
            arrival_buffer,
            config_buffer,
            error_buffer,
        ],
    )
    scatter_group = _bind_group(
        device,
        scatter_layout,
        [
            source_buffer,
            target_buffer,
            weight_buffer,
            delay_buffer,
            spike_buffer,
            arrival_buffer,
            config_buffer,
            error_buffer,
        ],
    )

    spike_history: list[tuple[bool, ...]] = []
    for current_bucket in range(timesteps):
        values = [
            neuron_count,
            current_bucket % bucket_count,
            bucket_count,
            config.threshold,
            config.reset,
            config.refractory_steps,
            len(config.decay_shifts),
            edge_count,
            *config.decay_shifts,
            *([0] * (CONFIG_WORDS - 8 - len(config.decay_shifts))),
        ]
        device.queue.write_buffer(config_buffer, 0, _pack_i32(values))
        device.queue.write_buffer(error_buffer, 0, _pack_i32([0]))
        bind_groups = [step_group, scatter_group]
        _dispatch(
            device,
            step_pipeline,
            bind_groups,
            (neuron_count + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE,
        )
        _dispatch(
            device,
            scatter_pipeline,
            bind_groups,
            (edge_count + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE,
        )

        error = struct.unpack("<i", bytes(device.queue.read_buffer(error_buffer))[:4])[0]
        if error:
            raise FixedPointOverflowError("webgpu fixed-point operation exceeded int32 range")
        raw_spikes = bytes(device.queue.read_buffer(spike_buffer))
        spike_values = (
            struct.unpack(f"<{neuron_count}I", raw_spikes[: neuron_count * 4])
            if neuron_count
            else ()
        )
        spike_history.append(tuple(value != 0 for value in spike_values))

    raw_potentials = bytes(device.queue.read_buffer(potential_buffer))
    raw_refractory = bytes(device.queue.read_buffer(refractory_buffer))
    final_potentials = (
        struct.unpack(f"<{neuron_count}i", raw_potentials[: neuron_count * 4])
        if neuron_count
        else ()
    )
    final_refractory = (
        struct.unpack(f"<{neuron_count}I", raw_refractory[: neuron_count * 4])
        if neuron_count
        else ()
    )
    return SimulationResult(
        spikes=tuple(spike_history),
        final_potentials=tuple(final_potentials),
        final_refractory=tuple(final_refractory),
    )


class WebGPUBackend:
    name = "webgpu"

    def __init__(self) -> None:
        self.available, self.reason = availability()

    def run(
        self,
        graph: GraphSource,
        config: SimulationConfig,
        *,
        timesteps: int,
        initial_potentials: tuple[int, ...] | None = None,
        initial_refractory: tuple[int, ...] | None = None,
        residency: Residency | str = Residency.resident,
    ) -> SimulationResult:
        if not self.available:
            raise BackendUnavailableError(self.reason)
        if timesteps < 0:
            raise ValueError("timesteps cannot be negative")
        if Residency(residency) is Residency.streamed:
            raise BackendNotImplementedError(
                "WebGPU streamed block dispatch is not implemented; use resident residency"
            )
        if not isinstance(graph, ConnectomeGraph):
            raise GraphValidationError("WebGPU resident execution requires a materialized graph")
        _wgpu, utils = _require_wgpu()
        device = utils.get_default_device()
        try:
            return _run_resident(
                _wgpu,
                device,
                graph,
                config,
                timesteps=timesteps,
                initial_potentials=initial_potentials,
                initial_refractory=initial_refractory,
            )
        except (
            BackendUnavailableError,
            ConfigurationError,
            FixedPointOverflowError,
            GraphValidationError,
            ValueError,
        ):
            raise
        except Exception as exc:  # pragma: no cover - adapter/driver-specific failures
            raise BackendUnavailableError(f"WebGPU execution failed on this host: {exc}") from exc
