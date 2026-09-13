import assert from "node:assert/strict";
import test from "node:test";

import {
  WGSL_LIF_SHADER,
  parseBlock,
  parseManifest,
  runWasmStreamed,
} from "../src/index.js";

function exampleBlock() {
  const bytes = new ArrayBuffer(20 + 3 * 4 + 2 * 4 + 2 * 2 + 2 * 2);
  const view = new DataView(bytes);
  new Uint8Array(bytes, 0, 4).set([0x53, 0x43, 0x42, 0x31]);
  view.setUint32(4, 1, true);
  view.setUint32(8, 4, true);
  view.setUint32(12, 2, true);
  view.setUint32(16, 2, true);
  let offset = 20;
  [0, 1, 2].forEach((value) => { view.setUint32(offset, value, true); offset += 4; });
  [7, 6].forEach((value) => { view.setUint32(offset, value, true); offset += 4; });
  [1, -2].forEach((value) => { view.setInt16(offset, value, true); offset += 2; });
  [1, 2].forEach((value) => { view.setUint16(offset, value, true); offset += 2; });
  return bytes;
}

test("parses little-endian SCB CSR blocks", () => {
  const block = parseBlock(exampleBlock(), { nNeurons: 8 });
  assert.equal(block.sourceStart, 4);
  assert.deepEqual([...block.sources], [4, 5]);
  assert.deepEqual([...block.targets], [7, 6]);
  assert.deepEqual([...block.weights], [1, -2]);
  assert.deepEqual([...block.delays], [1, 2]);
});

test("rejects malformed manifest block coverage", () => {
  assert.throws(() => parseManifest({
    format_id: "scx",
    format_version: 1,
    n_neurons: 2,
    n_edges: 0,
    blocks: [],
  }), /do not cover/);
});

test("rejects a manifest edge-count mismatch", () => {
  assert.throws(() => parseManifest({
    format_id: "scx",
    format_version: 1,
    n_neurons: 0,
    n_edges: 1,
    blocks: [],
  }), /edge count/);
});

test("WGSL stays on integer atomics", () => {
  assert.match(WGSL_LIF_SHADER, /atomicCompareExchangeWeak/);
  assert.doesNotMatch(WGSL_LIF_SHADER, /i64|f32/);
});

test("streams raw blocks through the WASM runtime boundary", async () => {
  const calls = [];
  class FakeRuntime {
    constructor(...args) { calls.push(["new", args]); }
    set_initial_state() { calls.push(["initial"]); }
    begin_step() { calls.push(["begin"]); return Uint8Array.from([1, 0]); }
    schedule_block(bytes) { calls.push(["block", [...bytes]]); }
    finish_step() { calls.push(["finish"]); }
    final_potentials() { return Int32Array.from([0, 0]); }
    final_refractory() { return Uint32Array.from([0, 0]); }
  }
  const artifact = {
    manifest: { n_neurons: 2 },
    maxDelay: async () => 1,
    async *iterBlocks() { yield { bytes: Uint8Array.from([7, 8]) }; },
  };
  const result = await runWasmStreamed({
    Runtime: FakeRuntime,
    artifact,
    config: { threshold: 4, reset: 0, decay_shifts: [1], refractory_steps: 0 },
    timesteps: 1,
  });
  assert.deepEqual(result.spikes, [[true, false]]);
  assert.deepEqual(calls.map(([name]) => name), ["new", "initial", "begin", "block", "finish"]);
  assert.deepEqual(calls[3][1], [7, 8]);
});
