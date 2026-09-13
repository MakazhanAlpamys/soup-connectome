# Browser WebGPU host

This directory contains a dependency-free ES module for browser-side streamed
execution of `.scx` artifacts. It uses the same fixed-point WGSL kernels as the
Python WebGPU backend and keeps neuron state plus the delay line resident on the
`GPUDevice`; one validated source block is fetched and uploaded at a time.

The host is intentionally a low-level runtime boundary, not a demo UI. A page
can use it like this:

```js
import { openArtifact, requestWebGPUDevice, runStreamed } from "./src/index.js";

const artifact = await openArtifact("/connectome/example.scx/");
const device = await requestWebGPUDevice();
const result = await runStreamed({
  device,
  artifact,
  config: {
    threshold: 20000,
    reset: 0,
    decay_shifts: [2],
    refractory_steps: 2,
  },
  timesteps: 4,
});
```

The same artifact can use the optional WASM CPU runtime. After loading the
`wasm-pack --target web` package and calling its default initializer, pass its
`ConnectomeRuntime` constructor to `runWasmStreamed`; the host sends each raw
block directly to WASM and retains no complete graph copy in JavaScript.

`npm test` runs parser and shader-contract tests in Node. The optional
`web/wasm` crate provides a small wasm-bindgen validation boundary; build it
with `wasm-pack build --target web --out-dir pkg --release` from that directory.
Browser GPU execution requires a browser with WebGPU enabled and is not
exercised by the Node suite; throughput and browser compatibility are `not
tested`.
