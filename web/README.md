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

`npm test` runs parser and shader-contract tests in Node. Browser GPU execution
requires a browser with WebGPU enabled and is not exercised by the Node suite;
throughput, browser compatibility, and WASM packaging are `not tested`.
