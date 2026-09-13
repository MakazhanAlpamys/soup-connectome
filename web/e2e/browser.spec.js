import { expect, test } from "@playwright/test";

test.use({
  headless: true,
  launchOptions: {
    args: [
      "--enable-unsafe-webgpu",
      "--enable-features=Vulkan,WebGPU",
      "--use-angle=vulkan",
    ],
  },
});

test("browser WebGPU and WASM agree on the streamed fixture", async ({ page }) => {
  await page.goto("http://127.0.0.1:8765/test/e2e.html");
  const result = page.locator("#result");
  await expect(result).not.toHaveText("running…", { timeout: 30_000 });
  const text = await result.textContent();
  if (text?.includes('"status": "ERROR"') && text.includes("WebGPU adapter request failed")) {
    test.skip(true, "Chromium has no WebGPU adapter in this environment");
  }
  expect(text).toContain('"status": "PASS"');
  expect(text).toContain('"webgpu"');
  expect(text).toContain('"wasm"');
});
