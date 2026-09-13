import { chromium } from "playwright";

const executablePath = process.env.SOUP_CONNECTOME_E2E_EXECUTABLE;
if (!executablePath) {
  throw new Error("SOUP_CONNECTOME_E2E_EXECUTABLE must point to an installed Chrome/Chromium");
}

const browser = await chromium.launch({
  headless: true,
  executablePath,
  args: [
    "--enable-unsafe-webgpu",
    "--enable-features=WebGPU",
    "--use-angle=d3d11",
  ],
});

try {
  const page = await browser.newPage();
  await page.goto("http://127.0.0.1:8765/test/e2e.html");
  await page.waitForFunction(
    () => document.title.startsWith("PASS") || document.title.startsWith("FAIL") || document.title.startsWith("ERROR"),
    null,
    { timeout: 30_000 },
  );
  const title = await page.title();
  const result = JSON.parse(await page.locator("#result").textContent());
  if (result.status !== "PASS") {
    throw new Error(`${title}: ${JSON.stringify(result)}`);
  }
  console.log(`measured browser=${executablePath} status=PASS`);
} finally {
  await browser.close();
}
