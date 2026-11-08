/**
 * UI proof: open the real Gradio app, confirm the SnapRef panel renders, fill it in and
 * generate, then screenshot the result. Uses Playwright with the local Helium binary.
 *
 * usage: node snapref_ui_proof.js <ui_base_url> <ref_name> <shot_prefix> [adapter]
 */
const path = require('path');
const NM = 'C:/Users/avero/AppData/Local/hermes/hermes-agent/node_modules';

async function main() {
  const [base, refName, shotPrefix, adapter] = process.argv.slice(2);
  const { chromium } = require(path.join(NM, 'playwright'));
  const browser = await chromium.launch({
    executablePath: 'C:/Program Files/imput/Helium/Application/chrome.exe',
    headless: true,
    args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage'],
  });
  const page = await browser.newPage({ viewport: { width: 1500, height: 1250 } });
  const log = (...a) => console.log('[ui]', ...a);

  await page.goto(base, { waitUntil: 'networkidle', timeout: 180000 });
  await page.waitForTimeout(6000);
  log('title:', await page.title());

  // open the SnapRef accordion by its summary text
  const head = page.getByText('SnapRef — character reference (reference adapter)', { exact: false }).first();
  await head.scrollIntoViewIfNeeded();
  await head.click();
  await page.waitForTimeout(1500);

  const snap = `${shotPrefix}_panel.png`;
  const panel = page.locator('.accordion, [data-testid="accordion"], .gradio-accordion').filter({ hasText: 'SnapRef' }).first();
  if (await panel.count()) { await panel.screenshot({ path: snap }); }
  else { await page.screenshot({ path: snap, fullPage: true }); }
  log('panel shot:', snap);

  // tick "Use SnapRef"
  const cb = page.getByLabel('Use SnapRef', { exact: false }).first();
  await cb.check();
  await page.waitForTimeout(500);

  // pick the reference picture in the multi-select dropdown
  const refBox = page.locator('label', { hasText: 'Reference(s)' }).first();
  const scope = page.getByTestId('snapref-ref').or(page.locator('.gr-box, input[role="listbox"]')).first();
  try {
    await page.locator('input[role="listbox"]').last().click();
    await page.waitForTimeout(800);
    await page.getByRole('option', { name: refName, exact: false }).first().click();
    await page.waitForTimeout(600);
    log('picked reference', refName);
  } catch (e) {
    log('reference dropdown fallback:', e.message.split('\n')[0]);
    await page.keyboard.press('Escape');
  }

  // fill the prompt with a scene-only prompt, per the adapter's documented usage
  const promptBox = page.locator('textarea').first();
  await promptBox.fill('masterpiece, best quality, 1girl, solo, upper body, "looking at viewer", simple background, white background');
  await page.waitForTimeout(400);

  const shotFilled = `${shotPrefix}_filled.png`;
  await page.screenshot({ path: shotFilled, fullPage: true });
  log('filled shot:', shotFilled);

  // generate
  const gen = page.getByRole('button', { name: /^Generate$/ }).first();
  await gen.click();
  log('clicked Generate; waiting for the result ...');
  let done = false;
  for (let i = 0; i < 500; i++) {
    await page.waitForTimeout(2000);
    const status = await page.locator('#anima-status textarea, #anima-status').first().inputValue().catch(() => '');
    if (i % 15 === 0) log(`status: ${String(status).slice(0, 120)}`);
    if (/done in|error/i.test(String(status))) { done = true; log('final status:', status); break; }
  }
  await page.waitForTimeout(2500);
  const shotDone = `${shotPrefix}_result.png`;
  await page.screenshot({ path: shotDone, fullPage: true });
  log('result shot:', shotDone, 'completed:', done);
  await browser.close();
  if (!done) process.exitCode = 3;
}

main().catch((e) => { console.error('UI PROOF FAILED:', e); process.exit(1); });
