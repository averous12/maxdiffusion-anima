/**
 * Fill the SnapRef panel end to end in the REAL Gradio app and screenshot it:
 * tick "Use SnapRef", open the Reference(s) dropdown container, pick the reference,
 * then leave the panel filled in for the shot.
 *
 * usage: node snapref_ui_fill.js <ui_base> <ref_stem> <out_png>
 */
const fs = require('fs');
const path = require('path');
const NM = 'C:/Users/avero/AppData/Local/hermes/hermes-agent/node_modules';
const ROOT = 'E:/Workspace/colab/snapref_local/content';

async function main() {
  const [base, stem, outPng] = process.argv.slice(2);
  const { chromium } = require(path.join(NM, 'playwright'));
  const out = [];
  const say = (...a) => out.push('[fill] ' + a.join(' '));
  const browser = await chromium.launch({
    executablePath: 'C:/Program Files/imput/Helium/Application/chrome.exe',
    headless: true, args: ['--no-sandbox', '--disable-gpu'],
  });
  const page = await browser.newPage({ viewport: { width: 1500, height: 1300 } });
  await page.goto(base, { waitUntil: 'networkidle', timeout: 180000 });
  await page.waitForTimeout(6000);
  const head = page.getByText('SnapRef — character reference (reference adapter)', { exact: false }).first();
  await head.scrollIntoViewIfNeeded(); await head.click(); await page.waitForTimeout(1500);

  // 1. tick the checkbox
  await page.evaluate(() => {
    const c = [...document.querySelectorAll('input[type=checkbox]')].find(
      (e) => ((e.closest('label') || e.parentElement || {}).textContent || '').includes('Use SnapRef'));
    if (c) c.setAttribute('data-cb', '1');
  });
  await page.locator('[data-cb="1"]').click({ force: true });
  await page.waitForTimeout(900);
  say('checked =', await page.locator('[data-cb="1"]').isChecked());

  // 2. open the reference dropdown by clicking its container (the input alone does not open it)
  const opened = await page.evaluate(() => {
    const ins = [...document.querySelectorAll('input.border-none')].filter((i) => !i.value);
    const inp = ins[ins.length - 1] || ins[0];
    if (!inp) return { ok: false };
    const box = inp.closest('.wrap') || inp.parentElement.parentElement || inp.parentElement;
    box.setAttribute('data-ddbox', '1');
    inp.setAttribute('data-ddinput', '1');
    inp.focus();
    return { ok: true, html: box.outerHTML.replace(/\s+/g, ' ').slice(0, 200) };
  });
  say('dropdown container:', JSON.stringify(opened));
  await page.locator('[data-ddbox="1"]').click({ position: { x: 40, y: 12 } }).catch((e) => say('container click:', e.message.split('\n')[0]));
  await page.waitForTimeout(1200);

  let li = await page.evaluate(() => [...document.querySelectorAll('li, [role=option]')]
    .map((e) => e.textContent.trim()).filter(Boolean).slice(0, 20));
  say('options right after open:', JSON.stringify(li));

  if (!li.length) {
    await page.locator('[data-ddinput="1"]').click();
    await page.waitForTimeout(800);
    await page.keyboard.type(stem, { delay: 70 });
    await page.waitForTimeout(1600);
    li = await page.evaluate(() => [...document.querySelectorAll('li, [role=option]')]
      .map((e) => e.textContent.trim()).filter(Boolean).slice(0, 20));
    say('options after typing:', JSON.stringify(li));
  }

  const opt = page.locator('li, [role=option]').filter({ hasText: stem }).first();
  if (await opt.count()) {
    await opt.click();
    await page.waitForTimeout(1500);
    const st = await page.evaluate(() => {
      const inp = document.querySelector('[data-ddinput="1"]');
      let n = inp;
      for (let i = 0; i < 3 && n && n.parentElement; i++) n = n.parentElement;
      return { val: inp && inp.value, text: n ? n.textContent.replace(/\s+/g, ' ').slice(0, 200) : '' };
    });
    say('selected value:', JSON.stringify(st.val), 'block:', st.text);
  } else { say('NO OPTION MATCHED'); }
  await page.keyboard.press('Escape');
  await page.waitForTimeout(600);

  // 3. scene-only prompt
  await page.locator('textarea').first().fill(
    'masterpiece, best quality, 1girl, solo, upper body, looking at viewer, simple background, white background');
  await page.waitForTimeout(600);
  await page.screenshot({ path: outPng, fullPage: true });
  say('shot', outPng);
  fs.writeFileSync(path.join(ROOT, 'ui_fill.log'), out.join('\n') + '\n');
  await browser.close();
}
main().catch((e) => { fs.appendFileSync(path.join(ROOT, 'ui_fill.log'), 'FAILED ' + (e.stack || e) + '\n'); process.exit(1); });