/**
 * FULL UI proof: real Gradio app -> tick Use SnapRef -> pick reference in the app's own
 * dropdown -> scene-only prompt -> click Generate -> wait for the server's progress file to
 * reach "done" -> screenshot the app showing the finished image.
 *
 * Also reads back anima_request.json to prove the UI built the snapref request itself.
 *
 * usage: node snapref_ui_full.js <ui_base> <ref_stem> <out_prefix>
 */
const fs = require('fs');
const path = require('path');
const NM = 'C:/Users/avero/AppData/Local/hermes/hermes-agent/node_modules';
const ROOT = 'E:/Workspace/colab/snapref_local/content';
const REQ = path.join(ROOT, 'anima_request.json');
const PROG = path.join(ROOT, 'anima_progress.json');

const readJ = (p) => { try { return JSON.parse(fs.readFileSync(p, 'utf8')); } catch (e) { return null; } };

async function main() {
  const [base, stem, prefix] = process.argv.slice(2);
  const { chromium } = require(path.join(NM, 'playwright'));
  const out = [];
  const say = (...a) => { const l = '[full] ' + a.join(' '); out.push(l); fs.appendFileSync(path.join(ROOT, 'ui_full.log'), l + '\n'); };
  fs.writeFileSync(path.join(ROOT, 'ui_full.log'), '');

  const browser = await chromium.launch({
    executablePath: 'C:/Program Files/imput/Helium/Application/chrome.exe',
    headless: true, args: ['--no-sandbox', '--disable-gpu'],
  });
  const page = await browser.newPage({ viewport: { width: 1500, height: 1300 } });
  await page.goto(base, { waitUntil: 'networkidle', timeout: 180000 });
  await page.waitForTimeout(6000);
  const head = page.getByText('SnapRef — character reference (reference adapter)', { exact: false }).first();
  await head.scrollIntoViewIfNeeded(); await head.click(); await page.waitForTimeout(1500);

  // tick Use SnapRef
  await page.evaluate(() => {
    const c = [...document.querySelectorAll('input[type=checkbox]')].find(
      (e) => ((e.closest('label') || e.parentElement || {}).textContent || '').includes('Use SnapRef'));
    if (c) c.setAttribute('data-cb', '1');
  });
  await page.locator('[data-cb="1"]').click({ force: true });
  await page.waitForTimeout(900);
  say('Use SnapRef checked =', await page.locator('[data-cb="1"]').isChecked());

  // open the reference dropdown by its container and pick
  await page.evaluate(() => {
    const ins = [...document.querySelectorAll('input.border-none')].filter((i) => !i.value);
    const inp = ins[ins.length - 1] || ins[0];
    const box = inp.closest('.wrap') || inp.parentElement.parentElement;
    box.setAttribute('data-ddbox', '1');
  });
  await page.locator('[data-ddbox="1"]').click({ position: { x: 40, y: 12 } });
  await page.waitForTimeout(1200);
  const opts = await page.evaluate(() => [...document.querySelectorAll('li, [role=option]')]
    .map((e) => e.textContent.trim()).filter(Boolean).slice(0, 20));
  say('dropdown options:', JSON.stringify(opts));
  const opt = page.locator('li, [role=option]').filter({ hasText: stem }).first();
  if (!(await opt.count())) throw new Error('reference option not offered');
  await opt.click();
  await page.waitForTimeout(1400);
  await page.keyboard.press('Escape');
  await page.waitForTimeout(500);
  say('picked reference:', stem);

  // scene-only prompt
  await page.locator('textarea').first().fill(
    'masterpiece, best quality, 1girl, solo, upper body, looking at viewer, simple background, white background');
  await page.waitForTimeout(500);

  // click Generate
  const before = readJ(REQ);
  await page.evaluate(() => {
    const b = [...document.querySelectorAll('button')].find((e) => (e.textContent || '').trim() === 'Generate');
    if (b) b.setAttribute('data-gen', '1');
  });
  await page.locator('[data-gen="1"]').click();
  say('clicked Generate');

  // the UI's own request is the proof of the wiring
  let req = null;
  for (let i = 0; i < 40; i++) {
    await page.waitForTimeout(1000);
    req = readJ(REQ);
    if (req && (!before || req.gen_id !== before.gen_id)) break;
  }
  if (!req) throw new Error('request file never appeared');
  say('REQUEST gen_id', req.gen_id, 'go', req.go);
  say('REQUEST snapref', JSON.stringify(req.snapref));
  say('REQUEST prompt', JSON.stringify(req.prompt).slice(0, 110));

  await page.screenshot({ path: `${prefix}_panel.png`, fullPage: true });
  say('panel shot written');

  // wait on the server's progress file (the 1 Hz UI timer masks the status box)
  let last = '';
  let prog = readJ(PROG) || {};
  for (let i = 0; i < 900; i++) {
    await page.waitForTimeout(2000);
    prog = readJ(PROG) || {};
    if (prog.gen_id !== req.gen_id) continue;
    const key = `${prog.stage} ${prog.step}/${prog.steps}`;
    if (key !== last) { last = key; say('server:', key, prog.note || ''); }
    if (prog.stage === 'done' || prog.stage === 'error') break;
  }
  say('final stage:', prog.stage, 'snapref:', JSON.stringify(prog.snapref || null).slice(0, 200));

  await page.waitForTimeout(9000);   // let the timer pull the image in
  await page.screenshot({ path: `${prefix}_result.png`, fullPage: true });
  say('result shot written');
  const gal = page.locator('#anima-result').first();
  if (await gal.count()) { await gal.screenshot({ path: `${prefix}_image.png` }).catch(() => {}); say('image crop written'); }
  await browser.close();
  say('FULL_DONE', prog.stage);
}
main().catch((e) => {
  fs.appendFileSync(path.join(ROOT, 'ui_full.log'), 'FAILED ' + (e.stack || e) + '\n');
  process.exit(1);
});