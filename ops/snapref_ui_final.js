/**
 * FINAL SnapRef proof through the real Gradio UI + real server.
 *
 * 1. open the app, expand the SnapRef accordion
 * 2. tick "Use SnapRef", pick the reference in the app's own dropdown
 * 3. set Custom size 512x512 and 4 steps (CPU-tractable, same code path)
 * 4. screenshot the filled panel, click Generate
 * 5. read back anima_request.json (the UI wrote it) and prove it carries "snapref"
 * 6. wait on the server's progress file, then screenshot the finished image
 *
 * usage: node snapref_ui_final.js <ui_base> <ref_stem> <out_prefix>
 */
const fs = require('fs');
const path = require('path');
const NM = 'C:/Users/avero/AppData/Local/hermes/hermes-agent/node_modules';
const ROOT = 'E:/Workspace/colab/snapref_local/content';
const REQ = path.join(ROOT, 'anima_request.json');
const PROG = path.join(ROOT, 'anima_progress.json');
const LOGF = path.join(ROOT, 'ui_final.log');
const readJ = (p) => { try { return JSON.parse(fs.readFileSync(p, 'utf8')); } catch (e) { return null; } };

async function main() {
  const [base, stem, prefix] = process.argv.slice(2);
  fs.writeFileSync(LOGF, '');
  const say = (...a) => { const l = '[final] ' + a.join(' '); fs.appendFileSync(LOGF, l + '\n'); };
  const { chromium } = require(path.join(NM, 'playwright'));
  const browser = await chromium.launch({
    executablePath: 'C:/Program Files/imput/Helium/Application/chrome.exe',
    headless: true, args: ['--no-sandbox', '--disable-gpu'],
  });
  const page = await browser.newPage({ viewport: { width: 1500, height: 1300 } });
  await page.goto(base, { waitUntil: 'networkidle', timeout: 180000 });
  await page.waitForTimeout(6000);

  // ---- size and steps first, so the panel shot shows the real settings
  const nums = await page.evaluate(() => [...document.querySelectorAll('input[type=number]')]
    .map((e, i) => ({ i, aria: e.getAttribute('aria-label'), val: e.value })));
  say('number inputs:', JSON.stringify(nums.map((n) => n.aria)));

  async function setNum(ariaRe, val) {
    const el = page.locator(`input[type=number][aria-label*="${ariaRe}"]`).first();
    if (await el.count()) {
      await el.fill(String(val));
      await el.press('Enter').catch(() => {});
      await page.waitForTimeout(400);
      return await el.inputValue();
    }
    return null;
  }
  say('steps ->', await setNum('Steps', 4));
  // 1 MP OOMs on this box (SnapRef's 4096x5068 attention needs ~10.7 GB), so use the app's
  // own 0.5 MP preset -- a normal UI control, same code path, 2028 target tokens.
  const mpClicked = await page.evaluate(() => {
    const sp = [...document.querySelectorAll('span, label, div')].find(
      (e) => e.children.length === 0 && (e.textContent || '').trim() === '0.5 MP');
    if (!sp) return false;
    sp.setAttribute('data-mp05', '1');
    return true;
  });
  say('0.5 MP control found:', mpClicked);
  if (mpClicked) {
    await page.locator('[data-mp05="1"]').click();
    await page.waitForTimeout(1200);
  }
  say('size indicator:', await page.evaluate(() => [...document.querySelectorAll('div,span')]
    .map((e) => (e.textContent || '').trim())
    .filter((s) => /^\d+ × \d+ ·|MP ·/.test(s)).slice(0, 3)));

  // ---- SnapRef panel
  const head = page.getByText('SnapRef — character reference (reference adapter)', { exact: false }).first();
  await head.scrollIntoViewIfNeeded(); await head.click(); await page.waitForTimeout(1500);
  await page.evaluate(() => {
    const c = [...document.querySelectorAll('input[type=checkbox]')].find(
      (e) => ((e.closest('label') || e.parentElement || {}).textContent || '').includes('Use SnapRef'));
    if (c) c.setAttribute('data-cb', '1');
  });
  await page.locator('[data-cb="1"]').click({ force: true });
  await page.waitForTimeout(800);
  say('Use SnapRef checked =', await page.locator('[data-cb="1"]').isChecked());

  await page.evaluate(() => {
    const ins = [...document.querySelectorAll('input.border-none')].filter((i) => !i.value);
    const inp = ins[ins.length - 1] || ins[0];
    const box = inp.closest('.wrap') || inp.parentElement.parentElement;
    box.setAttribute('data-ddbox', '1');
  });
  await page.locator('[data-ddbox="1"]').click({ position: { x: 40, y: 12 } });
  await page.waitForTimeout(1100);
  const opts = await page.evaluate(() => [...document.querySelectorAll('li, [role=option]')]
    .map((e) => e.textContent.trim()).filter(Boolean).slice(0, 20));
  say('reference options:', JSON.stringify(opts));
  const opt = page.locator('li, [role=option]').filter({ hasText: stem }).first();
  if (!(await opt.count())) throw new Error('reference option not offered');
  await opt.click();
  await page.waitForTimeout(1300);
  await page.keyboard.press('Escape');
  await page.waitForTimeout(500);

  // ---- prompt: scene only
  await page.locator('textarea').first().fill(
    'masterpiece, best quality, 1girl, solo, upper body, looking at viewer, simple background, white background');
  await page.waitForTimeout(600);
  await page.screenshot({ path: `${prefix}_panel.png`, fullPage: true });
  say('panel shot written');

  // ---- Generate
  const before = readJ(REQ);
  await page.evaluate(() => {
    const b = [...document.querySelectorAll('button')].find((e) => (e.textContent || '').trim() === 'Generate');
    if (b) b.setAttribute('data-gen', '1');
  });
  await page.locator('[data-gen="1"]').click();
  say('clicked Generate');

  let req = null;
  for (let i = 0; i < 45; i++) {
    await page.waitForTimeout(1000);
    req = readJ(REQ);
    if (req && (!before || req.gen_id !== before.gen_id)) break;
  }
  if (!req) throw new Error('the UI never wrote a request');
  say('REQUEST gen_id', req.gen_id);
  say('REQUEST size/steps', `${req.width}x${req.height}, ${req.steps} steps`);
  say('REQUEST snapref', JSON.stringify(req.snapref));
  say('REQUEST prompt', JSON.stringify(req.prompt).slice(0, 110));

  // ---- wait for the server
  let last = ''; let prog = {};
  for (let i = 0; i < 1200; i++) {
    await page.waitForTimeout(2000);
    prog = readJ(PROG) || {};
    if (prog.gen_id !== req.gen_id) continue;
    const key = `${prog.stage} ${prog.step}/${prog.steps}`;
    if (key !== last) { last = key; say('server:', key, prog.note || ''); }
    if (prog.stage === 'done' || prog.stage === 'error') break;
  }
  say('final stage:', prog.stage, prog.error ? 'ERROR ' + String(prog.error).slice(0, 160) : '');
  if (prog.snapref) say('server snapref info:', JSON.stringify(prog.snapref).slice(0, 220));

  await page.waitForTimeout(9000);
  await page.screenshot({ path: `${prefix}_result.png`, fullPage: true });
  say('result shot written');
  const gal = page.locator('#anima-result').first();
  if (await gal.count()) { await gal.screenshot({ path: `${prefix}_image.png` }).catch(() => {}); say('image crop written'); }
  await browser.close();
  say('FINAL_DONE', prog.stage);
}
main().catch((e) => {
  fs.appendFileSync(LOGF, 'FAILED ' + (e.stack || e) + '\n');
  process.exit(1);
});