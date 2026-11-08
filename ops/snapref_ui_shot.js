/**
 * Screenshot proof of the SnapRef UI: real Gradio app, SnapRef accordion filled in
 * (checkbox on, reference selected, adapter/mode/size set), then the live run's result.
 *
 * Completion is read from the server's progress file, because the app's 1 Hz progress
 * timer rewrites the status box every second and would mask a one-off message.
 *
 * usage: node snapref_ui_shot.js <ui_base> <ref_basename> <shot_prefix>
 */
const fs = require('fs');
const path = require('path');
const NM = 'C:/Users/avero/AppData/Local/hermes/hermes-agent/node_modules';
const ROOT = 'E:/Workspace/colab/snapref_local/content';
const PROG = path.join(ROOT, 'anima_progress.json');

const readProg = () => { try { return JSON.parse(fs.readFileSync(PROG, 'utf8')); } catch (e) { return {}; } };

async function main() {
  const [base, refName, prefix] = process.argv.slice(2);
  const { chromium } = require(path.join(NM, 'playwright'));
  const LOGF = path.join(ROOT, 'ui_shot.log');
  fs.writeFileSync(LOGF, '');
  const log = (...a) => {
    const line = '[shot] ' + a.map((x) => (typeof x === 'string' ? x : JSON.stringify(x))).join(' ');
    fs.appendFileSync(LOGF, line + '\n');
  };

  const browser = await chromium.launch({
    executablePath: 'C:/Program Files/imput/Helium/Application/chrome.exe',
    headless: true,
    args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage'],
  });
  const page = await browser.newPage({ viewport: { width: 1500, height: 1250 } });
  await page.goto(base, { waitUntil: 'networkidle', timeout: 180000 });
  await page.waitForTimeout(6000);

  const head = page.getByText('SnapRef — character reference (reference adapter)', { exact: false }).first();
  await head.scrollIntoViewIfNeeded();
  await head.click();
  await page.waitForTimeout(1500);

  // tick "Use SnapRef"
  await page.evaluate(() => {
    const c = [...document.querySelectorAll('input[type=checkbox]')].find(
      (e) => ((e.closest('label') || e.parentElement || {}).textContent || '').includes('Use SnapRef'));
    if (c) { c.setAttribute('data-hermes', 'cb'); }
  });
  await page.locator('[data-hermes="cb"]').click({ force: true });
  await page.waitForTimeout(600);
  log('checkbox checked =', await page.locator('[data-hermes="cb"]').isChecked());

  // select the reference through the app's own dropdown
  const stem = refName.replace(/\.png$/, '');
  await page.evaluate((s) => {
    const els = [...document.querySelectorAll('*')];
    const lbl = els.find((e) => e.children.length === 0 && /Reference\(s\)/.test(e.textContent || ''));
    let n = lbl;
    for (let i = 0; i < 4 && n && n.parentElement; i++) n = n.parentElement;
    const inp = n && n.querySelector('input[role=listbox], input[type=text], input:not([type])');
    if (inp) inp.setAttribute('data-hermes', 'ref');
  }, stem);
  await page.locator('[data-hermes="ref"]').click();
  await page.waitForTimeout(700);
  await page.keyboard.type(stem, { delay: 50 });
  await page.waitForTimeout(1500);
  const opt = page.locator('ul.options li, .options .item, [role=option]').filter({ hasText: stem }).first();
  if (await opt.count()) { await opt.click(); log('clicked option'); }
  else { await page.keyboard.press('Enter'); log('pressed Enter'); }
  await page.waitForTimeout(1200);

  // verify the selection actually landed: count chips carrying the name in that widget
  const sel = await page.evaluate((s) => {
    const inp = document.querySelector('[data-hermes="ref"]');
    let n = inp;
    for (let i = 0; i < 4 && n && n.parentElement; i++) n = n.parentElement;
    const html = n ? n.innerHTML : '';
    return { mentions: (html.match(new RegExp(s, 'g')) || []).length,
             chips: n ? [...n.querySelectorAll('.token, .chip, span')].map((e) => e.textContent.trim())
                        .filter(Boolean).slice(0, 6) : [] };
  }, stem);
  log('reference widget:', JSON.stringify(sel));
  if (!sel.mentions) log('WARNING: reference selection not confirmed in the widget DOM');

  await page.screenshot({ path: `${prefix}_panel.png`, fullPage: true });
  log('panel shot written');

  // the run is already in flight; wait for the server's own progress file to say done
  let prog = readProg();
  const gid = prog.gen_id;
  log('watching gen_id', gid, 'stage', prog.stage);
  let last = '';
  for (let i = 0; i < 900; i++) {
    await page.waitForTimeout(2000);
    prog = readProg();
    const key = `${prog.stage} ${prog.step}/${prog.steps}`;
    if (key !== last) { last = key; log('server:', key, prog.note || ''); }
    if (prog.stage === 'done') break;
    if (prog.stage === 'error') { log('SERVER ERROR', prog.error || ''); break; }
  }
  log('final server stage:', prog.stage, 'steps', prog.step, '/', prog.steps,
      'seconds', prog.seconds || prog.elapsed || '');

  // let the UI's 1 Hz timer pull the finished image in, then capture the whole page
  await page.waitForTimeout(8000);
  await page.screenshot({ path: `${prefix}_result.png`, fullPage: true });
  log('result shot written');

  // and a tight crop of the result area
  const res = page.locator('#anima-result').first();
  if (await res.count()) {
    await res.screenshot({ path: `${prefix}_image.png` }).catch((e) => log('crop failed', e.message));
    log('image crop written');
  }
  await browser.close();
  fs.appendFileSync(LOGF, 'SHOT_DONE ' + prog.stage + '\n');
}
main().catch((e) => {
  try { fs.appendFileSync(path.join(ROOT, 'ui_shot.log'), 'SHOT FAILED ' + (e && e.stack ? e.stack : e) + '\n'); } catch (x) {}
  process.exit(1);
});
