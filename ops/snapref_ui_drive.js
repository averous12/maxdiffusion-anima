/**
 * UI proof for SnapRef: drive the REAL Gradio app — tick "Use SnapRef", pick the reference
 * through the app's own Dropdown (it takes basenames from the SnapRef dir), set a scene-only
 * prompt, press Generate, and screenshot the panel and the finished result.
 *
 * The UI wiring itself is proven by reading back the request file the UI wrote to disk.
 *
 * usage: node snapref_ui_drive.js <ui_base> <ref_basename> <shot_prefix>
 */
const fs = require('fs');
const path = require('path');
const NM = 'C:/Users/avero/AppData/Local/hermes/hermes-agent/node_modules';
const ROOT = 'E:/Workspace/colab/snapref_local/content';

// Locate a control in Gradio's DOM and tag it, then drive it via Playwright.
const tag = (id) => `(() => {
  const want = ${JSON.stringify(id)};
  const els = [...document.querySelectorAll('*')];
  let target = null;
  if (want === 'cb-on') {
    target = els.find(e => e.tagName === 'INPUT' && e.type === 'checkbox'
      && ((e.closest('label') || e.parentElement || {}).textContent || '').includes('Use SnapRef'));
  } else if (want === 'ref-select') {
    const lbl = els.find(e => e.children.length === 0 && /Reference\\(s\\)/.test(e.textContent || ''));
    if (lbl) { let n = lbl; for (let i = 0; i < 5 && n.parentElement; i++) n = n.parentElement;
      target = n.querySelector('input[role=listbox], input[type=text], input:not([type])'); }
  }
  if (!target) return null;
  target.setAttribute('data-hermes', want);
  return { tag: target.tagName, w: target.getBoundingClientRect().width | 0 };
})()`;

async function main() {
  const [base, refName, prefix] = process.argv.slice(2);
  const { chromium } = require(path.join(NM, 'playwright'));
  const browser = await chromium.launch({
    executablePath: 'C:/Program Files/imput/Helium/Application/chrome.exe',
    headless: true,
    args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage'],
  });
  const page = await browser.newPage({ viewport: { width: 1500, height: 1400 } });
  const LOGF = path.join(ROOT, 'ui_drive.log');
  fs.writeFileSync(LOGF, '');
  const log = (...a) => {
    const line = '[ui] ' + a.map((x) => (typeof x === 'string' ? x : JSON.stringify(x))).join(' ');
    console.log(line);
    try { fs.appendFileSync(LOGF, line + '\n'); } catch (e) { /* ignore */ }
  };
  const errs = [];
  page.on('pageerror', (e) => errs.push('pageerror: ' + e.message));
  const api = [];
  page.on('response', async (r) => {
    try {
      if (!r.url().includes('/gradio_api/')) return;
      const body = r.request().method() === 'POST' ? (await r.text()).slice(0, 600) : '';
      api.push({ status: r.status(), method: r.request().method(), url: r.url().split('/gradio_api/')[1],
                 body });
    } catch (e) { /* body unavailable */ }
  });

  await page.goto(base, { waitUntil: 'networkidle', timeout: 180000 });
  await page.waitForTimeout(6000);
  log('title:', await page.title());

  const head = page.getByText('SnapRef — character reference (reference adapter)', { exact: false }).first();
  await head.scrollIntoViewIfNeeded();
  await head.click();
  await page.waitForTimeout(2000);

  // ---- 1. tick "Use SnapRef"
  const cbInfo = await page.evaluate(tag('cb-on'));
  log('checkbox located:', JSON.stringify(cbInfo));
  if (!cbInfo) throw new Error('Use SnapRef checkbox not found');
  await page.locator('[data-hermes="cb-on"]').click({ force: true });
  await page.waitForTimeout(900);
  const cbChecked = await page.locator('[data-hermes="cb-on"]').isChecked();
  log('Use SnapRef checked =', cbChecked);
  if (!cbChecked) throw new Error('checkbox did not tick');

  // ---- 2. pick the reference through the app's own dropdown (basenames)
  const refInfo = await page.evaluate(tag('ref-select'));
  log('reference dropdown located:', JSON.stringify(refInfo));
  if (!refInfo) throw new Error('reference dropdown not found');
  const stem = refName.replace(/\.png$/, '');
  await page.locator('[data-hermes="ref-select"]').click();
  await page.waitForTimeout(800);
  await page.keyboard.type(stem, { delay: 50 });
  await page.waitForTimeout(1600);
  const opt = page.locator('ul.options li, .options .item, [role=option]')
    .filter({ hasText: stem }).first();
  if (await opt.count()) { await opt.click(); log('clicked the dropdown option'); }
  else { await page.keyboard.press('Enter'); log('committed the selection with Enter'); }
  await page.waitForTimeout(1200);

  const refState = await page.evaluate(() => {
    const lbl = [...document.querySelectorAll('*')]
      .find(e => e.children.length === 0 && /Reference\(s\)/.test(e.textContent || ''));
    let n = lbl;
    for (let i = 0; i < 5 && n && n.parentElement; i++) n = n.parentElement;
    return n ? (n.textContent || '').trim().slice(0, 300) : '';
  });
  log('reference block now reads:', JSON.stringify(refState));
  if (!refState.includes(stem)) log('WARNING: selected reference not visible in the block text');

  // ---- 3. scene-only prompt, as the adapter documents
  await page.locator('textarea').first().fill(
    'masterpiece, best quality, 1girl, solo, upper body, looking at viewer, simple background, white background');
  await page.waitForTimeout(500);
  await page.screenshot({ path: `${prefix}_panel.png`, fullPage: true });
  log('panel shot:', `${prefix}_panel.png`);

  // ---- 4. Generate (tag the exact submit button from inside the page)
  const gInfo = await page.evaluate(() => {
    const b = [...document.querySelectorAll('button')].find((e) => (e.textContent || '').trim() === 'Generate');
    if (!b) return null;
    b.setAttribute('data-hermes', 'gen');
    return { disabled: b.disabled };
  });
  log('generate button:', JSON.stringify(gInfo));
  if (!gInfo) throw new Error('Generate button not found');
  await page.locator('[data-hermes="gen"]').click();
  log('clicked Generate');

  // ---- 5. prove the UI built the snapref request
  await page.waitForTimeout(9000);
  for (const a of api) log(`API ${a.method} ${a.status} ${a.url} ${a.body ? ':: ' + a.body.slice(0, 300) : ''}`);
  try {
    const req = JSON.parse(fs.readFileSync(path.join(ROOT, 'anima_request.json'), 'utf8'));
    log('REQUEST gen_id =', req.gen_id);
    log('REQUEST snapref =', JSON.stringify(req.snapref));
    log('REQUEST prompt  =', JSON.stringify(req.prompt).slice(0, 130));
    log('REQUEST size    =', `${req.width}x${req.height}, ${req.steps} steps, seed ${req.seed}`);
  } catch (e) { log('request read failed:', e.message); }

  // ---- 6. wait for the result
  let status = '';
  for (let i = 0; i < 500; i++) {
    await page.waitForTimeout(2000);
    status = await page.evaluate(() => {
      const el = document.getElementById('anima-status');
      if (!el) return '';
      const t = el.querySelector('textarea') || el;
      return (t.value || t.textContent || '').slice(0, 200);
    });
    if (i % 15 === 0) log(`status: ${JSON.stringify(status).slice(0, 140)}`);
    if (/done in|error/i.test(status)) break;
  }
  log('final status:', JSON.stringify(status));
  await page.waitForTimeout(3000);
  await page.screenshot({ path: `${prefix}_result.png`, fullPage: true });
  log('result shot:', `${prefix}_result.png`);
  if (errs.length) log('page errors:', JSON.stringify(errs).slice(0, 400));
  await browser.close();
  console.log('UI_PROOF_DONE status=' + JSON.stringify(status));
}
main().catch((e) => { console.error('UI PROOF FAILED:', e); process.exit(1); });
