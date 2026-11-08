/**
 * Inspect the SnapRef panel's real DOM so the proof script can drive it exactly.
 * usage: node snapref_ui_inspect.js <ui_base_url> <out_html> [shot]
 */
const fs = require('fs');
const path = require('path');
const NM = 'C:/Users/avero/AppData/Local/hermes/hermes-agent/node_modules';

async function main() {
  const [base, outHtml, shot] = process.argv.slice(2);
  const { chromium } = require(path.join(NM, 'playwright'));
  const browser = await chromium.launch({
    executablePath: 'C:/Program Files/imput/Helium/Application/chrome.exe',
    headless: true,
    args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage'],
  });
  const page = await browser.newPage({ viewport: { width: 1500, height: 1250 } });
  const errs = [];
  page.on('console', (m) => { if (m.type() === 'error') errs.push(m.text()); });
  page.on('pageerror', (e) => errs.push('pageerror: ' + e.message));

  await page.goto(base, { waitUntil: 'networkidle', timeout: 180000 });
  await page.waitForTimeout(6000);

  const head = page.getByText('SnapRef — character reference (reference adapter)', { exact: false }).first();
  await head.scrollIntoViewIfNeeded();
  await head.click();
  await page.waitForTimeout(2000);

  // the accordion body: walk up from the header to the enclosing panel
  const html = await page.evaluate(() => {
    const h = [...document.querySelectorAll('*')].find(
      (e) => e.children.length === 0 && /SnapRef — character/.test(e.textContent || ''));
    if (!h) return 'HEADER NOT FOUND';
    let n = h;
    for (let i = 0; i < 6 && n.parentElement; i++) n = n.parentElement;
    return n.outerHTML;
  });
  fs.writeFileSync(outHtml, html);
  console.log('[inspect] wrote', outHtml, html.length, 'chars');

  const info = await page.evaluate(() => {
    const out = { checkboxes: [], labels: [], fileInputs: [], listboxes: [], buttons: [] };
    document.querySelectorAll('input[type=checkbox]').forEach((c) => out.checkboxes.push({
      checked: c.checked, aria: c.getAttribute('aria-label'),
      id: c.id, near: (c.closest('label') || c.parentElement)?.textContent?.trim().slice(0, 60),
    }));
    document.querySelectorAll('input[type=file]').forEach((f) => out.fileInputs.push({
      accept: f.accept, multiple: f.multiple, hidden: f.offsetParent === null,
      near: (f.closest('.block,.form,.gr-block') || f.parentElement)?.textContent?.trim().slice(0, 70),
    }));
    document.querySelectorAll('input[role=listbox], [role=listbox], select').forEach((l) => out.listboxes.push({
      tag: l.tagName, cls: String(l.className).slice(0, 90), id: l.id,
      opts: [...l.querySelectorAll('option')].map((o) => o.textContent.trim()).slice(0, 10),
      val: l.value || null,
    }));
    document.querySelectorAll('button').forEach((b) => out.buttons.push({
      text: (b.textContent || '').trim().slice(0, 40), id: b.id,
      cls: String(b.className).slice(0, 60), disabled: b.disabled,
    }));
    return out;
  });
  fs.writeFileSync(outHtml.replace(/\.html$/, '.json'), JSON.stringify({ info, errs }, null, 1));
  console.log('[inspect] checkboxes:', JSON.stringify(info.checkboxes).slice(0, 700));
  console.log('[inspect] fileInputs:', JSON.stringify(info.fileInputs).slice(0, 700));
  console.log('[inspect] listboxes:', JSON.stringify(info.listboxes).slice(0, 700));
  console.log('[inspect] buttons:', JSON.stringify(info.buttons.map((b) => b.text + (b.disabled ? '(disabled)' : ''))).slice(0, 600));
  if (shot) { await page.screenshot({ path: shot, fullPage: true }); console.log('[inspect] shot', shot); }
  await browser.close();
}
main().catch((e) => { console.error('INSPECT FAILED:', e); process.exit(1); });
