/**
 * Dump the SnapRef reference dropdown's popup/list markup after typing, so the proof
 * script can select the reference the way a user does.
 * usage: node snapref_dd_inspect.js <ui_base> <ref_stem>
 */
const fs = require('fs');
const path = require('path');
const NM = 'C:/Users/avero/AppData/Local/hermes/hermes-agent/node_modules';
const ROOT = 'E:/Workspace/colab/snapref_local/content';

async function main() {
  const [base, stem] = process.argv.slice(2);
  const { chromium } = require(path.join(NM, 'playwright'));
  const out = [];
  const say = (...a) => out.push('[dd] ' + a.join(' '));
  const browser = await chromium.launch({
    executablePath: 'C:/Program Files/imput/Helium/Application/chrome.exe',
    headless: true, args: ['--no-sandbox', '--disable-gpu'],
  });
  const page = await browser.newPage({ viewport: { width: 1500, height: 1250 } });
  await page.goto(base, { waitUntil: 'networkidle', timeout: 180000 });
  await page.waitForTimeout(6000);
  const head = page.getByText('SnapRef — character reference (reference adapter)', { exact: false }).first();
  await head.scrollIntoViewIfNeeded(); await head.click(); await page.waitForTimeout(1500);

  // tag the reference input
  const info = await page.evaluate(() => {
    const els = [...document.querySelectorAll('*')];
    const lbl = els.find((e) => e.children.length === 0 && /Reference\(s\)/.test(e.textContent || ''));
    let n = lbl;
    for (let i = 0; i < 4 && n && n.parentElement; i++) n = n.parentElement;
    const inp = n && n.querySelector('input');
    if (inp) inp.setAttribute('data-dd', '1');
    return { found: !!inp, html: n ? n.innerHTML.slice(0, 1200) : '' };
  });
  say('widget before typing:', info.html.replace(/\s+/g, ' ').slice(0, 700));

  await page.locator('[data-dd="1"]').click();
  await page.waitForTimeout(600);
  await page.keyboard.type(stem, { delay: 60 });
  await page.waitForTimeout(1800);

  const after = await page.evaluate(() => {
    const lists = [...document.querySelectorAll('ul, [role=listbox], .options, .dropdown-content')]
      .map((e) => ({ cls: String(e.className).slice(0, 60), items: [...e.querySelectorAll('li,[role=option],.item')]
        .map((x) => x.textContent.trim()).slice(0, 8) }));
    const inp = document.querySelector('[data-dd="1"]');
    let n = inp;
    for (let i = 0; i < 4 && n && n.parentElement; i++) n = n.parentElement;
    return { lists: lists.filter((l) => l.items.length), widget: n ? n.innerHTML.replace(/\s+/g, ' ').slice(0, 900) : '' };
  });
  say('lists:', JSON.stringify(after.lists).slice(0, 900));
  say('widget after typing:', after.widget.slice(0, 700));
  fs.writeFileSync(path.join(ROOT, 'dd_inspect.log'), out.join('\n') + '\n');
  await browser.close();
}
main().catch((e) => {
  fs.appendFileSync(path.join(ROOT, 'dd_inspect.log'), 'FAILED ' + (e.stack || e) + '\n');
  process.exit(1);
});