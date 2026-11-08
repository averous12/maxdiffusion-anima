/**
 * Select a reference in the SnapRef multiselect (the empty `.subdued` dropdown input) and
 * verify the selection lands, then leave the page on the filled panel.
 * usage: node snapref_dd_pick.js <ui_base> <ref_stem>
 */
const fs = require('fs');
const path = require('path');
const NM = 'C:/Users/avero/AppData/Local/hermes/hermes-agent/node_modules';
const ROOT = 'E:/Workspace/colab/snapref_local/content';

async function main() {
  const [base, stem] = process.argv.slice(2);
  const { chromium } = require(path.join(NM, 'playwright'));
  const out = [];
  const say = (...a) => out.push('[pick] ' + a.join(' '));
  const browser = await chromium.launch({
    executablePath: 'C:/Program Files/imput/Helium/Application/chrome.exe',
    headless: true, args: ['--no-sandbox', '--disable-gpu'],
  });
  const page = await browser.newPage({ viewport: { width: 1500, height: 1250 } });
  await page.goto(base, { waitUntil: 'networkidle', timeout: 180000 });
  await page.waitForTimeout(6000);
  const head = page.getByText('SnapRef — character reference (reference adapter)', { exact: false }).first();
  await head.scrollIntoViewIfNeeded(); await head.click(); await page.waitForTimeout(1500);

  // the multiselect is the only dropdown input with no value
  const found = await page.evaluate(() => {
    const ins = [...document.querySelectorAll('input.border-none, input.svelte-1xfsv4t')];
    const info = ins.map((i, k) => ({ k, val: i.value, cls: String(i.className) }));
    const empty = ins.findIndex((i) => !i.value);
    if (empty >= 0) ins[empty].setAttribute('data-pick', '1');
    return { info, empty };
  });
  say('dropdown inputs:', JSON.stringify(found.info));
  if (found.empty < 0) { say('NO EMPTY DROPDOWN FOUND'); fs.writeFileSync(path.join(ROOT, 'dd_pick.log'), out.join('\n') + '\n'); await browser.close(); return; }

  await page.locator('[data-pick="1"]').click();
  await page.waitForTimeout(800);
  await page.keyboard.type(stem, { delay: 60 });
  await page.waitForTimeout(2000);

  const pop = await page.evaluate(() => {
    const lists = [...document.querySelectorAll('ul, .options, [role=listbox]')]
      .map((e) => ({ cls: String(e.className).slice(0, 40),
                     items: [...e.querySelectorAll('li,[role=option],.item')].map((x) => x.textContent.trim()).slice(0, 8) }))
      .filter((l) => l.items.length);
    return lists;
  });
  say('popup items:', JSON.stringify(pop).slice(0, 600));

  const opt = page.locator('li, [role=option], .item').filter({ hasText: stem }).first();
  if (await opt.count()) {
    await opt.click();
    await page.waitForTimeout(1500);
    const state = await page.evaluate(() => {
      const inp = document.querySelector('[data-pick="1"]');
      let n = inp;
      for (let i = 0; i < 3 && n && n.parentElement; i++) n = n.parentElement;
      return { val: inp ? inp.value : null, text: n ? n.textContent.replace(/\s+/g, ' ').slice(0, 160) : '' };
    });
    say('after click -> value:', JSON.stringify(state.val));
    say('after click -> block text:', state.text);
  } else { say('no matching option'); }

  await page.screenshot({ path: path.join(ROOT, 'dd_pick_shot.png'), fullPage: true });
  fs.writeFileSync(path.join(ROOT, 'dd_pick.log'), out.join('\n') + '\n');
  await browser.close();
}
main().catch((e) => { fs.appendFileSync(path.join(ROOT, 'dd_pick.log'), 'FAILED ' + (e.stack || e) + '\n'); process.exit(1); });