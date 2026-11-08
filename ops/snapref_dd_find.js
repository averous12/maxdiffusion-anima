/**
 * Robustly identify the SnapRef reference dropdown input by its nearest visible label,
 * then dump what it looks like so the proof script can select a value like a user.
 * usage: node snapref_dd_find.js <ui_base> [stem]
 */
const fs = require('fs');
const path = require('path');
const NM = 'C:/Users/avero/AppData/Local/hermes/hermes-agent/node_modules';
const ROOT = 'E:/Workspace/colab/snapref_local/content';

const FIND = `(() => {
  const out = [];
  const inputs = [...document.querySelectorAll('input, textarea, [role=combobox]')];
  inputs.forEach((inp, idx) => {
    let n = inp, label = null;
    for (let i = 0; i < 8 && n && n.parentElement; i++) {
      n = n.parentElement;
      const l = n.querySelector('label');
      if (l && (l.textContent || '').trim()) { label = (l.textContent || '').trim().slice(0, 40); break; }
    }
    if (label) out.push({ idx, tag: inp.tagName, type: inp.type || '', role: inp.getAttribute('role') || '',
                          cls: String(inp.className).slice(0, 40), label,
                          aria: inp.getAttribute('aria-label') || '', ph: inp.placeholder || '',
                          val: inp.value || '' });
  });
  return out;
})()`;

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

  const all = await page.evaluate(FIND);
  const snap = all.filter((a) => /Reference|adapter|Mode|Strength|Fidelity|size \(MP\)|caption|negative|guidance|correction/i.test(a.label));
  say('SnapRef-related inputs:', JSON.stringify(snap, null, 1));

  const ref = all.find((a) => /^Reference/i.test(a.label));
  say('reference input:', JSON.stringify(ref));
  if (ref && stem) {
    // tag it by a unique attribute we can address from Playwright
    await page.evaluate((r) => {
      const ins = [...document.querySelectorAll('input, textarea, [role=combobox]')];
      const el = ins[r.idx];
      if (el) { el.setAttribute('data-refdd', '1'); el.removeAttribute('data-dd'); }
    }, ref);
    await page.locator('[data-refdd="1"]').click();
    await page.waitForTimeout(700);
    await page.keyboard.type(stem, { delay: 60 });
    await page.waitForTimeout(2000);
    const pop = await page.evaluate(() => {
      const lists = [...document.querySelectorAll('ul, [role=listbox], .options')]
        .map((e) => ({ cls: String(e.className).slice(0, 50),
                       items: [...e.querySelectorAll('li,[role=option],.item')].map((x) => x.textContent.trim()).slice(0, 8) }))
        .filter((l) => l.items.length);
      const inp = document.querySelector('[data-refdd="1"]');
      return { lists, val: inp ? inp.value : null };
    });
    say('popup lists:', JSON.stringify(pop.lists).slice(0, 800));
    say('input value after typing:', pop.val);
    const opt = page.locator('li,[role=option],.item').filter({ hasText: stem }).first();
    if (await opt.count()) {
      await opt.click();
      await page.waitForTimeout(1200);
      const after = await page.evaluate(() => {
        const inp = document.querySelector('[data-refdd="1"]');
        let n = inp;
        for (let i = 0; i < 3 && n && n.parentElement; i++) n = n.parentElement;
        return { val: inp ? inp.value : null, text: n ? n.textContent.replace(/\s+/g, ' ').slice(0, 200) : '' };
      });
      say('after clicking option:', JSON.stringify(after));
    } else { say('no option element matched'); }
  }
  fs.writeFileSync(path.join(ROOT, 'dd_find.log'), out.join('\n') + '\n');
  await browser.close();
}
main().catch((e) => {
  fs.appendFileSync(path.join(ROOT, 'dd_find.log'), 'FAILED ' + (e.stack || e) + '\n');
  process.exit(1);
});