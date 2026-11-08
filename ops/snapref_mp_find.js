/**
 * Find the size-preset dropdown (the one whose options read "0.5 MP".."Custom size").
 * usage: node snapref_mp_find.js <ui_base>
 */
const fs = require('fs');
const path = require('path');
const NM = 'C:/Users/avero/AppData/Local/hermes/hermes-agent/node_modules';
const ROOT = 'E:/Workspace/colab/snapref_local/content';

async function main() {
  const [base] = process.argv.slice(2);
  const out = [];
  const say = (...a) => out.push('[mp] ' + a.join(' '));
  const { chromium } = require(path.join(NM, 'playwright'));
  const browser = await chromium.launch({
    executablePath: 'C:/Program Files/imput/Helium/Application/chrome.exe',
    headless: true, args: ['--no-sandbox', '--disable-gpu'],
  });
  const page = await browser.newPage({ viewport: { width: 1500, height: 1300 } });
  await page.goto(base, { waitUntil: 'networkidle', timeout: 180000 });
  await page.waitForTimeout(6000);

  const info = await page.evaluate(() => {
    const hits = [...document.querySelectorAll('*')].filter(
      (e) => e.children.length === 0 && /MP$/.test((e.textContent || '').trim()));
    const res = hits.slice(0, 6).map((e) => {
      let wrap = e;
      for (let i = 0; i < 6 && wrap && wrap.parentElement; i++) {
        wrap = wrap.parentElement;
        if (wrap.className && String(wrap.className).includes('wrap')) break;
      }
      const inp = wrap && wrap.querySelector('input');
      if (inp) inp.setAttribute('data-mpbox-inner', '1');
      if (wrap) wrap.setAttribute('data-mpbox', '1');
      return { text: (e.textContent || '').trim(), tag: e.tagName,
               inpVal: inp ? inp.value : null, inpCls: inp ? String(inp.className) : null };
    });
    return res;
  });
  say('MP-labelled elements:', JSON.stringify(info, null, 1));
  fs.writeFileSync(path.join(ROOT, 'mp_find.log'), out.join('\n') + '\n');
  await browser.close();
}
main().catch((e) => { fs.appendFileSync(path.join(ROOT, 'mp_find.log'), 'FAILED ' + (e.stack || e) + '\n'); process.exit(1); });