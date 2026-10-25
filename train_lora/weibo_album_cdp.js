// 微博相册 CDP 采集：逐张按 →，取 /large/ 原图 + caption，按明星分目录
// usage: node weibo_album_cdp.js "<album_url>" [count] [outdir]
//   node weibo_album_cdp.js "https://weibo.com/u/7619798979?tabtype=album&uid=7619798979&index=0" 200 /dataset/_src/weibo_album
// 依赖：已用 operator/tools/chrome_debug.sh 启动固定端口 Chrome(9222)
const fs = require('fs');
const path = require('path');

const ALBUM = process.argv[2];
const COUNT = parseInt(process.argv[3] || '200', 10);
const OUT = process.argv[4] || '/dataset/_src/weibo_album';
const SCRIPT_DIR = __dirname;
const NAMES_FILE = process.env.CELEB_NAMES || path.join(SCRIPT_DIR, 'celebrities.txt');
const NO_DOWNLOAD = process.env.NO_DOWNLOAD === '1';

const sleep = ms => new Promise(r => setTimeout(r, ms));
const tidy = u => u.replace(/\/(orj360|orj480|thumb150|thumb180|thumbnail|mw690|small|wap360|bmiddle|mw1024|webp720|mw2000)\//, '/large/').replace(/\?.*$/, '');
const safe = s => s.replace(/[\/\\:*?"<>|\s]+/g, '_').slice(0, 40) || '_unknown';

function loadNames() {
  const list = [];
  for (const line of fs.readFileSync(NAMES_FILE, 'utf8').split('\n')) {
    const s = line.trim();
    if (!s || s.startsWith('#')) continue;
    const [canon, ...rest] = s.split('\t');
    const aliases = (rest.join('\t') || '').split(',').map(x => x.trim()).filter(Boolean);
    for (const a of [canon, ...aliases]) list.push({ canon, a, n: a.length });
  }
  return list.sort((x, y) => y.n - x.n);
}
const NAMES = loadNames();

function matchCeleb(caption) {
  const c = caption.replace(/\s+/g, ' ');
  for (const e of NAMES) if (c.includes(e.a)) return e.canon;
  const tags = [...caption.matchAll(/#([^#\n]{1,40})#/g)].map(m => m[1].trim());
  const bad = /壁纸|写真|美女|穿搭|时尚|摄影|视频|图片|高清|素材|头像|表情包|之夜|演唱会|晚宴|奖池|造型|发言|争议|偶遇/;
  for (const t of tags) {
    const tok = t.split(/\s+/)[0].replace(/[的了吧啊呀！!，,。.]$/, '');
    if (tok.length >= 2 && tok.length <= 12 && !bad.test(tok)) return tok;
  }
  return 'unknown';
}

async function wsUrl() {
  try {
    const r = await fetch('http://127.0.0.1:9222/json/version');
    const j = await r.json();
    if (j.webSocketDebuggerUrl) return j.webSocketDebuggerUrl;
  } catch (e) {}
  const pf = process.env.HOME + '/.config/google-chrome/DevToolsActivePort';
  const [port, bpath] = fs.readFileSync(pf, 'utf8').trim().split('\n');
  return 'ws://127.0.0.1:' + port + bpath;
}

(async () => {
  if (!ALBUM) { console.error('usage: node weibo_album_cdp.js <album_url> [count] [outdir]'); process.exit(2); }
  fs.mkdirSync(OUT, { recursive: true });

  const ws = new WebSocket(await wsUrl());
  let id = 0; const pend = {};
  const send = (method, params, sessionId) => new Promise((res, rej) => {
    const i = ++id; pend[i] = { res, rej };
    ws.send(JSON.stringify({ id: i, method, params: params || {}, sessionId }));
    setTimeout(() => { if (pend[i]) { delete pend[i]; rej(new Error('timeout ' + method)); } }, 15000);
  });
  ws.onmessage = e => {
    const m = JSON.parse(e.data);
    if (m.id && pend[m.id]) { m.error ? pend[m.id].rej(m.error) : pend[m.id].res(m.result); delete pend[m.id]; }
  };
  await new Promise(r => ws.onopen = r);
  const { targetId } = await send('Target.createTarget', { url: 'about:blank' });
  const { sessionId } = await send('Target.attachToTarget', { targetId, flatten: true });
  await send('Page.enable', {}, sessionId);
  await send('Runtime.enable', {}, sessionId);
  await send('Page.navigate', { url: ALBUM }, sessionId);
  await send('Page.bringToFront', {}, sessionId);
  await sleep(7000);

  const ev = async expr => {
    const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true }, sessionId);
    return r.result && r.result.value;
  };
  const pressRight = async () => {
    await send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'ArrowRight', code: 'ArrowRight', windowsVirtualKeyCode: 39, nativeVirtualKeyCode: 39 }, sessionId);
    await send('Input.dispatchKeyEvent', { type: 'keyUp', key: 'ArrowRight', code: 'ArrowRight', windowsVirtualKeyCode: 39, nativeVirtualKeyCode: 39 }, sessionId);
  };

  await ev("(()=>{const e=[...document.querySelectorAll('div')].find(x=>x.textContent.trim()==='全部图片');if(e)e.click();return !!e})()");
  await sleep(2500);
  const opened = await ev("(()=>{const c=document.querySelector('.woo-picture-main.woo-picture-square');if(!c)return false;c.click();return true})()");
  await sleep(3000);
  console.error('viewer_open=' + opened);

  const EXTRACT = `(()=>{
    const roots=[...document.querySelectorAll('div')].filter(e=>{
      const t=e.innerText||'';
      return /来自|发布于/.test(t) && e.querySelector('img[src*="sinaimg"]');
    });
    const root=roots.sort((a,b)=>b.querySelectorAll('img').length-a.querySelectorAll('img').length)[0];
    if(!root) return null;
    let big=root.querySelector('._imgElm_x308k_114 img');
    if(!big){ big=[...root.querySelectorAll('img[src*="sinaimg"]')].map(i=>({i,r:i.getBoundingClientRect()}))
      .filter(x=>x.r.width>400&&x.r.height>300).sort((a,b)=>b.r.width*b.r.height-a.r.width*a.r.height)[0]?.i; }
    if(!big) return null;
    return {src:big.currentSrc||big.src, text:(root.innerText||'').slice(0,3000)};
  })()`;

  const rows = [];
  const seen = new Set();
  let prev = '', same = 0, miss = 0;
  for (let n = 0; n < COUNT; n++) {
    const v = await ev(EXTRACT);
    let url = '';
    if (v && v.src && /sinaimg\.cn/.test(v.src)) {
      url = tidy(v.src);
      if (!seen.has(url)) {
        seen.add(url);
        const cap = (v.text || '').replace(/\s+/g, ' ').trim();
        const celeb = matchCeleb(cap);
        rows.push({ url, celeb, caption: cap });
        console.error(`[${rows.length}/${COUNT}] ${celeb} ${url}`);
      }
    }
    if (url && url === prev) same++; else same = 0;
    prev = url;
    if (!url) miss++; else miss = 0;
    if (same >= 4 || miss >= 4) { console.error('stop: no progress'); break; }
    await pressRight();
    await sleep(1300);
  }

  const tsv = path.join(OUT, 'metadata.tsv');
  fs.writeFileSync(tsv, rows.map(r => `${r.url}\t${r.celeb}\t${r.caption}`).join('\n') + '\n');
  console.error(`collected=${rows.length} -> ${tsv}`);

  if (NO_DOWNLOAD) { await send('Target.closeTarget', { targetId }); ws.close(); process.exit(0); }

  const jobs = [];
  rows.forEach((r, i) => {
    const dir = path.join(OUT, safe(r.celeb));
    fs.mkdirSync(dir, { recursive: true });
    const base = (r.url.split('/').pop() || 'img.jpg').replace(/[^\w.\-]+/g, '_');
    jobs.push({ url: r.url, path: path.join(dir, String(i + 1).padStart(3, '0') + '_' + base), celeb: r.celeb });
  });
  let ok = 0, fail = 0, idx = 0;
  async function worker() {
    while (true) {
      const k = idx++; if (k >= jobs.length) return;
      const j = jobs[k];
      if (fs.existsSync(j.path) && fs.statSync(j.path).size > 1024) { ok++; continue; }
      try {
        const r = await fetch(j.url, { headers: { 'User-Agent': 'Mozilla/5.0', 'Referer': 'https://weibo.com/' } });
        if (!r.ok) throw new Error(String(r.status));
        const b = Buffer.from(await r.arrayBuffer());
        if (b.length < 1024) throw new Error('small');
        fs.writeFileSync(j.path, b); ok++;
      } catch (e) { fail++; console.error('DLFAIL', j.url, e.message); }
      if ((ok + fail) % 20 === 0) console.error(`download ${ok + fail}/${jobs.length}`);
    }
  }
  await Promise.all(Array.from({ length: 6 }, worker));
  console.error(`downloaded ok=${ok} fail=${fail} -> ${OUT}`);

  await send('Target.closeTarget', { targetId });
  ws.close();
  process.exit(0);
})().catch(e => { console.error('ERR', e.message || JSON.stringify(e)); process.exit(1); });
