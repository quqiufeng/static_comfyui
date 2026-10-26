// 微博相册 CDP 采集：逐张按 →，取 /large/ 原图 + caption，按明星分目录
// usage: node weibo_album_cdp.js "<album_url>" [count] [outdir]
//   node weibo_album_cdp.js "https://weibo.com/u/7619798979?tabtype=album&uid=7619798979&index=0" 200 /dataset/_src/weibo_album
// 依赖：已用 operator/tools/chrome_debug.sh 启动固定端口 Chrome(9222)
const fs = require('fs');
const path = require('path');
const { execFile } = require('child_process');
const { promisify } = require('util');
const execFileAsync = promisify(execFile);

const ALBUM = process.argv[2];
const COUNT = parseInt(process.argv[3] || '200', 10);
const OUT = process.argv[4] || '/dataset/_src/weibo_album';
const SCRIPT_DIR = __dirname;
const NAMES_FILE = process.env.CELEB_NAMES || path.join(SCRIPT_DIR, 'celebrities.txt');
const NO_DOWNLOAD = process.env.NO_DOWNLOAD === '1';
const MAX = parseInt(process.env.MAX || '2048', 10);
const FF_MS = parseInt(process.env.FF_MS || '60', 10);

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

  const tsv = path.join(OUT, 'metadata.tsv');
  const progressFile = path.join(OUT, 'progress.json');
  const uid = (ALBUM.match(/uid=(\d+)/) || [])[1] || '';
  const existing = [];
  if (fs.existsSync(tsv)) {
    for (const line of fs.readFileSync(tsv, 'utf8').split('\n')) {
      if (!line.trim()) continue;
      const p = line.split('\t');
      if (!p[0]) continue;
      existing.push({ url: p[0], celeb: p[1] || 'unknown', caption: p.slice(2).join('\t') });
    }
  }
  const writeProgress = (count, last, ended = false) => {
    fs.writeFileSync(progressFile, JSON.stringify({
      uid, out: OUT, count, target: COUNT, last_url: last || '',
      ended, status: ended ? 'end' : 'running', updated: new Date().toISOString()
    }, null, 2) + '\n');
  };
  writeProgress(existing.length, existing.length ? existing[existing.length - 1].url : '', existing.length >= COUNT);
  if (existing.length >= COUNT) {
    console.error(`already ${existing.length}/${COUNT} -> ${tsv}`);
    process.exit(0);
  }
  console.error(`resume existing=${existing.length} target=${COUNT}`);

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
  await new Promise((res, rej) => {
    ws.onopen = res;
    ws.onerror = () => rej(new Error('CDP websocket error（Chrome 9222 是否还活着？）'));
    setTimeout(() => rej(new Error('CDP connect timeout')), 8000);
  });
  let targetId = '', reused = false;
  if (process.env.TARGET_ID) { targetId = process.env.TARGET_ID; reused = true; console.error('use TARGET_ID ' + targetId); }
  if (!targetId && process.env.REUSE_TAB !== '0') {
    const { targetInfos } = await send('Target.getTargets');
    const p = (targetInfos || []).find(t => t.type === 'page' &&
      t.url.includes(`/u/${uid}`) && t.url.includes('tabtype=album'));
    if (p) { targetId = p.targetId; reused = true; console.error('reuse tab ' + p.url); }
  }
  if (!targetId) targetId = (await send('Target.createTarget', { url: 'about:blank' })).targetId;
  const { sessionId } = await send('Target.attachToTarget', { targetId, flatten: true });
  await send('Page.enable', {}, sessionId);
  await send('Runtime.enable', {}, sessionId);
  if (reused && process.env.NO_ACTIVATE !== '1') {
    try { await send('Target.activateTarget', { targetId }); } catch (e) {}
  }
  if (!reused) {
    await send('Page.navigate', { url: ALBUM }, sessionId);
    await send('Page.bringToFront', {}, sessionId);
    await sleep(7000);
  } else {
    await sleep(1200);
  }

  const ev = async expr => {
    const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true }, sessionId);
    return r.result && r.result.value;
  };
  const pressRight = async () => {
    await send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'ArrowRight', code: 'ArrowRight', windowsVirtualKeyCode: 39, nativeVirtualKeyCode: 39 }, sessionId);
    await send('Input.dispatchKeyEvent', { type: 'keyUp', key: 'ArrowRight', code: 'ArrowRight', windowsVirtualKeyCode: 39, nativeVirtualKeyCode: 39 }, sessionId);
  };
  const pressLeft = async () => {
    await send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'ArrowLeft', code: 'ArrowLeft', windowsVirtualKeyCode: 37, nativeVirtualKeyCode: 37 }, sessionId);
    await send('Input.dispatchKeyEvent', { type: 'keyUp', key: 'ArrowLeft', code: 'ArrowLeft', windowsVirtualKeyCode: 37, nativeVirtualKeyCode: 37 }, sessionId);
  };
  const curIndex = async () => {
    const href = await ev('location.href');
    const m = String(href).match(/[?&]index=(\d+)/);
    return m ? parseInt(m[1], 10) : -1;
  };

  const N = existing.length;
  let opened = false, needFF = 0;
  const reusedViewer = reused && await ev("!!document.querySelector('._imgElm_x308k_114 img') || !!document.querySelector('[class*=\"_imgElm\"] img')");
  if (reusedViewer) {
    let cur = await curIndex(), guard = 0;
    while (cur > N && guard < 800) { await pressLeft(); await sleep(80); cur = await curIndex(); guard++; }
    while (cur >= 0 && cur < N && guard < 1600) { await pressRight(); await sleep(150); cur = await curIndex(); guard++; }
    console.error(`reuse tab index=${cur} target=${N}`);
    opened = true;
    await sleep(1500);
  } else {
  await ev("(()=>{const e=[...document.querySelectorAll('div')].find(x=>x.textContent.trim()==='全部图片');if(e)e.click();return !!e})()");
  await sleep(2500);

  // 续采：数相册网格 div.woo-box-item-inlineBlock img，滚到第 N 张直接点开
  let imgCount = 0;
  {
    let last = -1, stable = 0;
    for (let s = 0; s < 200; s++) {
      const info = await ev(`(()=>{const a=[...document.querySelectorAll('div.woo-box-item-inlineBlock img')].filter(i=>/sinaimg\\.cn/.test(i.src)&&!/avatar|crop\\./.test(i.src));return {n:a.length}})()`);
      imgCount = info ? info.n : 0;
      if (imgCount > N) break;
      if (imgCount === last) stable++; else stable = 0;
      last = imgCount;
      await ev("(()=>{const a=[...document.querySelectorAll('div.woo-box-item-inlineBlock img')];if(a.length)a[a.length-1].scrollIntoView({block:'center'});else window.scrollBy(0,1600);})()");
      await sleep(280);
      if (stable >= 8) break;
    }
    console.error(`grid_images=${imgCount} resume_index=${N}`);
    // 选起点：有下一张点 N；相册正好 N 张则点最后一张 N-1（进 viewer 后自然结束）
    let clickIdx = -1;
    if (imgCount > N) clickIdx = N;
    else if (imgCount === N && N > 0) clickIdx = N - 1;
    else if (N === 0 && imgCount > 0) clickIdx = 0;
    if (clickIdx >= 0) {
      await ev(`(()=>{const a=[...document.querySelectorAll('div.woo-box-item-inlineBlock img')].filter(i=>/sinaimg\\.cn/.test(i.src)&&!/avatar|crop\\./.test(i.src));const im=a[${clickIdx}];if(!im)return false;im.scrollIntoView({block:'center'});im.click();return true})()`);
      await sleep(2500);
      opened = await ev("!!document.querySelector('._imgElm_x308k_114 img') || !!document.querySelector('[class*=\"_imgElm\"] img')");
      if (!opened) {
        await ev(`(()=>{const a=[...document.querySelectorAll('div.woo-box-item-inlineBlock img')].filter(i=>/sinaimg\\.cn/.test(i.src)&&!/avatar|crop\\./.test(i.src));const im=a[${clickIdx}];const p=im&&im.closest('div.woo-box-item-inlineBlock');if(p){p.click();return true}return false})()`);
        await sleep(2500);
        opened = await ev("!!document.querySelector('._imgElm_x308k_114 img') || !!document.querySelector('[class*=\"_imgElm\"] img')");
      }
    }
  }

  if (!opened && N > 0) {
    // 网格虚拟化时拿不到第 N 张：从第一张进 viewer，快速 → 到上次最后一张
    console.error(`grid insufficient (${imgCount} <= ${N}); fast-forward from 0`);
    await ev("(()=>{const c=document.querySelector('.woo-picture-main.woo-picture-square');if(!c)return false;c.click();return true})()");
    await sleep(2500);
    opened = await ev("!!document.querySelector('._imgElm_x308k_114 img') || !!document.querySelector('[class*=\"_imgElm\"] img')");
    needFF = Math.max(0, N - 1);
  }
  if (!opened) {
    console.error('resume: cannot locate image, abort');
    // 保留 viewer tab，供下次续采用
    ws.close();
    process.exit(3);
  }
  await sleep(1200);
  console.error('viewer_open=' + opened + ' ff=' + needFF);
  if (needFF > 0) {
    for (let i = 0; i < needFF; i++) {
      await pressRight();
      await sleep(FF_MS);
      if ((i + 1) % 200 === 0) console.error(`  ff ${i + 1}/${needFF}`);
    }
    // 校验实际 index，防止快进丢事件/超调
    let cur = await curIndex();
    let guard = 0;
    while (cur >= 0 && cur < N - 1 && guard < 600) {
      await pressRight(); await sleep(90); cur = await curIndex(); guard++;
    }
    while (cur > N - 1 && guard < 1200) {
      await pressLeft(); await sleep(90); cur = await curIndex(); guard++;
    }
    console.error(`  ff position=${cur} target=${N - 1}`);
  }

  }

  const EXTRACT = `(()=>{
    const big = document.querySelector('[class*="_imgElm"] img');
    if (!big) return null;
    let root = big.closest('div._wrap_100l0_2');
    if (!root) {
      let p = big;
      while (p && !/来自|发布于/.test(p.innerText || '')) p = p.parentElement;
      root = p || document.body;
    }
    const hm = String(location.href).match(/[?&]index=(\\d+)/);
    return { src: big.currentSrc || big.src, text: (root.innerText || '').slice(0, 3000), idx: hm ? parseInt(hm[1], 10) : -1 };
  })()`;

  const waitNext = async (prevIdx, prevSrc) => {
    let cur = prevIdx;
    for (let k = 0; k < 50; k++) {
      await sleep(350);
      const c = await ev(EXTRACT);
      if (!c) continue;
      if (c.idx > cur) {
        cur = c.idx;
        if (c.src && tidy(c.src) !== prevSrc) return c;
      } else if (k % 5 === 4) {
        await pressRight();
      }
    }
    return null;
  };

  const rows = existing.slice();
  const seen = new Set(existing.map(r => r.url));
  let prev = '', same = 0, miss = 0, stale = 0, attempts = 0, albumEnd = false;
  const need = COUNT - rows.length;
  const maxAttempts = need * 4 + 100;
  while (rows.length < COUNT && attempts < maxAttempts) {
    attempts++;
    let isNew = false;
    const v = await ev(EXTRACT);
    let url = '';
    if (v && v.src && /sinaimg\.cn/.test(v.src)) {
      url = tidy(v.src);
      if (!seen.has(url)) {
        isNew = true;
        seen.add(url);
        const cap = (v.text || '').replace(/\s+/g, ' ').trim();
        const celeb = matchCeleb(cap);
        const row = { url, celeb, caption: cap };
        rows.push(row);
        fs.appendFileSync(tsv, `${row.url}\t${row.celeb}\t${row.caption}\n`);
        writeProgress(rows.length, url);
        console.error(`[${rows.length}/${COUNT}] ${celeb} ${url}`);
      }
    }
    if (isNew) stale = 0; else stale++;
    if (url && url === prev) same++; else same = 0;
    prev = url;
    if (!url) miss++; else miss = 0;
    if (same >= 6 || stale >= 30) { albumEnd = true; console.error('album end: no new image'); break; }
    if (miss >= 6) { console.error('stop: no page'); break; }
    await pressRight();
    await waitNext(v && v.idx >= 0 ? v.idx : -1, url || (v && v.src ? tidy(v.src) : ''));
  }
  writeProgress(rows.length, rows.length ? rows[rows.length - 1].url : '', albumEnd || rows.length >= COUNT);
  console.error(`collected=${rows.length}/${COUNT}${albumEnd ? ' (album end)' : ''} -> ${tsv}`);

  if (NO_DOWNLOAD) {
    // 保留 viewer tab，供下次续采用
    ws.close();
    process.exit(0);
  }

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

  // 下载后立即 resize（长边 <= MAX，只缩不放；4 路并行）
  if (process.env.RESIZE !== '0') {
    const dirs = [...new Set(jobs.map(j => path.dirname(j.path)))];
    const files = [];
    for (const d of dirs) {
      for (const f of fs.readdirSync(d)) {
        if (/\.(jpe?g|png|webp)$/i.test(f)) files.push(path.join(d, f));
      }
    }
    let ri = 0;
    async function resizeWorker() {
      while (true) {
        const k = ri++; if (k >= files.length) return;
        try {
          await execFileAsync('mogrify', ['-resize', `${MAX}x${MAX}>`, '-quality', '95', files[k]], { timeout: 120000 });
        } catch (e) { console.error('resize FAIL', files[k], e.message); }
      }
    }
    await Promise.all(Array.from({ length: 4 }, resizeWorker));
    console.error(`resize done ${files.length} (max ${MAX})`);
  }

  // 保留 viewer tab，供下次续采用
  ws.close();
  process.exit(0);
})().catch(e => { console.error('ERR', e.message || JSON.stringify(e)); process.exit(1); });
