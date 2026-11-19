// Verify an upgrade in a browser that has already cached the old two-column UI.
// A real HTTP proxy primes the browser cache; Playwright routing disables caching.
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { chromium } = require('@playwright/test');
const base = process.argv[2] || 'http://127.0.0.1:8765';

(async () => {
  const records = await (await fetch(`${base}/6d_pose_history/api/records?limit=1&status=complete`)).json();
  const detailPath = records.records[0].history_path;
  const html = await (await fetch(base + detailPath)).text();
  const legacyHTML = html.replace(/(\.css|\.js)\?v=[a-f0-9]+/g, '$1')
    .replace('<body class="history-detail">', '<body>')
    .replace(/<aside id="history-score-candidates"[\s\S]*?<\/aside>/, '');
  const webRoot = path.join(__dirname, '../../web/pose_history');
  const legacyJS = fs.readFileSync(path.join(webRoot, 'detail.js'), 'utf8')
    .replace(/    const candidates =[\s\S]*?(?=    if \(data.status)/, '');
  const legacyCSS = fs.readFileSync(path.join(webRoot, 'style.css'), 'utf8').split('.history-detail')[0];
  let upgraded = false;
  const staleRequests = [];
  const server = http.createServer(async (request, response) => {
    try {
      const url = new URL(request.url, 'http://local');
      if (url.pathname === detailPath) {
        response.writeHead(200, { 'Content-Type': 'text/html', 'Cache-Control': 'no-store' });
        response.end(upgraded ? html : legacyHTML); return;
      }
      if (!url.search && ['/6d_pose_history/assets/detail.js', '/6d_pose_history/assets/history.css'].includes(url.pathname)) {
        staleRequests.push({ upgraded, url: request.url });
        const isJS = url.pathname.endsWith('.js');
        response.writeHead(200, { 'Content-Type': isJS ? 'application/javascript' : 'text/css',
          'Cache-Control': 'public, max-age=3600' });
        response.end(isJS ? legacyJS : legacyCSS); return;
      }
      const upstream = await fetch(base + request.url);
      const headers = {};
      for (const key of ['content-type', 'cache-control']) {
        if (upstream.headers.has(key)) headers[key] = upstream.headers.get(key);
      }
      response.writeHead(upstream.status, headers);
      response.end(Buffer.from(await upstream.arrayBuffer()));
    } catch (error) { response.writeHead(500); response.end(String(error)); }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  let browser;
  try {
    browser = await chromium.launch({ executablePath: process.env.PLAYWRIGHT_CHROME_PATH || '/usr/bin/google-chrome',
      headless: true, args: ['--no-sandbox', '--enable-unsafe-swiftshader'] });
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errors = []; page.on('pageerror', error => errors.push(error.message));
    const origin = `http://127.0.0.1:${server.address().port}`;
    await page.goto(origin + detailPath + '?prime-cache=1');
    await page.waitForFunction(() => document.querySelector('#history-images img[alt="最高分候选"]')?.naturalHeight === 100);
    assert.equal(staleRequests.length, 2);
    upgraded = true;
    await page.goto(origin + detailPath + '?after-upgrade=1');
    await page.waitForFunction(() => document.querySelector('#score-candidates-image')?.naturalHeight > 0);
    assert.equal(staleRequests.filter(request => request.upgraded).length, 0);
    for (const width of [1440, 1024, 412]) {
      await page.setViewportSize({ width, height: 1000 });
      await page.waitForFunction(() => document.documentElement.scrollWidth <= innerWidth);
      // Plotly resizes asynchronously after its container changes width.
      await page.waitForFunction(() => {
        const plot = document.querySelector('#scene-plot');
        return plot._fullLayout?.scene && Math.abs(plot._fullLayout.width - plot.clientWidth) <= 1;
      });
      const viewer = await page.locator('.viewer-area').boundingBox();
      const panel = await page.locator('#history-score-candidates').boundingBox();
      if (width > 900) {
        assert.ok(panel.x >= viewer.x + viewer.width - 1, `candidates must be on the right at ${width}px`);
        assert.ok(Math.abs(panel.y - viewer.y) < 1);
      } else { assert.ok(panel.y >= viewer.y + viewer.height - 1); }
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
      assert.ok(await page.locator('#score-candidates-image').evaluate(img => img.naturalHeight <= 3145));
      assert.equal(await page.locator('#history-images img[alt="最高分候选"]').evaluate(img => img.naturalHeight), 100);
      await page.locator('#history-score-candidates').scrollIntoViewIfNeeded();
      await page.screenshot({ path: `/tmp/history-cache-fixed-${width}.png` });
    }
    assert.deepEqual(errors, []);
    console.log('Cached UI upgrade, candidate images and desktop/mobile layout passed.');
  } finally {
    if (browser) await browser.close();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
