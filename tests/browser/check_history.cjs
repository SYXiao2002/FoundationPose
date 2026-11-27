// Run against a history service containing at least one completed record.
const { chromium } = require('@playwright/test');
const assert = require('node:assert/strict');
const base = process.argv[2] || 'http://127.0.0.1:8095';
(async () => {
  const browser = await chromium.launch({ executablePath: process.env.PLAYWRIGHT_CHROME_PATH || '/usr/bin/google-chrome',
    headless: true, args: ['--no-sandbox', '--enable-unsafe-swiftshader'] });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errors = []; page.on('pageerror', error => errors.push(error.message));
    await page.goto(`${base}/6d_pose_history/?status=complete`);
    await page.waitForSelector('.history-record');
    await page.waitForFunction(() => [...document.querySelector('.history-record').querySelectorAll('img')].every(img => img.complete && img.naturalWidth > 0));
    await page.screenshot({ path: '/tmp/pose-history-list.png' });
    await page.locator('.history-record').first().click();
    await page.waitForFunction(() => document.querySelector('#scene-plot')._fullLayout?.scene && document.querySelector('#point-count').textContent.includes('个点'));
    const detailURL = page.url();
    await page.waitForFunction(() => { const img = document.querySelector('#score-candidates-image'); return img?.complete && img.naturalHeight > 0; });
    assert.equal(await page.locator('#history-images img[alt="最高分候选"]').evaluate(img => img.naturalHeight), 100);
    assert.ok(await page.locator('#score-candidates-image').evaluate(img => img.naturalHeight <= 1045));
    const plotBox = await page.locator('.viewer-area').boundingBox();
    const candidatesBox = await page.locator('#history-score-candidates').boundingBox();
    assert.ok(candidatesBox.x >= plotBox.x + plotBox.width - 1);
    assert.match(await page.locator('#score-candidates-image').getAttribute('src'), /\/preview\/score-top10$/);
    assert.equal(await page.locator('#history-score-candidates').evaluate(el => getComputedStyle(el).overflowY), 'auto');
    assert.equal(await page.locator('#show-model').isChecked(), true);
    await page.waitForFunction(() => document.querySelector('#scene-plot').data.some(trace => trace.type === 'mesh3d' && trace.visible === true));
    await page.locator('#crop-cloud').uncheck();
    await page.locator('#side-view').click();
    await page.locator('#focus-selection').click();
    await page.locator('#expand-mask').click(); await page.locator('#close-mask').click();
    await page.screenshot({ path: '/tmp/pose-history-detail.png' });
    assert.notEqual(await page.locator('.instance-label').innerText(), '');
    await page.reload(); await page.waitForFunction(() => document.querySelector('#scene-plot')._fullLayout?.scene);
    assert.equal(await page.locator('#show-model').isChecked(), true);
    await page.waitForFunction(() => document.querySelector('#scene-plot').data.some(trace => trace.type === 'mesh3d' && trace.visible === true));
    await page.goBack(); await page.waitForSelector('.history-record');
    await page.locator('#object').fill('does-not-exist'); await page.locator('#filters button[type=submit]').click();
    await page.waitForFunction(() => document.querySelector('#message').textContent.includes('没有符合'));
    await page.setViewportSize({ width: 412, height: 915 }); await page.goto(`${base}/6d_pose_history/`);
    await page.waitForSelector('.history-record');
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    await page.waitForFunction(() => [...document.querySelector('.history-record').querySelectorAll('img')].every(img => img.complete && img.naturalWidth > 0));
    await page.screenshot({ path: '/tmp/pose-history-mobile.png' });
    await page.goto(detailURL);
    await page.waitForFunction(() => { const img = document.querySelector('#score-candidates-image'); return img?.complete && img.naturalHeight > 0; });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    await page.locator('#history-score-candidates').scrollIntoViewIfNeeded();
    await page.screenshot({ path: '/tmp/pose-history-candidates-mobile.png' });
    assert.deepEqual(errors, []);
    console.log('History list, RGB/score images, default OBJ rendering (including reload), filters, 3D controls, deep link, back navigation and mobile viewport passed.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
