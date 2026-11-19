'use strict';
const $ = id => document.getElementById(id);
const labels = { complete: '完成', running: '运行中', failed: '失败', interrupted: '中断' };
let cursor = null, generation = 0, pollTimer;
function timeText(ns) {
  const milliseconds = Number(BigInt(ns) / 1000000n);
  return new Intl.DateTimeFormat('zh-CN', { timeZone: 'Asia/Shanghai', year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit', fractionalSecondDigits: 3, hour12: false }).format(new Date(milliseconds));
}
function element(tag, text, className) {
  const node = document.createElement(tag); node.textContent = text;
  if (className) node.className = className;
  return node;
}
function preview(record, kind, caption, available) {
  const figure = document.createElement('figure'); figure.append(element('figcaption', caption));
  const frame = element('div', available ? '' : (record.status === 'running' ? '正在生成…' : '未生成预览'), 'history-preview-frame');
  if (available) {
    const img = document.createElement('img'); img.alt = caption; img.loading = 'lazy';
    img.src = `/6d_pose_history/api/records/${record.history_id}/preview/${kind}`;
    img.onerror = () => { frame.replaceChildren(element('span', '预览不可用')); };
    frame.append(img);
  }
  figure.append(frame); return figure;
}
function render(record) {
  const card = document.createElement('a'); card.className = 'history-record'; card.href = record.history_path;
  const head = element('div', '', 'history-record-head');
  const time = element('time', timeText(record.history_id)); time.title = `UTC ${new Date(Number(BigInt(record.history_id) / 1000000n)).toISOString()} · ID ${record.history_id}`;
  head.append(time, element('strong', record.object_id), element('span', labels[record.status] || record.status, `history-badge ${record.status}`));
  if (record.score != null) head.append(element('span', `Score ${record.score.toFixed(3)}`));
  card.append(head, element('div', `${record.instance_id || record.mask_source || '未知实例'} · ${record.image_size.join(' × ')} · ID ${record.history_id}`, 'history-meta'));
  if (record.archive_status === 'failed') card.append(element('p', '归档失败，部分产物不可用'));
  const images = element('div', '', 'history-previews');
  images.append(preview(record, 'rgb', '原始 RGB', record.archive_status === 'complete'),
    preview(record, 'score', '最高分候选 · RGB / Depth 对比', record.preview_status === 'complete'));
  card.append(images); return card;
}
async function load() {
  const current = ++generation; clearTimeout(pollTimer); $('more').disabled = true; $('refresh').disabled = true;
  $('message').textContent = '正在加载历史…';
  try {
    const query = new URLSearchParams(location.search); query.set('limit', '30');
    const response = await fetch(`/6d_pose_history/api/records?${query}`);
    if (!response.ok) throw new Error((await response.json()).message || `HTTP ${response.status}`);
    const data = await response.json(); if (current !== generation) return;
    $('records').replaceChildren(...data.records.map(render)); cursor = data.next_cursor;
    $('message').textContent = data.records.length ? `本页 ${data.records.length} 条 · 上海时间 · 分数为原始网络输出` : '没有符合条件的历史记录';
    $('more').hidden = !cursor;
    if (data.records.some(record => record.status === 'running')) pollTimer = setTimeout(load, 3000);
  } catch (error) {
    if (current === generation) { $('records').replaceChildren(); $('message').textContent = `加载失败：${error.message}`; }
  } finally { if (current === generation) { $('more').disabled = false; $('refresh').disabled = false; } }
}
const params = new URLSearchParams(location.search);
for (const [id, key] of [['object', 'object_id'], ['status', 'status']]) $(id).value = params.get(key) || '';
for (const id of ['from', 'to']) {
  const ns = params.get(id);
  if (ns && /^\d+$/.test(ns)) $(id).value = new Date(Number(BigInt(ns) / 1000000n) + 8 * 3600000).toISOString().slice(0, 19);
}
$('filters').onsubmit = event => {
  event.preventDefault(); const query = new URLSearchParams();
  for (const id of ['from', 'to']) if ($(id).value) query.set(id, String(BigInt(Date.parse(`${$(id).value}+08:00`)) * 1000000n));
  for (const [id, key] of [['object', 'object_id'], ['status', 'status']]) if ($(id).value) query.set(key, $(id).value);
  history.pushState(null, '', `?${query}`); load();
};
$('clear').onclick = () => { location.href = '/6d_pose_history/'; };
$('refresh').onclick = load;
$('more').onclick = () => { const query = new URLSearchParams(location.search); query.set('before', cursor); history.pushState(null, '', `?${query}`); load(); window.scrollTo(0, 0); };
window.onpopstate = () => location.reload();
load();
