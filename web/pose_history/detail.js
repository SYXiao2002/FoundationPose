'use strict';
(async () => {
  const id = location.pathname.split('/').filter(Boolean).pop();
  const root = `/6d_pose_history/api/records/${id}`;
  const container = document.getElementById('history-images');
  try {
    const response = await fetch(root); const data = await response.json();
    if (!response.ok) throw new Error(data.message);
    const heading = document.createElement('strong');
    heading.textContent = new Date(Number(BigInt(id) / 1000000n)).toLocaleString('zh-CN', { timeZone: 'Asia/Shanghai', hour12: false });
    container.append(heading);
    const meta = document.createElement('p'); meta.className = 'history-meta';
    meta.textContent = `${data.status} · ${data.camera_frame || '未知相机'} · ${data.instance_id || data.mask_source || '未知实例'} · ID ${id}`;
    container.append(meta);
    container.append(window.createHistoryReviewControl(data));
    const candidates = document.getElementById('history-score-candidates');
    const candidateStatus = document.getElementById('score-candidates-status');
    if (data.archive_status === 'complete') {
      const link = document.createElement('a');
      link.href = `${root}/preview/score-top30`; link.target = '_blank'; link.rel = 'noopener';
      link.title = '打开前三十候选原图';
      const img = document.createElement('img'); img.id = 'score-candidates-image';
      img.alt = 'vis_score 按评分降序排列的前 30 个候选，含候选 ID、分数及 RGB / Depth 对比';
      img.onload = () => { candidateStatus.textContent = '最多显示 30 个候选 · 点击图片查看原图'; };
      img.onerror = () => { link.hidden = true; candidateStatus.textContent = '本次记录的评分候选图暂不可用'; };
      link.append(img); candidates.append(link);
      img.src = link.href;
    } else {
      candidateStatus.textContent = data.status === 'running' ? '解算进行中，完成后自动刷新' : '本次记录没有可用的评分候选图';
    }
    if (data.status !== 'complete' || data.preview_status !== 'complete') {
      const note = document.createElement('p'); note.className = 'history-notice';
      note.textContent = data.result?.message || (data.status === 'running' ? '解算进行中，完成后自动刷新' : '部分产物未生成'); container.append(note);
      if (data.status === 'running') { setTimeout(() => location.reload(), 3000); return; }
    }
    for (const [name, title, available] of [['RGB.png', '原始 RGB', data.archive_status === 'complete'], ['vis_score.png', '最高分候选', data.preview_status === 'complete']]) {
      if (!available) continue;
      const figure = document.createElement('figure'), caption = document.createElement('figcaption'); caption.textContent = title;
      const link = document.createElement('a'); link.href = `${root}/files/${name}`; link.target = '_blank'; link.rel = 'noopener';
      const img = document.createElement('img'); img.alt = title; img.src = name === 'RGB.png' ? link.href : `${root}/preview/score`;
      link.append(img); figure.append(caption, link); container.append(figure);
    }
    if (data.result?.T_camera_object) {
      const details = document.createElement('details'), summary = document.createElement('summary'); summary.textContent = '4×4 位姿矩阵与记录';
      const pre = document.createElement('pre'); pre.textContent = JSON.stringify({ T_camera_object: data.result.T_camera_object,
        score: data.result.score, model_sha256: data.model_sha256, capture_timestamp_ns: data.capture_timestamp_ns,
        parameters: data.parameters }, null, 2);
      details.append(summary, pre); container.append(details);
      const download = document.createElement('a'); download.href = `${root}/files/pose.txt`; download.textContent = '下载 pose.txt'; container.append(download);
    }
  } catch (error) { container.textContent = `历史读取失败：${error.message}`; }
})();
