'use strict';
(() => {
  const choices = [
    ['unreviewed', '待判断'],
    ['correct', '正确'],
    ['incorrect', '错误'],
  ];

  window.createHistoryReviewControl = record => {
    const control = document.createElement('section');
    control.className = 'review-control';
    control.setAttribute('aria-label', `记录 ${record.history_id} 的人工判断`);
    const heading = document.createElement('span');
    heading.className = 'review-heading';
    heading.textContent = '人工判断';
    const switcher = document.createElement('div');
    switcher.className = 'review-switch';
    switcher.setAttribute('role', 'group');
    switcher.setAttribute('aria-label', '位姿判断');
    const feedback = document.createElement('span');
    feedback.className = 'review-feedback';
    feedback.setAttribute('role', 'status');
    feedback.textContent = record.review_error || '';
    let status = choices.some(([value]) => value === record.review_status) ? record.review_status : 'unreviewed';

    const paint = () => {
      for (const button of switcher.children) {
        const selected = button.dataset.status === status;
        button.setAttribute('aria-pressed', String(selected));
        button.classList.toggle('selected', selected);
      }
      control.dataset.status = status;
      control.title = record.review_updated_at_ns && status !== 'unreviewed'
        ? `判断时间 ${new Date(Number(BigInt(record.review_updated_at_ns) / 1000000n)).toLocaleString('zh-CN', { timeZone: 'Asia/Shanghai', hour12: false })}`
        : '尚未保存人工判断';
    };

    for (const [value, label] of choices) {
      const button = document.createElement('button');
      button.type = 'button';
      button.dataset.status = value;
      button.textContent = label;
      button.onclick = async () => {
        if (value === status) return;
        const previous = status;
        status = value;
        feedback.textContent = '正在保存…';
        for (const option of switcher.children) option.disabled = true;
        paint();
        try {
          const response = await fetch(`/6d_pose_history/api/records/${record.history_id}/review`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ status }),
          });
          const data = await response.json();
          if (!response.ok) throw new Error(data.message || `HTTP ${response.status}`);
          status = data.review_status;
          record.review_updated_at_ns = data.review_updated_at_ns;
          delete record.review_error;
          feedback.textContent = status === 'unreviewed' ? '已清除判断' : '已保存';
          paint();
        } catch (error) {
          status = previous;
          feedback.textContent = `保存失败：${error.message}`;
          paint();
        } finally {
          for (const option of switcher.children) option.disabled = false;
        }
      };
      switcher.append(button);
    }
    control.append(heading, switcher, feedback);
    paint();
    return control;
  };
})();
