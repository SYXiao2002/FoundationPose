'use strict';
const state = {
  info: null, model: null, cloud: null, selected: new Set(), hiddenModels: new Set(),
  traceIndices: new Map(), maskImage: null, maskPixels: null, ready: false,
};
const $ = selector => document.querySelector(selector);
const plot = $('#scene-plot');
const axisColors = ['#db514b', '#38a45a', '#2583bc'];
const frontCamera = { eye: { x: 0, y: 0, z: -1.65 }, up: { x: 0, y: -1, z: 0 } };

function setError(message) {
  $('#error-banner').textContent = message;
  $('#error-banner').hidden = false;
}

async function get(url, type = 'json') {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response[type]();
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('无法加载实例 Mask 图'));
    image.src = url;
  });
}

async function loadInputs() {
  let info, sceneBuffer, model, maskSource, mapSource;
  if (window.POSE_DATA) {
    const embedded = window.POSE_DATA;
    info = embedded.info;
    model = embedded.model;
    sceneBuffer = Uint8Array.from(atob(embedded.cloud), char => char.charCodeAt(0)).buffer;
    maskSource = embedded.mask;
    mapSource = embedded.maskMap;
  } else {
    [info, sceneBuffer, model] = await Promise.all([
      get('/api/info'), get('/api/scene', 'arrayBuffer'), get('/api/model'),
    ]);
    maskSource = '/api/mask.jpg';
    mapSource = '/api/masks.png';
  }
  const count = new DataView(sceneBuffer).getUint32(0, true);
  if (sceneBuffer.byteLength !== 4 + count * 15) throw new Error('点云数据长度错误');
  state.info = info;
  state.model = model;
  state.cloud = {
    count, positions: new Float32Array(sceneBuffer, 4, count * 3),
    colors: new Uint8Array(sceneBuffer, 4 + count * 12, count * 3),
  };
  const [maskImage, maskMap] = await Promise.all([loadImage(maskSource), loadImage(mapSource)]);
  state.maskImage = maskImage;
  const canvas = document.createElement('canvas');
  [canvas.width, canvas.height] = info.image_size;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  context.drawImage(maskMap, 0, 0);
  state.maskPixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
  state.bounds = objectBounds(info.instances);
  state.selected = new Set([info.instances[0].id]);
}

function transformedVertices(pose) {
  return state.model.vertices.map(([x, y, z]) => [0, 1, 2].map(row =>
    pose[row][0] * x + pose[row][1] * y + pose[row][2] * z + pose[row][3]));
}

function objectBounds(instances, padding = 0.045) {
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (const instance of instances) {
    for (const vertex of transformedVertices(instance.pose)) {
      for (let axis = 0; axis < 3; axis++) {
        min[axis] = Math.min(min[axis], vertex[axis] - padding);
        max[axis] = Math.max(max[axis], vertex[axis] + padding);
      }
    }
  }
  return { min, max };
}

function renderInstanceList() {
  const list = $('#instance-list');
  for (const instance of state.info.instances) {
    const row = document.createElement('div');
    row.className = 'instance-row';
    row.dataset.id = instance.id;
    row.title = `t = [${instance.origin.map(v => v.toFixed(5)).join(', ')}] m` +
      (instance.score == null ? '' : ` · Register score: ${instance.score.toFixed(3)}`);
    row.innerHTML = `<label class="instance-choice"><input type="checkbox" class="pose-checkbox" value="${instance.id}" aria-label="显示 instance ${instance.id}" checked><span class="swatch" style="background:${instance.color}"></span><span class="instance-label">Instance ${instance.id}</span></label><label class="mesh-choice" title="该实例的 OBJ 显示开关"><input type="checkbox" class="mesh-checkbox" aria-label="instance ${instance.id} 的 OBJ" checked>OBJ</label>`;
    row.querySelector('.pose-checkbox').addEventListener('change', event => {
      if (event.target.checked) state.selected.add(instance.id);
      else state.selected.delete(instance.id);
      updateSelection();
    });
    row.querySelector('.mesh-checkbox').addEventListener('change', event => {
      if (event.target.checked) state.hiddenModels.delete(instance.id);
      else state.hiddenModels.add(instance.id);
      updateVisibility();
    });
    list.append(row);
  }
}

function setSelection(ids) {
  state.selected = new Set(ids);
  updateSelection();
}

function updateSelection() {
  const selected = state.info.instances.filter(instance => state.selected.has(instance.id));
  $('#selection-count').textContent = `已选 ${selected.length} / ${state.info.instances.length}`;
  document.querySelectorAll('.instance-row').forEach(row => {
    const checked = state.selected.has(Number(row.dataset.id));
    row.classList.toggle('is-selected', checked);
    row.querySelector('.pose-checkbox').checked = checked;
  });
  const readout = $('#pose-readout');
  readout.replaceChildren(...selected.map(instance => {
    const chip = document.createElement('span');
    chip.className = 'pose-chip';
    chip.innerHTML = `<strong style="color:${instance.color}">#${instance.id}</strong>${instance.origin.map(value => value.toFixed(3)).join(', ')} m`;
    return chip;
  }));
  if (!selected.length) readout.textContent = '勾选或在 Mask 图上框选实例，显示对应的 pose。';
  drawMask($('#mask-selection'));
  drawMask($('#large-mask'), true);
  updateVisibility();
}

function updateVisibility() {
  if (!state.ready) return;
  const indices = [], visibility = [];
  for (const instance of state.info.instances) {
    const traces = state.traceIndices.get(instance.id);
    const selected = state.selected.has(instance.id);
    indices.push(traces.mesh, ...traces.axes, traces.origin);
    const axes = selected && $('#show-axes').checked;
    visibility.push(selected && $('#show-model').checked && !state.hiddenModels.has(instance.id), axes, axes, axes, axes);
  }
  Plotly.restyle(plot, { visible: visibility }, indices).catch(error => setError(error.message));
}

function cloudData() {
  const { positions, colors, count } = state.cloud;
  const x = [], y = [], z = [], rgb = [];
  const crop = $('#crop-cloud').checked;
  const { min, max } = state.bounds;
  for (let index = 0; index < count; index++) {
    const offset = index * 3;
    const point = [positions[offset], positions[offset + 1], positions[offset + 2]];
    if (crop && point.some((value, axis) => value < min[axis] || value > max[axis])) continue;
    x.push(point[0]); y.push(point[1]); z.push(point[2]);
    rgb.push(`rgb(${colors[offset]},${colors[offset + 1]},${colors[offset + 2]})`);
  }
  $('#point-count').textContent = `${x.length.toLocaleString()} / ${count.toLocaleString()} 个点`;
  return { x, y, z, rgb };
}

function axisTrace(instance, axis) {
  const origin = instance.origin;
  const length = Number($('#axis-size').value) / 1000;
  const end = origin.map((value, row) => value + instance.pose[row][axis] * length);
  return {
    type: 'scatter3d', mode: 'lines+text',
    x: [origin[0], end[0]], y: [origin[1], end[1]], z: [origin[2], end[2]],
    line: { color: axisColors[axis], width: 5 },
    text: ['', ['X', 'Y', 'Z'][axis]], textfont: { color: axisColors[axis], size: 10 },
    textposition: 'top center', hoverinfo: 'skip', showlegend: false,
    visible: state.selected.has(instance.id) && $('#show-axes').checked,
  };
}

async function drawScene() {
  const cloud = cloudData();
  const traces = [{
    type: 'scatter3d', mode: 'markers', x: cloud.x, y: cloud.y, z: cloud.z,
    marker: { size: Number($('#point-size').value), color: cloud.rgb, opacity: 1 },
    hovertemplate: 'RGB-D<br>x=%{x:.3f} m<br>y=%{y:.3f} m<br>z=%{z:.3f} m<extra></extra>',
    name: 'RGB-D', showlegend: false,
  }];
  const faces = state.model.faces;
  const i = faces.map(face => face[0]), j = faces.map(face => face[1]), k = faces.map(face => face[2]);
  for (const instance of state.info.instances) {
    const vertices = transformedVertices(instance.pose);
    const mesh = traces.length;
    traces.push({
      type: 'mesh3d', x: vertices.map(v => v[0]), y: vertices.map(v => v[1]), z: vertices.map(v => v[2]), i, j, k,
      color: instance.color, opacity: Number($('#model-opacity').value) / 100,
      flatshading: true, lighting: { ambient: 0.72, diffuse: 0.65, specular: 0.12, roughness: 0.85 },
      hoverinfo: 'skip', name: `Instance ${instance.id} OBJ`, showlegend: false, visible: false,
    });
    const axes = [0, 1, 2].map(axis => {
      const index = traces.length;
      traces.push(axisTrace(instance, axis));
      return index;
    });
    const origin = traces.length;
    traces.push({
      type: 'scatter3d', mode: 'markers+text',
      x: [instance.origin[0]], y: [instance.origin[1]], z: [instance.origin[2]], text: [`#${instance.id}`],
      textposition: 'bottom center', textfont: { size: 12, color: instance.color },
      marker: { size: 4, color: instance.color, line: { color: '#ffffff', width: 1 } },
      hovertemplate: `Instance ${instance.id}<br>x=%{x:.4f} m<br>y=%{y:.4f} m<br>z=%{z:.4f} m<extra></extra>`,
      name: `Instance ${instance.id}`, showlegend: false,
      visible: state.selected.has(instance.id),
    });
    state.traceIndices.set(instance.id, { mesh, axes, origin });
  }
  const axisStyle = { showbackground: true, backgroundcolor: '#f5f7f4', gridcolor: '#d9e0d9',
    zerolinecolor: '#c5cec6', color: '#69736c', tickfont: { size: 10 } };
  await Plotly.newPlot(plot, traces, {
    autosize: true, uirevision: 'first-frame', margin: { l: 0, r: 0, t: 2, b: 0 },
    showlegend: false, paper_bgcolor: '#fafbf9',
    scene: {
      aspectmode: 'data', bgcolor: '#fafbf9', camera: frontCamera,
      xaxis: { ...axisStyle, title: { text: 'X (m)' } },
      yaxis: { ...axisStyle, title: { text: 'Y (m)' } },
      zaxis: { ...axisStyle, title: { text: 'Z (m)' } },
    },
  }, { responsive: true, displaylogo: false, scrollZoom: true, modeBarButtonsToRemove: ['toImage'] });
  state.ready = true;
  plot.on('plotly_click', event => {
    const traceIndex = event.points[0]?.curveNumber;
    for (const [id, indices] of state.traceIndices) {
      if ([indices.mesh, indices.origin, ...indices.axes].includes(traceIndex)) {
        setSelection(event.event?.shiftKey ? [...state.selected, id] : [id]);
        break;
      }
    }
  });
  new ResizeObserver(() => Plotly.Plots.resize(plot)).observe(plot);
}

function maskID(x, y) {
  const [width, height] = state.info.image_size;
  x = Math.max(0, Math.min(width - 1, Math.floor(x)));
  y = Math.max(0, Math.min(height - 1, Math.floor(y)));
  const offset = (y * width + x) * 4;
  const pixels = state.maskPixels;
  return pixels[offset] + (pixels[offset + 1] << 8) + (pixels[offset + 2] << 16);
}

function drawMask(canvas, withImage = false, rectangle = null) {
  if (!state.maskPixels) return;
  const context = canvas.getContext('2d');
  context.clearRect(0, 0, canvas.width, canvas.height);
  if (withImage) context.drawImage(state.maskImage, 0, 0);
  const [width, height] = state.info.image_size;
  const shade = context.createImageData(width, height);
  for (let offset = 0; offset < state.maskPixels.length; offset += 4) {
    const id = state.maskPixels[offset] + (state.maskPixels[offset + 1] << 8) + (state.maskPixels[offset + 2] << 16);
    if (id && !state.selected.has(id)) {
      shade.data[offset] = 24; shade.data[offset + 1] = 30; shade.data[offset + 2] = 26; shade.data[offset + 3] = 165;
    }
  }
  const overlay = document.createElement('canvas');
  overlay.width = width; overlay.height = height;
  overlay.getContext('2d').putImageData(shade, 0, 0);
  context.drawImage(overlay, 0, 0);
  if (rectangle) {
    context.fillStyle = '#287a5238'; context.strokeStyle = '#ffffff'; context.lineWidth = 2;
    const [start, end] = rectangle;
    context.fillRect(start.x, start.y, end.x - start.x, end.y - start.y);
    context.strokeRect(start.x, start.y, end.x - start.x, end.y - start.y);
  }
}

function bindMaskCanvas(canvas, withImage = false) {
  [canvas.width, canvas.height] = state.info.image_size;
  let start = null;
  const position = event => {
    const rect = canvas.getBoundingClientRect();
    return { x: (event.clientX - rect.left) / rect.width * canvas.width,
      y: (event.clientY - rect.top) / rect.height * canvas.height };
  };
  canvas.addEventListener('pointerdown', event => {
    if (event.button !== 0) return;
    start = position(event);
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener('pointermove', event => {
    if (start) drawMask(canvas, withImage, [start, position(event)]);
  });
  canvas.addEventListener('pointerup', event => {
    if (!start) return;
    const end = position(event);
    const previous = start;
    start = null;
    const additive = event.shiftKey || event.ctrlKey || event.metaKey;
    const selected = additive ? new Set(state.selected) : new Set();
    if (Math.hypot(end.x - previous.x, end.y - previous.y) < 7) {
      const id = maskID(end.x, end.y);
      if (id) {
        if (additive && state.selected.has(id)) selected.delete(id);
        else if (!additive && state.selected.size === 1 && state.selected.has(id)) selected.delete(id);
        else selected.add(id);
      }
    } else {
      const x0 = Math.max(0, Math.floor(Math.min(end.x, previous.x)));
      const x1 = Math.min(canvas.width - 1, Math.ceil(Math.max(end.x, previous.x)));
      const y0 = Math.max(0, Math.floor(Math.min(end.y, previous.y)));
      const y1 = Math.min(canvas.height - 1, Math.ceil(Math.max(end.y, previous.y)));
      for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
        const id = maskID(x, y);
        if (id) selected.add(id);
      }
    }
    setSelection(selected);
  });
  canvas.addEventListener('pointercancel', () => { start = null; drawMask(canvas, withImage); });
}

function bindControls() {
  $('#select-all').onclick = () => setSelection(state.info.instances.map(item => item.id));
  $('#clear-selection').onclick = () => setSelection([]);
  $('#invert-selection').onclick = () => setSelection(state.info.instances.filter(item => !state.selected.has(item.id)).map(item => item.id));
  $('#show-axes').onchange = updateVisibility;
  $('#show-model').onchange = updateVisibility;
  $('#model-opacity').oninput = event => {
    $('#opacity-value').value = `${event.target.value}%`;
    const indices = [...state.traceIndices.values()].map(item => item.mesh);
    Plotly.restyle(plot, { opacity: Number(event.target.value) / 100 }, indices);
  };
  $('#axis-size').oninput = event => {
    $('#axis-size-value').value = `${event.target.value} mm`;
    const indices = [], x = [], y = [], z = [];
    for (const instance of state.info.instances) {
      [0, 1, 2].forEach(axis => {
        const trace = axisTrace(instance, axis);
        indices.push(state.traceIndices.get(instance.id).axes[axis]);
        x.push(trace.x); y.push(trace.y); z.push(trace.z);
      });
    }
    Plotly.restyle(plot, { x, y, z }, indices);
  };
  $('#point-size').oninput = event => {
    $('#point-size-value').value = `${event.target.value} px`;
    Plotly.restyle(plot, { 'marker.size': Number(event.target.value) }, [0]);
  };
  $('#show-cloud').onchange = event => Plotly.restyle(plot, { visible: event.target.checked }, [0]);
  $('#crop-cloud').onchange = async () => {
    const cloud = cloudData();
    await Plotly.restyle(plot, { x: [cloud.x], y: [cloud.y], z: [cloud.z], 'marker.color': [cloud.rgb] }, [0]);
    await resetView();
  };
  $('#reset-view').onclick = resetView;
  $('#side-view').onclick = () => Plotly.relayout(plot, {
    'scene.camera': { eye: { x: 1.5, y: -0.55, z: -0.7 }, up: { x: 0, y: -1, z: 0 } },
  });
  $('#focus-selection').onclick = () => {
    const selected = state.info.instances.filter(item => state.selected.has(item.id));
    if (!selected.length) return;
    const bounds = objectBounds(selected, 0.025);
    const update = { 'scene.camera': frontCamera };
    ['x', 'y', 'z'].forEach((axis, index) => { update[`scene.${axis}axis.range`] = [bounds.min[index], bounds.max[index]]; });
    Plotly.relayout(plot, update);
  };
  $('#expand-mask').onclick = () => { $('#mask-dialog').showModal(); drawMask($('#large-mask'), true); };
  $('#close-mask').onclick = () => $('#mask-dialog').close();
  bindMaskCanvas($('#mask-selection'));
  bindMaskCanvas($('#large-mask'), true);
}

function resetView() {
  return Plotly.relayout(plot, { 'scene.camera': frontCamera,
    'scene.xaxis.autorange': true, 'scene.yaxis.autorange': true, 'scene.zaxis.autorange': true });
}

async function init() {
  try {
    await loadInputs();
    $('#scene-name').textContent = state.info.scene;
    $('#frame-id').textContent = state.info.frame;
    $('#instance-total').textContent = String(state.info.instances.length);
    renderInstanceList();
    await drawScene();
    bindControls();
    updateSelection();
  } catch (error) {
    console.error(error);
    setError(`加载失败：${error.message}`);
  }
}

init();
