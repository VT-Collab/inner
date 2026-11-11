// figures for the project page; the data is in data.js
(function () {
  const NS = 'http://www.w3.org/2000/svg';
  const METHODS = ['BC', 'Nuclear', 'InneR'];
  const LABEL = { BC: 'Behavior cloning', Nuclear: 'Nuclear norm', InneR: 'InneR' };
  const COLOR = { BC: '#a8a8a8', Nuclear: '#7a5230', InneR: '#ff9900' };
  const INK = '#222', MUTED = '#6b6b6b', GRID = '#e6e6e6', BASE = '#9a9a9a';

  function svg(tag, attrs, parent, text) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text !== undefined) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function html(tag, cls, parent, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  const rate = (s, n) => 100 * s / n;
  const stderr = (s, n) => 100 * Math.sqrt((s / n) * (1 - s / n) / n);

  // bar with a rounded top
  function bar(parent, x, yTop, w, yBase, color) {
    const r = Math.max(0, Math.min(4, w / 2, yBase - yTop));
    const d = `M${x},${yBase} L${x},${yTop + r} Q${x},${yTop} ${x + r},${yTop} L${x + w - r},${yTop} ` +
      `Q${x + w},${yTop} ${x + w},${yTop + r} L${x + w},${yBase} Z`;
    return svg('path', { d, fill: color, class: 'mark' }, parent);
  }

  // one tooltip per figure
  function tooltip(fig) {
    const tip = html('div', 'tip', fig);
    tip.setAttribute('role', 'status');
    return {
      show(x, y, rows) {
        tip.textContent = '';
        rows.forEach(r => {
          const row = html('div', 'tip-row', tip);
          if (r.key) html('span', 'tip-key', row).style.background = r.key;
          if (r.value !== undefined) html('strong', '', row, r.value);
          if (r.text) html('span', r.value !== undefined ? 'tip-sub' : 'tip-head', row, r.text);
        });
        tip.style.display = 'block';
        const box = fig.getBoundingClientRect();
        const left = Math.min(Math.max(x - box.left + 14, 4), box.width - tip.offsetWidth - 4);
        tip.style.left = left + 'px';
        tip.style.top = Math.max(y - box.top - tip.offsetHeight - 10, 4) + 'px';
      },
      hide() { tip.style.display = 'none'; },
    };
  }

  // legend that switches methods on and off
  function legend(fig, hidden, redraw) {
    const row = html('div', 'legend', fig);
    METHODS.forEach(m => {
      const b = html('button', 'legend-item', row);
      b.type = 'button';
      b.setAttribute('aria-pressed', 'true');
      html('span', 'swatch', b).style.background = COLOR[m];
      html('span', '', b, LABEL[m]);
      b.addEventListener('click', () => {
        hidden.has(m) ? hidden.delete(m) : hidden.add(m);
        b.setAttribute('aria-pressed', String(!hidden.has(m)));
        redraw();
      });
    });
  }

  function tabs(fig, names, select) {
    const row = html('div', 'tab-row', fig);
    names.forEach((name, i) => {
      const b = html('button', 'tab' + (i === 0 ? ' is-on' : ''), row, name);
      b.type = 'button';
      b.addEventListener('click', () => {
        row.querySelectorAll('.tab').forEach(t => t.classList.remove('is-on'));
        b.classList.add('is-on');
        select(name);
      });
    });
  }

  function table(fig, head, rows) {
    const old = fig.querySelector('details.table-view');
    const open = old ? old.open : false;
    if (old) old.remove();
    const d = html('details', 'table-view', fig);
    d.open = open;
    html('summary', '', d, 'Data table');
    const t = html('table', 'table is-narrow is-fullwidth', d);
    const tr = html('tr', '', html('thead', '', t));
    head.forEach(h => html('th', '', tr, h));
    const body = html('tbody', '', t);
    rows.forEach(r => { const row = html('tr', '', body); r.forEach(c => html('td', '', row, c)); });
  }

  function yAxis(g, x0, x1, y, ticks, fmt, labels) {
    ticks.forEach(v => {
      svg('line', { x1: x0, x2: x1, y1: y(v), y2: y(v), stroke: v === 0 ? BASE : GRID, 'stroke-width': 1 }, g);
      if (labels) svg('text', { x: x0 - 8, y: y(v) + 4, 'text-anchor': 'end', class: 'tick' }, g, fmt(v));
    });
  }

  // success rate per task and number of distractors
  function successChart(fig) {
    const hidden = new Set(), tip = tooltip(fig);
    legend(fig, hidden, draw);
    const W = 900, H = 310, L = 52, R = 8, T = 34, B = 48, GAP = 30, BW = 22;
    const root = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Success rate by task, number of distractors and method' }, fig);
    const tasks = Object.keys(SUCCESS);
    const groups = tasks.map(t => Object.keys(SUCCESS[t]).length);
    const unit = (W - L - R - GAP * (tasks.length - 1)) / groups.reduce((a, b) => a + b, 0);
    const y = v => T + (H - T - B) * (1 - v / 100);

    function draw() {
      root.textContent = '';
      let x0 = L;
      tasks.forEach((task, k) => {
        const w = unit * groups[k], g = svg('g', {}, root);
        yAxis(g, x0, x0 + w, y, [0, 25, 50, 75, 100], v => v, k === 0);
        svg('text', { x: x0 + w / 2, y: 18, 'text-anchor': 'middle', class: 'panel-title' }, g, task);
        svg('text', { x: x0 + w / 2, y: H - 8, 'text-anchor': 'middle', class: 'axis-title' }, g, 'Distractors');
        Object.keys(SUCCESS[task]).forEach((d, i) => {
          const cx = x0 + unit * (i + 0.5);
          svg('text', { x: cx, y: H - B + 18, 'text-anchor': 'middle', class: 'tick' }, g, d);
          METHODS.forEach((m, j) => {
            if (hidden.has(m)) return;
            const [s, n] = SUCCESS[task][d][m], p = rate(s, n), e = stderr(s, n);
            const x = cx + (j - 1) * (BW + 2) - BW / 2;
            if (e > 0) svg('line', { x1: x + BW / 2, x2: x + BW / 2, y1: y(p), y2: y(Math.min(100, p + e)), stroke: MUTED, 'stroke-width': 1 }, g);
            const mark = bar(g, x, y(p), BW, y(0), COLOR[m]);
            svg('text', { x: x + BW / 2, y: y(Math.min(100, p + e)) - 5, 'text-anchor': 'middle', class: 'value' }, g, Math.round(p));
            const hit = svg('rect', { x: x - 1, y: T, width: BW + 2, height: H - T - B, fill: 'transparent', tabindex: 0, class: 'hit',
              'aria-label': `${LABEL[m]}, ${task}, ${d} distractors: ${s} of ${n} trials` }, g);
            const rows = [{ value: Math.round(p) + '%', text: `${s} of ${n} trials` }, { key: COLOR[m], text: LABEL[m] },
              { text: `${task}, ${d} distractor${d === '1' ? '' : 's'}` }];
            const on = ev => {
              mark.classList.add('is-hover');
              const b = hit.getBoundingClientRect();
              tip.show(ev.clientX || b.left + b.width / 2, ev.clientY || b.top + b.height * (1 - p / 100), rows);
            };
            const off = () => { mark.classList.remove('is-hover'); tip.hide(); };
            hit.addEventListener('pointermove', on); hit.addEventListener('focus', on);
            hit.addEventListener('pointerleave', off); hit.addEventListener('blur', off);
          });
        });
        x0 += w + GAP;
      });
      svg('text', { x: 14, y: T + (H - T - B) / 2, 'text-anchor': 'middle', class: 'axis-title', transform: `rotate(-90 14 ${T + (H - T - B) / 2})` }, root, 'Success rate (%)');
    }
    draw();
    const rows = [];
    tasks.forEach(t => Object.keys(SUCCESS[t]).forEach(d => rows.push([t, d].concat(METHODS.map(m => SUCCESS[t][d][m].join(' / '))))));
    table(fig, ['Task', 'Distractors'].concat(METHODS.map(m => LABEL[m] + ' (successes / trials)')), rows);
  }

  // success rate against observation noise
  function noiseChart(fig) {
    const hidden = new Set(), tip = tooltip(fig);
    legend(fig, hidden, draw);
    const W = 560, H = 310, L = 52, R = 20, T = 16, B = 48;
    const root = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Success rate against added observation noise' }, fig);
    const levels = Object.keys(NOISE).map(Number).sort((a, b) => a - b);
    const x = v => L + (W - L - R) * v / levels[levels.length - 1];
    const y = v => T + (H - T - B) * (1 - v / 100);
    const get = (v, m) => NOISE[String(v)][m];

    function draw() {
      root.textContent = '';
      const g = svg('g', {}, root);
      yAxis(g, L, W - R, y, [0, 25, 50, 75, 100], v => v, true);
      levels.forEach(v => svg('text', { x: x(v), y: H - B + 18, 'text-anchor': 'middle', class: 'tick' }, g, v));
      svg('text', { x: L + (W - L - R) / 2, y: H - 8, 'text-anchor': 'middle', class: 'axis-title' }, g, 'Added noise, standard deviation (mm)');
      svg('text', { x: 14, y: T + (H - T - B) / 2, 'text-anchor': 'middle', class: 'axis-title', transform: `rotate(-90 14 ${T + (H - T - B) / 2})` }, g, 'Success rate (%)');
      const hair = svg('line', { y1: T, y2: H - B, stroke: BASE, 'stroke-width': 1, visibility: 'hidden' }, g);
      METHODS.filter(m => !hidden.has(m)).forEach(m => {
        const pts = levels.map(v => [x(v), y(rate(...get(v, m)))]);
        svg('path', { d: 'M' + pts.map(p => p.join(',')).join(' L'), fill: 'none', stroke: COLOR[m], 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }, g);
        pts.forEach(p => svg('circle', { cx: p[0], cy: p[1], r: 4.5, fill: COLOR[m], stroke: '#fff', 'stroke-width': 2 }, g));
      });
      const over = svg('rect', { x: L - 10, y: T, width: W - L - R + 20, height: H - T - B, fill: 'transparent' }, g);
      over.addEventListener('pointermove', ev => {
        const b = root.getBoundingClientRect(), px = (ev.clientX - b.left) * W / b.width;
        const v = levels.reduce((a, c) => Math.abs(x(c) - px) < Math.abs(x(a) - px) ? c : a);
        hair.setAttribute('x1', x(v)); hair.setAttribute('x2', x(v)); hair.setAttribute('visibility', 'visible');
        const rows = [{ text: v === 0 ? 'No added noise' : `${v} mm of added noise` }];
        METHODS.filter(m => !hidden.has(m)).forEach(m => {
          const [s, n] = get(v, m);
          rows.push({ key: COLOR[m], value: Math.round(rate(s, n)) + '%', text: `${LABEL[m]}, ${s} of ${n}` });
        });
        tip.show(b.left + x(v) * b.width / W, ev.clientY, rows);
      });
      over.addEventListener('pointerleave', () => { hair.setAttribute('visibility', 'hidden'); tip.hide(); });
    }
    draw();
    table(fig, ['Added noise (mm)'].concat(METHODS.map(m => LABEL[m] + ' (successes / trials)')),
      levels.map(v => [String(v)].concat(METHODS.map(m => get(v, m).join(' / ')))));
  }

  // change in the commanded position when one object moves 1 cm
  function sensitivityChart(fig) {
    const hidden = new Set(), tip = tooltip(fig);
    let task = Object.keys(SENS)[0];
    tabs(fig, Object.keys(SENS), name => { task = name; draw(); });
    legend(fig, hidden, draw);
    const W = 900, H = 330, L = 58, R = 8, T = 34, B = 34, BW = 24;
    const root = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Sensitivity of each policy to each object' }, fig);
    const tiles = html('div', 'tiles', fig);

    function draw() {
      root.textContent = '';
      const S = SENS[task], n = S.objects.length, slot = (W - L - R) / n;
      const top = Math.max(...METHODS.map(m => Math.max(...S.mm[m])));
      const step = top > 2 ? 1 : 0.5, ymax = Math.ceil(top / step) * step;
      const y = v => T + (H - T - B) * (1 - v / ymax);
      const ticks = []; for (let v = 0; v <= ymax + 1e-9; v += step) ticks.push(v);
      const g = svg('g', {}, root);
      yAxis(g, L, W - R, y, ticks, v => v.toFixed(1), true);
      const split = L + slot * S.n_task;
      svg('line', { x1: split, x2: split, y1: T - 14, y2: H - B, stroke: GRID, 'stroke-width': 1 }, g);
      svg('text', { x: L + slot * S.n_task / 2, y: 16, 'text-anchor': 'middle', class: 'group-title' }, g, S.n_task > 1 ? 'Task objects' : 'Task object');
      svg('text', { x: split + slot * (n - S.n_task) / 2, y: 16, 'text-anchor': 'middle', class: 'group-title' }, g, 'Distractors');
      S.objects.forEach((obj, i) => {
        const cx = L + slot * (i + 0.5);
        svg('text', { x: cx, y: H - B + 18, 'text-anchor': 'middle', class: 'tick' }, g, obj);
        METHODS.forEach((m, j) => {
          if (hidden.has(m)) return;
          const v = S.mm[m][i], x = cx + (j - 1) * (BW + 8) - BW / 2;
          const mark = bar(g, x, y(v), BW, y(0), COLOR[m]);
          svg('text', { x: x + BW / 2, y: y(v) - 5, 'text-anchor': 'middle', class: 'value' }, g, v.toFixed(2));
          const hit = svg('rect', { x: x - 1, y: T, width: BW + 2, height: H - T - B, fill: 'transparent', tabindex: 0, class: 'hit',
            'aria-label': `${LABEL[m]}, ${obj}: ${v.toFixed(2)} millimeters` }, g);
          const rows = [{ value: v.toFixed(2) + ' mm', text: 'of commanded motion' }, { key: COLOR[m], text: LABEL[m] },
            { text: `${obj} moved 1 cm (${task.toLowerCase()})` }];
          const on = ev => {
            mark.classList.add('is-hover');
            const b = hit.getBoundingClientRect();
            tip.show(ev.clientX || b.left + b.width / 2, ev.clientY || b.top + b.height / 2, rows);
          };
          const off = () => { mark.classList.remove('is-hover'); tip.hide(); };
          hit.addEventListener('pointermove', on); hit.addEventListener('focus', on);
          hit.addEventListener('pointerleave', off); hit.addEventListener('blur', off);
        });
      });
      svg('text', { x: 14, y: T + (H - T - B) / 2, 'text-anchor': 'middle', class: 'axis-title', transform: `rotate(-90 14 ${T + (H - T - B) / 2})` }, root, 'Commanded motion (mm)');

      tiles.textContent = '';
      html('div', 'tiles-title', tiles, 'Selectivity: task objects ÷ distractors');
      METHODS.forEach(m => {
        const t = html('div', 'tile', tiles);
        const name = html('div', 'tile-label', t);
        html('span', 'swatch', name).style.background = COLOR[m];
        html('span', '', name, LABEL[m]);
        html('div', 'tile-value', t, S.selectivity[m].toFixed(1) + '×');
      });
      table(fig, ['Object'].concat(METHODS.map(m => LABEL[m] + ' (mm)')),
        S.objects.map((o, i) => [o].concat(METHODS.map(m => S.mm[m][i].toFixed(3)))));
    }
    draw();
  }

  // method example: twelve states with a 2-D state and a 1-D action
  function idea(fig) {
    const DEG = Math.PI / 180, N = 12, W = 540, H = 380;
    const truth = [-60, -60, -60, -60, -60, -60, 20, 20, 20, 20, 20, 20].map(d => d * DEG);
    const start = [-86, -46, -38, -69, -30, -77, 39, -4, 28, 47, 7, -8].map(d => d * DEG);
    const data = start.slice();
    const pos = [[92, 120], [178, 96], [236, 150], [118, 206], [204, 232], [150, 290],
      [352, 104], [436, 130], [316, 190], [404, 212], [470, 262], [366, 288]];
    const wrap = d => Math.atan2(Math.sin(2 * d), Math.cos(2 * d)) / 2;
    let gamma = 0, k = 6, shrinkTarget = 1, sel = 2, stage = 0, probe = [66, 324];
    let target = data.slice(), shown = data.slice(), speed = new Array(N).fill(0), shrink = 1, shrinkSpeed = 0;

    const regions = (th, kk) => th.map((t, i) => th.map((u, j) => [Math.abs(Math.sin(t - u)), j])
      .sort((p, q) => p[0] - q[0] || p[1] - q[1]).slice(0, kk).map(p => p[1]));
    const average = (th, idx) => {
      let a = 0, c = 0, d = 0;
      idx.forEach(j => { a += Math.cos(th[j]) ** 2; c += Math.cos(th[j]) * Math.sin(th[j]); d += Math.sin(th[j]) ** 2; });
      a /= idx.length; c /= idx.length; d /= idx.length;
      const disc = Math.sqrt((a - d) ** 2 + 4 * c * c);
      return { l1: (a + d + disc) / 2, l2: Math.max(0, (a + d - disc) / 2), phi: 0.5 * Math.atan2(2 * c, a - d) };
    };
    function solve() {
      const th = data.slice();
      if (gamma === 0) return th;
      for (let it = 0; it < 3000; it++) {
        const g = th.map((t, i) => Math.sin(2 * (t - data[i])));
        regions(th, k).forEach(idx => {
          const phi = average(th, idx).phi, v = [-Math.sin(phi), Math.cos(phi)];
          idx.forEach(j => {
            const c = Math.cos(th[j]), s = Math.sin(th[j]);
            g[j] += gamma * 2 * (c * v[0] + s * v[1]) * (-s * v[0] + c * v[1]) / idx.length;
          });
        });
        let moved = 0;
        for (let i = 0; i < N; i++) { th[i] -= 0.06 * g[i]; moved = Math.max(moved, Math.abs(g[i])); }
        if (moved < 1e-5) break;
      }
      return th;
    }

    const wrapEl = html('div', 'explainer', fig);
    const root = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Twelve states, the direction each responds to, and the directions the policy ignores' }, wrapEl);
    const defs = svg('defs', {}, root);
    [['most', INK], ['least', '#8a8a8a']].forEach(([id, c]) => {
      const mk = svg('marker', { id: 'arrow-' + id, viewBox: '0 0 10 10', refX: 7, refY: 5, markerWidth: 5, markerHeight: 5, orient: 'auto-start-reverse' }, defs);
      svg('path', { d: 'M0,0 L10,5 L0,10 Z', fill: c }, mk);
    });
    svg('path', { d: `M24,14 L24,${H - 24} L${W - 14},${H - 24}`, fill: 'none', stroke: INK, 'stroke-width': 1.5 }, root);
    svg('text', { x: 150, y: 32, 'text-anchor': 'middle', class: 'group-title' }, root, 'early in the task');
    svg('text', { x: 404, y: 32, 'text-anchor': 'middle', class: 'group-title' }, root, 'late in the task');
    const seeds = [];
    for (let r = 0, y = 30; y < H - 28; y += 26, r++) for (let x = 40 + (r % 2) * 24; x < W - 16; x += 48) seeds.push([x, y]);
    const trueSeeds = [];
    for (let r = 0, y = 43; y < H - 28; y += 26, r++) for (let x = 64 - (r % 2) * 24; x < W - 16; x += 48) trueSeeds.push([x, y]);
    const trueG = svg('g', {}, root);
    const curveG = svg('g', {}, root);
    const curves = seeds.map(() => svg('path', { fill: 'none', stroke: '#c4c4c4', 'stroke-width': 1.3, 'stroke-linecap': 'round' }, curveG));
    const links = svg('g', {}, root);
    const linkLines = Array.from({ length: N }, () => svg('line', { stroke: '#8f8f8f', 'stroke-width': 1.1, visibility: 'hidden' }, links));
    const states = pos.map((p, i) => {
      const g = svg('g', { class: 'state', tabindex: 0, 'aria-label': `State ${i + 1}` }, root);
      svg('circle', { class: 'halo', cx: p[0], cy: p[1], r: 27 }, g);
      svg('line', { class: 'data-line', stroke: '#bdbdbd', 'stroke-width': 2, 'stroke-linecap': 'round' }, g);
      svg('line', { class: 'tick-line', stroke: INK, 'stroke-width': 2.8, 'stroke-linecap': 'round' }, g);
      svg('circle', { cx: p[0], cy: p[1], r: 3.2, fill: INK }, g);
      return g;
    });
    const most = svg('line', { stroke: INK, 'stroke-width': 3.5, 'marker-end': 'url(#arrow-most)' }, root);
    const least = svg('line', { stroke: '#8a8a8a', 'stroke-width': 3.5, 'marker-end': 'url(#arrow-least)' }, root);
    const mostLabel = svg('text', { class: 'probe-label' }, root, 'action changes');
    const leastLabel = svg('text', { class: 'probe-label' }, root, 'ignored');
    const dot = svg('circle', { r: 8, fill: '#fff', stroke: INK, 'stroke-width': 2, class: 'probe' }, root);
    const key = html('div', 'legend idea-key', fig);
    [['#222', 'direction a state responds to'], ['#bdbdbd', 'the same, in its data'], ['#c4c4c4', 'directions the policy ignores'], ['rgba(255, 153, 0, 0.55)', 'directions it should ignore']].forEach(([c, name]) => {
      const item = html('span', '', key);
      html('span', 'line-key', item).style.background = c;
      html('span', '', item, name);
    });

    // blend the states' directions by distance (doubled angles)
    const blend = (th, x, y) => {
      let c = 0, s = 0;
      for (let i = 0; i < N; i++) {
        const w = Math.exp(-((x - pos[i][0]) ** 2 + (y - pos[i][1]) ** 2) / 9800);
        c += w * Math.cos(2 * th[i]); s += w * Math.sin(2 * th[i]);
      }
      return 0.5 * Math.atan2(s, c);
    };
    const field = (x, y) => blend(shown, x, y);
    const inside = (x, y) => x > 26 && x < W - 16 && y > 16 && y < H - 26;
    function trace(x0, y0, th) {
      const half = sign => {
        const pts = [];
        let x = x0, y = y0, px = 0, py = 0;
        for (let n = 0; n < 8; n++) {
          const a = blend(th, x, y) + Math.PI / 2;
          let dx = Math.cos(a), dy = Math.sin(a);
          if (n === 0) { dx *= sign; dy *= sign; } else if (dx * px + dy * py < 0) { dx = -dx; dy = -dy; }
          x += 6 * dx; y += 6 * dy; px = dx; py = dy;
          if (!inside(x, y)) break;
          pts.push(x.toFixed(1) + ',' + y.toFixed(1));
        }
        return pts;
      };
      return 'M' + half(-1).reverse().concat([x0 + ',' + y0], half(1)).join(' L');
    }

    trueSeeds.forEach(p => svg('path', { d: trace(p[0], p[1], truth), fill: 'none', stroke: COLOR.InneR, 'stroke-width': 2.4, 'stroke-linecap': 'round', opacity: 0.3 }, trueG));

    const side = html('div', 'explainer-side', wrapEl);
    const inset = svg('svg', { viewBox: '0 0 240 128', class: 'inset', role: 'img', 'aria-label': 'Average over the selected region and its two eigenvalues' }, side);
    const ellipse = svg('ellipse', { cx: 84, cy: 64, fill: '#f1f1f1', stroke: INK, 'stroke-width': 1.5 }, inset);
    const insetMost = svg('line', { stroke: INK, 'stroke-width': 3.5, 'marker-end': 'url(#arrow-most)' }, inset);
    const insetLeast = svg('line', { stroke: '#8a8a8a', 'stroke-width': 3.5, 'marker-end': 'url(#arrow-least)' }, inset);
    svg('line', { x1: 176, x2: 232, y1: 108, y2: 108, stroke: BASE, 'stroke-width': 1 }, inset);
    const bar1 = svg('rect', { x: 182, width: 18, fill: '#555' }, inset), bar2 = svg('rect', { x: 208, width: 18, fill: '#8a8a8a' }, inset);
    svg('text', { x: 191, y: 124, 'text-anchor': 'middle', class: 'tick' }, inset, 'λ₁');
    svg('text', { x: 217, y: 124, 'text-anchor': 'middle', class: 'tick' }, inset, 'λ₂');
    const reads = html('div', 'idea-reads', side);
    const read = label => {
      const cell = html('div', '', reads), value = html('strong', '', cell);
      html('span', '', cell, label);
      return value;
    };
    const lossOut = read('null space ratio'), errOut = read('error from true'), sizeOut = read('sensitivity');
    const textStack = html('div', 'stack', side);
    const texts = [...fig.parentNode.querySelectorAll('.step-text')];
    texts.forEach(t => { t.classList.add('layer'); textStack.appendChild(t); });
    const nav = html('div', 'presets', side);
    const mk = (parent, name, click) => { const b = html('button', 'tab', parent, name); b.type = 'button'; b.addEventListener('click', click); return b; };
    const back = mk(nav, 'Back', () => go(stage - 1)), next = mk(nav, 'Next', () => go(stage + 1));
    const count = html('span', 'step-count', nav);

    const controls = html('div', 'idea-controls', fig);
    const slider = (label, min, max, step, value, set, show) => {
      const box = html('label', 'slider', controls), text = html('span', '', box), input = html('input', '', box);
      input.type = 'range'; input.min = min; input.max = max; input.step = step; input.value = value;
      const sync = () => { text.textContent = `${label}: ${show(Number(input.value))}`; };
      input.addEventListener('input', () => { set(Number(input.value)); syncAll(); retarget(); });
      return { input, sync };
    };
    const gammaIn = slider('InneR weight γ', 0, 6, 0.5, gamma, v => { gamma = v; }, v => v === 0 ? '0 (behavior cloning)' : v);
    const kIn = slider('Region size k', 2, N, 1, k, v => { k = v; }, v => v === N ? `${v} (no clustering)` : gamma === 0 ? `${v} (only used to measure the ratio)` : v);
    const shrinkIn = slider('Norm penalty', 0, 80, 5, 0, v => { shrinkTarget = 1 - v / 100; }, v => v === 0 ? 'off' : `shrinks by ${v}%`);
    const syncAll = () => { gammaIn.sync(); kIn.sync(); shrinkIn.sync(); };
    const setAll = (g, kk, sh) => {
      gamma = g; k = kk; shrinkTarget = 1 - sh / 100;
      gammaIn.input.value = g; kIn.input.value = kk; shrinkIn.input.value = sh;
      syncAll(); retarget();
    };
    const presets = html('div', 'presets', controls);
    [['Behavior cloning', 0, 6, 0], ['Norm penalty', 0, 6, 55], ['InneR', 3, 6, 0], ['InneR, no clustering', 3, N, 0]].forEach(([name, g, kk, sh]) => mk(presets, name, () => setAll(g, kk, sh)));
    mk(presets, 'Reset data', () => { start.forEach((v, i) => { data[i] = v; }); retarget(); });
    syncAll();

    function draw() {
      const half = 17 * shrink + 3, mine = stage >= 1 ? regions(target, k)[sel] : [];
      curves.forEach((path, n) => path.setAttribute('d', trace(seeds[n][0], seeds[n][1], shown)));
      const put = (line, p, angle, len) => {
        line.setAttribute('x1', p[0] - Math.cos(angle) * len); line.setAttribute('y1', p[1] - Math.sin(angle) * len);
        line.setAttribute('x2', p[0] + Math.cos(angle) * len); line.setAttribute('y2', p[1] + Math.sin(angle) * len);
      };
      states.forEach((g, i) => {
        const dataLine = g.querySelector('.data-line');
        put(dataLine, pos[i], data[i], 22);
        dataLine.setAttribute('visibility', Math.abs(wrap(shown[i] - data[i])) > 0.03 ? 'visible' : 'hidden');
        put(g.querySelector('.tick-line'), pos[i], shown[i], half);
        g.classList.toggle('is-member', mine.includes(i));
        g.classList.toggle('is-selected', stage >= 1 && i === sel);
        const link = linkLines[i], on = mine.includes(i) && i !== sel;
        link.setAttribute('visibility', on ? 'visible' : 'hidden');
        if (on) { link.setAttribute('x1', pos[sel][0]); link.setAttribute('y1', pos[sel][1]); link.setAttribute('x2', pos[i][0]); link.setAttribute('y2', pos[i][1]); }
      });
      const a = field(probe[0], probe[1]);
      const up = ang => Math.cos(ang) * (W / 2 - probe[0]) + Math.sin(ang) * (H / 2 - probe[1]) < 0 ? ang + Math.PI : ang;
      const arrowAt = (line, label, ang, len) => {
        const x = probe[0] + Math.cos(ang) * len, y = probe[1] + Math.sin(ang) * len;
        line.setAttribute('x1', probe[0]); line.setAttribute('y1', probe[1]); line.setAttribute('x2', x); line.setAttribute('y2', y);
        label.setAttribute('x', x + Math.cos(ang) * 10); label.setAttribute('y', y + Math.sin(ang) * 10 + 4);
        label.setAttribute('text-anchor', Math.cos(ang) < -0.3 ? 'end' : Math.cos(ang) > 0.3 ? 'start' : 'middle');
      };
      arrowAt(most, mostLabel, up(a), 46 * shrink + 14); arrowAt(least, leastLabel, up(a + Math.PI / 2), 56);
      dot.setAttribute('cx', probe[0]); dot.setAttribute('cy', probe[1]);

      inset.style.visibility = stage >= 2 ? 'visible' : 'hidden';
      if (stage >= 2) {
        const { l1, l2, phi } = average(shown, mine), c = Math.cos(phi), s = Math.sin(phi);
        const rx = 50 * Math.sqrt(l1) * shrink, ry = 50 * Math.sqrt(l2) * shrink;
        ellipse.setAttribute('rx', rx + 5); ellipse.setAttribute('ry', ry + 5);
        ellipse.setAttribute('transform', `rotate(${phi / DEG} 84 64)`);
        const arrow = (line, dx, dy, len) => {
          line.setAttribute('x1', 84); line.setAttribute('y1', 64); line.setAttribute('x2', 84 + dx * len); line.setAttribute('y2', 64 + dy * len);
          line.setAttribute('visibility', len > 12 ? 'visible' : 'hidden');
        };
        arrow(insetMost, c, s, rx + 16); arrow(insetLeast, -s, c, ry < 2 ? 0 : ry + 16);
        bar1.setAttribute('height', 90 * l1); bar1.setAttribute('y', 108 - 90 * l1);
        bar2.setAttribute('height', 90 * l2); bar2.setAttribute('y', 108 - 90 * l2);
      }
      const loss = regions(shown, k).reduce((sum, idx) => sum + average(shown, idx).l2, 0) / N;
      lossOut.textContent = (100 * loss).toFixed(1) + '%';
      errOut.textContent = Math.round(shown.reduce((sum, t, i) => sum + Math.abs(wrap(t - truth[i])), 0) / N / DEG) + '°';
      sizeOut.textContent = Math.round(100 * shrink * shrink) + '%';
    }

    let last = 0, animating = false;
    function frame(now) {
      const dt = Math.min(0.033, Math.max(0, (now - last) / 1000) || 0.016), w = 7.5;
      last = now;
      let busy = false;
      for (let i = 0; i < N; i++) {
        const off = wrap(shown[i] - target[i]);
        speed[i] += (-w * w * off - 2 * w * speed[i]) * dt;
        shown[i] += speed[i] * dt;
        if (Math.abs(off) > 2e-4 || Math.abs(speed[i]) > 2e-4) busy = true;
      }
      const off = shrink - shrinkTarget;
      shrinkSpeed += (-w * w * off - 2 * w * shrinkSpeed) * dt;
      shrink += shrinkSpeed * dt;
      if (Math.abs(off) > 2e-4 || Math.abs(shrinkSpeed) > 2e-4) busy = true;
      draw();
      if (busy) requestAnimationFrame(frame); else animating = false;
    }
    function retarget() {
      target = solve();
      if (!animating) { animating = true; last = performance.now(); requestAnimationFrame(frame); }
    }

    function go(to) {
      stage = Math.max(0, Math.min(4, to));
      texts.forEach((t, i) => t.classList.toggle('is-on', i === stage));
      back.disabled = stage === 0; next.disabled = stage === 4;
      count.textContent = `${stage + 1} / 5`;
      controls.classList.toggle('is-dim', stage < 4);
      if (stage < 3) setAll(0, 6, 0); else if (stage === 3) setAll(3, 6, 0); else draw();
    }

    const point = ev => {
      const b = root.getBoundingClientRect();
      return [(ev.clientX - b.left) * W / b.width, (ev.clientY - b.top) * H / b.height];
    };
    let drag = null;
    root.addEventListener('pointerdown', ev => {
      ev.preventDefault();
      const [x, y] = point(ev);
      const hit = pos.findIndex(p => Math.hypot(x - p[0], y - p[1]) < 27);
      root.setPointerCapture(ev.pointerId);
      if (hit >= 0) { sel = hit; drag = { state: hit }; draw(); } else if (inside(x, y)) { drag = { probe: true }; probe = [x, y]; draw(); }
    });
    root.addEventListener('pointermove', ev => {
      if (!drag) return;
      const [x, y] = point(ev);
      if (drag.probe) { if (inside(x, y)) { probe = [x, y]; if (!animating) draw(); } return; }
      const p = pos[drag.state];
      if (Math.hypot(x - p[0], y - p[1]) > 8) { data[drag.state] = Math.atan2(y - p[1], x - p[0]); retarget(); }
    });
    root.addEventListener('pointerup', () => { drag = null; });
    states.forEach((g, i) => {
      g.addEventListener('focus', () => { sel = i; draw(); });
      g.addEventListener('keydown', ev => {
        const turn = { ArrowLeft: -0.12, ArrowRight: 0.12 }[ev.key];
        if (turn) { ev.preventDefault(); data[i] += turn; retarget(); }
      });
    });
    go(0);
  }

  // training: behavior cloning and InneR from the same random start
  function training(fig) {
    const DEG = Math.PI / 180, N = 12, W = 540, H = 380, T = 120, LR = 0.025, K = 6;
    const truth = [-60, -60, -60, -60, -60, -60, 20, 20, 20, 20, 20, 20].map(d => d * DEG);
    const data = [-86, -46, -38, -69, -30, -77, 39, -4, 28, 47, 7, -8].map(d => d * DEG);
    const pos = [[92, 120], [178, 96], [236, 150], [118, 206], [204, 232], [150, 290],
      [352, 104], [436, 130], [316, 190], [404, 212], [470, 262], [366, 288]];
    const wrap = d => Math.atan2(Math.sin(2 * d), Math.cos(2 * d)) / 2;
    let init = [-90, 76, -38, -52, 41, 6, -31, 45, 46, 8, 33, 24].map(d => d * DEG);
    let gamma = 1, at = 0, playing = false, metric = 'error';
    const METRICS = {
      error: { name: 'Error from the true directions', max: 50, ticks: [0, 10, 20, 30, 40, 50], show: v => Math.round(v) + '°' },
      fit: { name: 'Imitation loss', max: 0.5, ticks: [0, 0.1, 0.2, 0.3, 0.4, 0.5], show: v => v.toFixed(3) },
      ratio: { name: 'Null space ratio', max: 50, ticks: [0, 10, 20, 30, 40, 50], show: v => v.toFixed(1) + '%' },
    };
    const runs = [{ key: 'BC', gamma: () => 0 }, { key: 'InneR', gamma: () => gamma }];

    const regions = th => th.map((t, i) => th.map((u, j) => [Math.abs(Math.sin(t - u)), j])
      .sort((p, q) => p[0] - q[0] || p[1] - q[1]).slice(0, K).map(p => p[1]));
    const average = (th, idx) => {
      let a = 0, c = 0, d = 0;
      idx.forEach(j => { a += Math.cos(th[j]) ** 2; c += Math.cos(th[j]) * Math.sin(th[j]); d += Math.sin(th[j]) ** 2; });
      a /= idx.length; c /= idx.length; d /= idx.length;
      const disc = Math.sqrt((a - d) ** 2 + 4 * c * c);
      return { l2: Math.max(0, (a + d - disc) / 2), phi: 0.5 * Math.atan2(2 * c, a - d) };
    };
    // gradient descent on mean sin^2(theta - data) + gamma * mean ratio
    function simulate() {
      runs.forEach(r => {
        let th = init.slice();
        r.path = []; r.error = []; r.fit = []; r.ratio = [];
        for (let t = 0; t <= T; t++) {
          const reg = regions(th);
          r.path.push(th.slice());
          r.error.push(th.reduce((sum, v, i) => sum + Math.abs(wrap(v - truth[i])), 0) / N / DEG);
          r.fit.push(th.reduce((sum, v, i) => sum + Math.sin(v - data[i]) ** 2, 0) / N);
          r.ratio.push(100 * reg.reduce((sum, idx) => sum + average(th, idx).l2, 0) / N);
          const g = th.map((v, i) => Math.sin(2 * (v - data[i])));
          if (r.gamma() > 0) reg.forEach(idx => {
            const phi = average(th, idx).phi, v = [-Math.sin(phi), Math.cos(phi)];
            idx.forEach(j => {
              const c = Math.cos(th[j]), sn = Math.sin(th[j]);
              g[j] += r.gamma() * 2 * (c * v[0] + sn * v[1]) * (-sn * v[0] + c * v[1]) / idx.length;
            });
          });
          th = th.map((v, i) => v - LR * g[i]);
        }
      });
    }

    const blend = (th, x, y) => {
      let c = 0, sn = 0;
      for (let i = 0; i < N; i++) {
        const w = Math.exp(-((x - pos[i][0]) ** 2 + (y - pos[i][1]) ** 2) / 9800);
        c += w * Math.cos(2 * th[i]); sn += w * Math.sin(2 * th[i]);
      }
      return 0.5 * Math.atan2(sn, c);
    };
    const inside = (x, y) => x > 26 && x < W - 16 && y > 16 && y < H - 26;
    function trace(x0, y0, th) {
      const half = sign => {
        const pts = [];
        let x = x0, y = y0, px = 0, py = 0;
        for (let n = 0; n < 8; n++) {
          const a = blend(th, x, y) + Math.PI / 2;
          let dx = Math.cos(a), dy = Math.sin(a);
          if (n === 0) { dx *= sign; dy *= sign; } else if (dx * px + dy * py < 0) { dx = -dx; dy = -dy; }
          x += 6 * dx; y += 6 * dy; px = dx; py = dy;
          if (!inside(x, y)) break;
          pts.push(x.toFixed(1) + ',' + y.toFixed(1));
        }
        return pts;
      };
      return 'M' + half(-1).reverse().concat([x0 + ',' + y0], half(1)).join(' L');
    }
    const seeds = [], trueSeeds = [];
    for (let r = 0, y = 30; y < H - 28; y += 26, r++) for (let x = 40 + (r % 2) * 24; x < W - 16; x += 48) seeds.push([x, y]);
    for (let r = 0, y = 43; y < H - 28; y += 26, r++) for (let x = 64 - (r % 2) * 24; x < W - 16; x += 48) trueSeeds.push([x, y]);

    const panels = html('div', 'train-panels', fig);
    runs.forEach(r => {
      const box = html('div', '', panels), title = html('div', 'train-title', box);
      html('span', 'swatch', title).style.background = COLOR[r.key];
      html('span', '', title, LABEL[r.key]);
      const root = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': `${LABEL[r.key]}: the directions each state responds to during training` }, box);
      svg('path', { d: `M24,14 L24,${H - 24} L${W - 14},${H - 24}`, fill: 'none', stroke: INK, 'stroke-width': 1.5 }, root);
      trueSeeds.forEach(p => svg('path', { d: trace(p[0], p[1], truth), fill: 'none', stroke: COLOR.InneR, 'stroke-width': 2.4, 'stroke-linecap': 'round', opacity: 0.3 }, root));
      r.strokes = seeds.map(() => svg('path', { fill: 'none', stroke: '#c4c4c4', 'stroke-width': 1.3, 'stroke-linecap': 'round' }, root));
      r.ticks = pos.map(p => {
        const line = svg('line', { stroke: INK, 'stroke-width': 2.8, 'stroke-linecap': 'round' }, root);
        svg('circle', { cx: p[0], cy: p[1], r: 3.2, fill: INK }, root);
        return line;
      });
    });

    const controls = html('div', 'train-controls', fig);
    const play = html('button', 'tab', controls, 'Play');
    play.type = 'button';
    const scrubBox = html('label', 'slider', controls), scrubText = html('span', '', scrubBox), scrub = html('input', '', scrubBox);
    scrub.type = 'range'; scrub.min = 0; scrub.max = T; scrub.step = 1; scrub.value = 0;
    const gammaBox = html('label', 'slider', controls), gammaText = html('span', '', gammaBox), gammaIn = html('input', '', gammaBox);
    gammaIn.type = 'range'; gammaIn.min = 0; gammaIn.max = 3; gammaIn.step = 0.25; gammaIn.value = gamma;
    const shuffle = html('button', 'tab', controls, 'New initialization');
    shuffle.type = 'button';

    const tabRow = html('div', 'tab-row', fig);
    Object.keys(METRICS).forEach((key, i) => {
      const b = html('button', 'tab' + (i === 0 ? ' is-on' : ''), tabRow, METRICS[key].name);
      b.type = 'button';
      b.addEventListener('click', () => {
        tabRow.querySelectorAll('.tab').forEach(t => t.classList.remove('is-on'));
        b.classList.add('is-on');
        metric = key; chart(); draw();
      });
    });
    const legendRow = html('div', 'legend', fig);
    runs.forEach(r => {
      const item = html('span', 'legend-item', legendRow);
      html('span', 'swatch', item).style.background = COLOR[r.key];
      html('span', '', item, LABEL[r.key]);
      r.value = html('strong', 'train-readout', item);
    });
    const CW = 960, CH = 230, L = 52, R = 16, TOP = 12, B = 42;
    const plot = svg('svg', { viewBox: `0 0 ${CW} ${CH}`, role: 'img', 'aria-label': 'Training curves for behavior cloning and InneR' }, fig);
    const axes = svg('g', {}, plot), curvesG = svg('g', {}, plot);
    const cursor = svg('line', { y1: TOP, y2: CH - B, stroke: BASE, 'stroke-width': 1 }, plot);
    runs.forEach(r => {
      r.line = svg('path', { fill: 'none', stroke: COLOR[r.key], 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }, curvesG);
      r.dot = svg('circle', { r: 4.5, fill: COLOR[r.key], stroke: '#fff', 'stroke-width': 2 }, plot);
    });
    const x = t => L + (CW - L - R) * t / T;
    const y = v => TOP + (CH - TOP - B) * (1 - Math.min(v, METRICS[metric].max) / METRICS[metric].max);

    function chart() {
      const m = METRICS[metric];
      axes.textContent = '';
      m.ticks.forEach(v => {
        svg('line', { x1: L, x2: CW - R, y1: y(v), y2: y(v), stroke: v === 0 ? BASE : GRID, 'stroke-width': 1 }, axes);
        svg('text', { x: L - 8, y: y(v) + 4, 'text-anchor': 'end', class: 'tick' }, axes, v);
      });
      [0, 30, 60, 90, 120].forEach(t => svg('text', { x: x(t), y: CH - B + 18, 'text-anchor': 'middle', class: 'tick' }, axes, t));
      svg('text', { x: L + (CW - L - R) / 2, y: CH - 6, 'text-anchor': 'middle', class: 'axis-title' }, axes, 'Gradient step');
      runs.forEach(r => r.line.setAttribute('d', 'M' + r[metric].map((v, t) => x(t).toFixed(1) + ',' + y(v).toFixed(1)).join(' L')));
    }
    function draw() {
      const t0 = Math.max(0, Math.min(T - 1, Math.floor(at))), f = at - t0, m = METRICS[metric];
      runs.forEach(r => {
        const th = r.path[t0].map((v, i) => v + f * wrap(r.path[t0 + 1][i] - v));
        r.strokes.forEach((path, n) => path.setAttribute('d', trace(seeds[n][0], seeds[n][1], th)));
        r.ticks.forEach((line, i) => {
          line.setAttribute('x1', pos[i][0] - Math.cos(th[i]) * 20); line.setAttribute('y1', pos[i][1] - Math.sin(th[i]) * 20);
          line.setAttribute('x2', pos[i][0] + Math.cos(th[i]) * 20); line.setAttribute('y2', pos[i][1] + Math.sin(th[i]) * 20);
        });
        const v = r[metric][t0] + f * (r[metric][t0 + 1] - r[metric][t0]);
        r.dot.setAttribute('cx', x(at)); r.dot.setAttribute('cy', y(v));
        r.value.textContent = m.show(v);
      });
      cursor.setAttribute('x1', x(at)); cursor.setAttribute('x2', x(at));
      scrub.value = Math.round(at);
      scrubText.textContent = `Gradient step: ${Math.round(at)}`;
      gammaText.textContent = `InneR weight γ: ${gamma}`;
    }

    let last = null;
    function frame(now) {
      if (!playing) return;
      const dt = last === null ? 0 : Math.max(0, Math.min(0.05, (now - last) / 1000));
      last = now;
      at = Math.max(0, Math.min(T, at + 30 * dt));
      draw();
      if (at >= T) { playing = false; play.textContent = 'Replay'; } else requestAnimationFrame(frame);
    }
    function start() {
      if (at >= T) at = 0;
      playing = true; play.textContent = 'Pause'; last = null;
      requestAnimationFrame(frame);
    }
    function stop() { playing = false; play.textContent = at >= T ? 'Replay' : 'Play'; }
    play.addEventListener('click', () => (playing ? stop() : start()));
    scrub.addEventListener('input', () => { stop(); at = Number(scrub.value); play.textContent = 'Play'; draw(); });
    gammaIn.addEventListener('input', () => { gamma = Number(gammaIn.value); simulate(); chart(); draw(); });
    shuffle.addEventListener('click', () => {
      init = Array.from({ length: N }, () => (Math.random() - 0.5) * Math.PI);
      simulate(); chart(); at = 0; draw(); start();
    });
    simulate(); chart(); draw();
  }

  // rollout clips, one pair of videos per task
  function clips(fig) {
    const tasks = ['Close', 'Stack', 'Push'], pairs = {};
    const row = html('div', 'tab-row', fig), stage = html('div', 'stack', fig);
    tasks.forEach((task, i) => {
      const pair = html('div', 'clips layer' + (i === 0 ? ' is-on' : ''), stage);
      [['inner', 'InneR', COLOR.InneR], ['bc', 'Behavior cloning', COLOR.BC]].forEach(([key, name, color]) => {
        const f = html('figure', '', pair), v = html('video', '', f);
        v.muted = true; v.loop = true; v.playsInline = true; v.preload = 'auto';
        v.src = `./static/resources/${task.toLowerCase()}_${key}.mp4`;
        if (i === 0) v.autoplay = true;
        const cap = html('figcaption', '', f);
        html('span', 'swatch', cap).style.background = color;
        html('span', '', cap, name);
      });
      pairs[task] = pair;
      const b = html('button', 'tab' + (i === 0 ? ' is-on' : ''), row, task);
      b.type = 'button';
      b.addEventListener('click', () => {
        row.querySelectorAll('.tab').forEach(t => t.classList.remove('is-on'));
        b.classList.add('is-on');
        tasks.forEach(t => {
          const on = t === task;
          pairs[t].classList.toggle('is-on', on);
          pairs[t].querySelectorAll('video').forEach(v => { if (on) { v.currentTime = 0; v.play().catch(() => {}); } else v.pause(); });
        });
      });
    });
  }

  // paper figures chosen with tabs
  function gallery(fig) {
    const items = JSON.parse(fig.dataset.items);
    const row = html('div', 'tab-row', fig), stage = html('div', 'stack', fig), cap = html('p', 'caption', fig, items[0].caption);
    const layers = items.map((it, i) => {
      const img = html('img', 'layer' + (i === 0 ? ' is-on' : ''), stage);
      img.src = './static/resources/' + it.src; img.alt = it.caption; img.style.width = it.width + '%';
      const b = html('button', 'tab' + (i === 0 ? ' is-on' : ''), row, it.label);
      b.type = 'button';
      b.addEventListener('click', () => {
        row.querySelectorAll('.tab').forEach(t => t.classList.remove('is-on'));
        b.classList.add('is-on');
        layers.forEach(l => l.classList.toggle('is-on', l === img));
        cap.textContent = it.caption;
      });
      return img;
    });
  }

  document.addEventListener('DOMContentLoaded', () => {
    const run = (id, f) => { const e = document.getElementById(id); if (e) f(e); };
    run('fig-idea', idea);
    run('fig-training', training);
    run('fig-clips', clips);
    run('fig-success', successChart);
    run('fig-noise', noiseChart);
    run('fig-sensitivity', sensitivityChart);
    document.querySelectorAll('.gallery').forEach(gallery);
  });
})();
