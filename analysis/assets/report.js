/* The page: one state, three panels over it.
 *
 * The rail on the left holds the state — model, task, the two time limits, the
 * backends in use, the fleet's composition — and every change re-renders the panel
 * in view. A machine is priced on its fastest device among the backends in use
 * (`taskMath`); the fleet's coverage comes from those prices (`fleetMath`).
 *
 * Every string that came from a submission reaches the DOM through `esc`.
 */
(function () {
  "use strict";

  const D = JSON.parse(document.getElementById("pack").textContent);
  const price = window.taskMath.price;

  /* ---------- the data, indexed ---------- */
  const FAMILIES = D.families, FAM = Object.fromEntries(FAMILIES.map(f => [f.key, f]));
  const KINDS = D.classes, KIND = Object.fromEntries(KINDS.map(k => [k.key, k]));
  const MACHINE = Object.fromEntries(D.machines.map(m => [m.id, m]));
  const REC = {};
  D.records.forEach(r => { REC[r.lane + "|" + r.model] = r; });
  D.machines.forEach(mc => { mc.lanes = D.lanes.filter(l => l.machine === mc.id); });

  /* ---------- what the controls offer ---------- */
  const STOPS = {
    prompt: [{ v: 40, l: "question" }, { v: 270, l: "email" }, { v: 670, l: "page" }, { v: 2000, l: "article" }, { v: 8000, l: "report" }],
    out: [{ v: 20, l: "line" }, { v: 130, l: "paragraph" }, { v: 670, l: "page" }, { v: 2000, l: "essay" }],
  };
  const SLIDERS = { prompt: 16384, out: 4096 };
  const LIMIT_RANGE = [0.5, 600];   /* seconds: where the two time limits can be set */
  const SWEEPS = {
    prompt: { label: "Longer prompt", axis: "prompt length (tokens)", lo: 64, hi: 8192 },
    out: { label: "Longer reply", axis: "reply length (tokens)", lo: 16, hi: 2048 },
  };
  const TIERS = [{ key: "low", label: "Low" }, { key: "mid", label: "Mid" }, { key: "high", label: "High" }];
  const MAX_SEL = 8;

  const state = {
    model: D.default_model, prompt: 4096, out: 500, optimal: 25, target: 45,
    /* the page opens on what every machine has: the CPU backend alone */
    on: Object.fromEntries(FAMILIES.map(f => [f.key, f.key === "cpu"])),
    fleet: { kind: Object.fromEntries(KINDS.map(k => [k.key, k.share])), tier: { low: 16, mid: 68, high: 16 } },
    sel: [], view: "coverage", sweep: "prompt",
    q: "", bestOnly: true, allModels: false, onlyOk: false, sort: { key: "total", dir: 1 },
  };

  /* ---------- pricing ---------- */
  const task = () => ({ prompt: state.prompt, out: state.out });
  const byTotal = (a, b) => (a.v ? a.v.total : 1e12) - (b.v ? b.v.total : 1e12);
  /* Every device of one machine, fastest first. `any` ignores the backend switches. */
  function devices(mc, model, p, any) {
    return mc.lanes.filter(l => any || state.on[l.family]).map(l => {
      const rec = REC[l.id + "|" + model];
      return { mc: mc, lane: l, rec: rec, v: price(rec, p) };
    }).filter(d => d.rec).sort(byTotal);
  }
  /* The machine as one row: its fastest device on the backends that are switched on. */
  function machineRow(mc, model, p) {
    const opts = devices(mc, model, p), top = opts[0] || null;
    return { mc: mc, opts: opts, lane: top && top.lane, rec: top && top.rec, v: top && top.v };
  }
  const priced = (model, p) => D.machines.map(mc => machineRow(mc, model, p)).filter(d => d.lane);
  const meets = v => !!v && v.total <= state.target;
  /* what the allocator reserves for a context that holds the task */
  const reserve = (rec, p) => rec.mem ? rec.mem.weights + rec.mem.compute + rec.mem.kv_base + rec.mem.kv_per_tok * Math.max(512, p.prompt + p.out) : null;

  /* The fleet for the task in view. Per class: the times of the CPUs it draws on
     — held to the cores a typical machine has, and with those as they ran and held
     to a weak machine's (`cores`; as they ran where the class names none) — and per
     machine its fastest GPU in use. Classes that draw on the same CPUs (`cpus`)
     share every one of them, and each leans toward its own machines' — a property
     of the machines, so the CPU times behind it are read whatever is switched on. */
  let FLEET = [];
  function fleetNow(p) {
    const rows = D.machines.map(mc => {
      const opts = devices(mc, state.model, p).filter(d => d.v);
      const gpu = opts.find(d => d.lane.family !== "cpu");
      return { mc: mc, cpu: devices(mc, state.model, p, true).find(d => d.lane.family === "cpu" && d.v), gpu: gpu ? gpu.v.total : null };
    });
    const classes = KINDS.map(k => {
      const cpus = rows.filter(r => r.cpu && KIND[r.mc.kind].cpus === k.cpus), norm = k.cores ? k.cores.typical : null;
      const at = (rs, n) => rs.map(r => price(r.cpu.rec, p, n).total);
      const typical = at(cpus, norm), cpu = k.cores ? typical.concat(at(cpus, null), at(cpus, k.cores.weak)) : typical;
      return { key: k.key, share: state.fleet.kind[k.key], typical: state.on.cpu ? typical : [], cpu: state.on.cpu ? cpu : [],
               lean: window.fleetMath.lean(typical, cpu, at(cpus.filter(r => r.mc.kind === k.key), norm)),
               gpu: rows.filter(r => r.mc.kind === k.key).map(r => r.gpu) };
    });
    return window.fleetMath.fleet(classes, TIERS.map(t => state.fleet.tier[t.key]), state.optimal, state.target);
  }

  /* ---------- formatting ---------- */
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const int = n => n.toLocaleString("en");
  const pct = v => Math.round(v * 100) + "%";
  const norm = a => { const s = a.reduce((x, y) => x + y, 0); return s > 0 ? a.map(v => v / s) : a.map(() => 1 / a.length); };
  const listOf = a => a.length < 2 ? a.join("") : a.slice(0, -1).join(", ") + " and " + a[a.length - 1];
  function fmtS(s) {
    if (s === null || s === undefined || !isFinite(s)) return "–";
    if (s < 0.095) return Math.max(1, Math.round(s * 1000)) + " ms";
    if (s < 0.95) return s.toFixed(2) + " s";
    if (s < 10) return s.toFixed(1) + " s";
    if (s < 60) return Math.round(s) + " s";
    if (s < 3600) { const m = Math.floor(s / 60), r = Math.round(s - 60 * m); return r ? m + " min " + r + " s" : m + " min"; }
    const hh = Math.floor(s / 3600);
    return hh + " h " + Math.round((s - 3600 * hh) / 60) + " min";
  }
  function fmtTick(s) {
    if (s < 60) return (s < 1 ? String(s) : String(Math.round(s))) + " s";
    if (s < 3600) return (s / 60) + " min";
    return (s / 3600) + " h";
  }
  function fmtN(v) {
    if (v === null || v === undefined || !isFinite(v)) return "–";
    if (v >= 10000) return Math.round(v / 1000) + "k";
    if (v >= 1000) return (v / 1000).toFixed(1).replace(/\.0$/, "") + "k";
    if (v >= 10) return String(Math.round(v));
    if (v >= 1) return v.toFixed(1).replace(/\.0$/, "");
    return String(+v.toPrecision(2));
  }
  const fmtGB = mb => mb === null ? "–" : (mb / 1024).toFixed(1) + " GB";
  /* a device named from inside its machine, and how it ran */
  const inMachine = l => l.cls === "CPU" ? "CPU" : l.cls === "discrete GPU" ? l.short : l.family === "mtl" ? "GPU" : "integrated GPU";
  const how = l => l.cls === "CPU" ? (l.threads ? l.threads + " threads" : "CPU") : FAM[l.family].label;
  const runsOn = l => l.short + " · " + how(l);

  /* ---------- scales and marks ---------- */
  function logScale(lo, hi, a, b) {
    const l0 = Math.log(lo), l1 = Math.log(hi);
    return v => a + (Math.log(v) - l0) / (l1 - l0) * (b - a);
  }
  function padDomain(vals, pad) {
    const lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
    return lo === hi ? [lo / 2, hi * 2] : [lo / pad, hi * pad];
  }
  const TIME_TICKS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 1200, 1800, 3600, 7200, 18000, 36000];
  const TIME_MAJOR = [0.01, 0.1, 1, 10, 60, 600, 3600, 36000];
  /* time ticks that fit: the round ones first, then whatever still has `gap` pixels of room */
  function timeTicks(lo, hi, sc, gap) {
    const keep = [];
    const tryAdd = t => { if (t >= lo && t <= hi && keep.every(k => Math.abs(sc(k) - sc(t)) >= gap)) keep.push(t); };
    TIME_MAJOR.forEach(tryAdd); TIME_TICKS.forEach(tryAdd);
    return keep.sort((a, b) => a - b);
  }
  /* The time axis does not move while the task, the limits or the backends change:
     one domain covers every record across the slider range, so the fleet slides
     under the limits. */
  const TIME_DOMAIN = (() => {
    const t = [];
    D.records.forEach(rec => [64, 512, 4096, 16384].forEach(P => [16, 256, 2048].forEach(O => {
      const v = price(rec, { prompt: P, out: O });
      if (v) t.push(v.total);
    })));
    return t.length ? padDomain(t, 1.8) : [0.1, 600];
  })();
  const dot = (x, y, r, cls, attrs) => `<circle cx="${x}" cy="${y}" r="${r}" class="${cls}" ${attrs || ""}/>`;
  const swatch = mc => `<i class="swatch bg-${mc.kind}"></i>`;
  const KEY_SEL = `<span><svg width="14" height="14" viewBox="0 0 14 14"><circle cx="7" cy="7" r="6" class="halo"/><circle cx="7" cy="7" r="3" style="fill:var(--ink2)"/></svg>in the comparison</span>`;

  /* ---------- the answer ---------- */
  function renderAnswer(all) {
    const none = !FAMILIES.some(f => state.on[f.key]), past = !none && all.length > 0 && !all.some(d => d.v);
    const cov = FLEET.reduce((s, c) => s + c.share * c.cov, 0), opt = FLEET.reduce((s, c) => s + c.share * c.opt, 0);
    document.getElementById("cov-num").textContent = none || past ? "–" : pct(cov);
    document.getElementById("cov-sub").textContent = none ? "no backend switched on" : past ? "past the model's context"
      : `${pct(opt)} optimal · ${Math.round(cov * 100) - Math.round(opt * 100)}% tolerable`;
    document.getElementById("cover").setAttribute("aria-label", none || past ? "No fleet coverage to report"
      : `${pct(cov)} of the fleet finishes within ${fmtS(state.target)}`);
    /* What to budget for the model and the task, whatever it runs on: the largest
       reservation any device made for it. Backends account for the same weights a
       little differently, so the switches do not move it. */
    const p = task();
    let most = null;
    D.records.forEach(rec => { if (rec.model === state.model && rec.mem && (!most || reserve(rec, p) > reserve(most, p))) most = rec; });
    document.getElementById("mem-num").textContent = most && !past ? fmtGB(reserve(most, p)) : "–";
    document.getElementById("mem-sub").textContent = !most ? "not measured" : past ? "past the model's context" : "";
    document.getElementById("memory").setAttribute("aria-label", most && !past
      ? `About ${fmtGB(reserve(most, p))} reserved for the model and this task` : "No memory figure to report");
  }

  /* ---------- coverage: how the fleet is spread over whole-task time ---------- */
  function renderMap(all) {
    /* the key first: its height is part of what the chart has to fit beside */
    document.getElementById("map-key").innerHTML = `<span><svg width="22" height="10"><line x1="0" x2="22" y1="5" y2="5" class="optimal"/></svg>optimal up to</span><span><svg width="22" height="10"><line x1="0" x2="22" y1="5" y2="5" class="target"/></svg>tolerable up to</span><span><svg width="14" height="12"><rect width="14" height="12" rx="2" class="miss"/></svg>unusable</span>`
      + `<span><svg width="26" height="12" viewBox="0 0 26 12"><path d="M0 12C6 12 8 1 13 1S20 12 26 12Z" style="fill:var(--ink2);opacity:.4"/></svg>a class: area is its share of the fleet, darker the sooner it finishes</span>`
      + `<span><svg width="14" height="14" viewBox="0 0 14 14"><circle cx="7" cy="7" r="5" class="dot pin" style="stroke:var(--ink2)"/></svg>a measured machine</span>` + (state.sel.length ? KEY_SEL : "");
    const el = document.getElementById("map"), W = Math.max(300, el.clientWidth), small = W < 640;
    const m = { l: 12, r: 12, t: 28, b: 42 };
    /* beside the rail, the chart grows so both columns end at the same height */
    let LH = small ? 92 : 128;
    const rail = document.querySelector(".rail"), stage = el.closest(".stage");
    if (rail && stage && getComputedStyle(rail).position === "sticky") {
      const rest = stage.getBoundingClientRect().height - el.getBoundingClientRect().height;
      LH = Math.max(100, Math.min(240, Math.floor((rail.getBoundingClientRect().height - rest - m.t - m.b) / KINDS.length)));
    }
    const H = m.t + LH * KINDS.length + m.b, clampX = x => Math.max(m.l, Math.min(W - m.r, x));
    const sx = logScale(TIME_DOMAIN[0], TIME_DOMAIN[1], m.l, W - m.r), tx = clampX(sx(state.target)), ox = clampX(sx(state.optimal));
    let svg = `<rect x="${ox.toFixed(1)}" y="${m.t}" width="${(tx - ox).toFixed(1)}" height="${H - m.t - m.b}" class="zone-tol"/>`
      + `<rect x="${tx.toFixed(1)}" y="${m.t}" width="${(W - m.r - tx).toFixed(1)}" height="${H - m.t - m.b}" class="miss"/>`;
    timeTicks(TIME_DOMAIN[0], TIME_DOMAIN[1], sx, small ? 46 : 64).forEach(t => {
      svg += `<line x1="${sx(t)}" x2="${sx(t)}" y1="${m.t}" y2="${H - m.b}" class="grid"/><text x="${sx(t)}" y="${H - m.b + 15}" text-anchor="middle">${fmtTick(t)}</text>`;
    });
    svg += `<text x="${(m.l + W - m.r) / 2}" y="${H - 8}" text-anchor="middle" class="t-ink">Whole task, log scale</text>`;
    /* one height scale for every class, so a class's area is its share of the fleet */
    const peak = Math.max.apply(null, [1e-9].concat(FLEET.map(c => c.curve ? Math.max.apply(null, c.curve.map(q => q.d)) : 0))), scale = (LH - 26) / peak;
    const base = {};
    FLEET.forEach((c, i) => {
      const y0 = m.t + LH * (i + 1) - 8, k = c.key, top = q => (y0 - Math.min(LH - 14, q.d * scale)).toFixed(1);
      base[k] = y0;
      svg += `<line x1="${m.l}" x2="${W - m.r}" y1="${y0}" y2="${y0}" class="grid"/><text x="${m.l + 4}" y="${y0 - LH + 22}" class="t-lane">${esc(KIND[k].label)}</text><text x="${m.l + 4}" y="${y0 - LH + 43}" class="t-pct">${c.curve ? pct(c.cov) : "–"}</text>`
        + (c.curve ? `<text x="${m.l + 4}" y="${y0 - LH + 59}" class="t-lane">${pct(c.opt)} optimal</text>` : "");
      if (c.curve && c.share > 0.0005) {
        const d = `M${sx(c.curve[0].t).toFixed(1)} ${y0}` + c.curve.map(q => `L${sx(q.t).toFixed(1)} ${top(q)}`).join("") + `L${sx(c.curve[c.curve.length - 1].t).toFixed(1)} ${y0}Z`;
        svg += `<path d="${d}" class="hump k-${k}" data-kind="${k}"/><path d="${d}" class="hump made k-${k}" clip-path="url(#made-clip)"/><path d="${d}" class="hump best k-${k}" clip-path="url(#best-clip)"/>`;
        c.cuts.forEach(q => { svg += `<line x1="${sx(q.t).toFixed(1)}" x2="${sx(q.t).toFixed(1)}" y1="${y0}" y2="${top(q)}" class="cut"/>`; });
      }
    });
    /* the two limits: optimal labelled on its left, tolerable on its right, so they never collide */
    svg += `<line x1="${ox.toFixed(1)}" x2="${ox.toFixed(1)}" y1="${m.t - 6}" y2="${H - m.b}" class="optimal"/><line x1="${tx.toFixed(1)}" x2="${tx.toFixed(1)}" y1="${m.t - 6}" y2="${H - m.b}" class="target"/>`;
    const oLeft = ox > m.l + 96, tRight = tx < W - m.r - 110;
    svg += `<text x="${(oLeft ? ox - 6 : ox + 6).toFixed(1)}" y="${oLeft ? m.t - 10 : m.t + 14}" text-anchor="${oLeft ? "end" : "start"}" class="t-opt">${esc(fmtS(state.optimal))} optimal</text>`
      + `<text x="${(tRight ? tx + 6 : tx - 6).toFixed(1)}" y="${tRight ? m.t - 10 : m.t + 14}" text-anchor="${tRight ? "start" : "end"}" class="t-target">${esc(fmtS(state.target))} tolerable</text>`;
    /* the measured machines, on their class's baseline */
    const pts = all.filter(d => d.v).map(d => ({ d: d, x: sx(d.v.total), y: base[d.mc.kind], sel: state.sel.indexOf(d.mc.id) >= 0 }));
    pts.forEach(q => {
      svg += dot(q.x.toFixed(1), q.y, 5, `dot pin k-${q.d.mc.kind}`,
        `data-machine="${esc(q.d.mc.id)}" tabindex="0" role="button" aria-label="${esc(q.d.mc.name)}, ${esc(fmtS(q.d.v.total))} on ${esc(runsOn(q.d.lane))}"`);
    });
    const rows = {};
    pts.filter(q => q.sel).sort((a, b) => a.x - b.x).forEach(q => {
      const k = q.d.mc.kind, n = rows[k] = (rows[k] || 0) + 1, right = q.x < W - m.r - 190;
      svg += `<circle cx="${q.x.toFixed(1)}" cy="${q.y}" r="9" class="halo"/><text x="${(right ? q.x + 12 : q.x - 12).toFixed(1)}" y="${q.y - 10 - 13 * (n - 1)}" text-anchor="${right ? "start" : "end"}" class="t-lab">${esc(q.d.mc.name)} · ${q.d.v.est ? "~" : ""}${esc(fmtS(q.d.v.total))}</text>`;
    });
    if (!pts.length) svg += `<text x="${(m.l + W - m.r) / 2}" y="${(m.t + H - m.b) / 2}" text-anchor="middle" class="t-ink">No machine to show. Switch a backend on.</text>`;
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="How each class of the fleet is spread over whole-task time">`
      + `<defs><clipPath id="made-clip"><rect x="0" y="0" width="${tx.toFixed(1)}" height="${H}"/></clipPath><clipPath id="best-clip"><rect x="0" y="0" width="${ox.toFixed(1)}" height="${H}"/></clipPath></defs>${svg}</svg>`;
  }

  /* ---------- compare: N machines side by side ---------- */
  function renderCompare() {
    const p = task();
    const rows = state.sel.map(id => machineRow(MACHINE[id], state.model, p));
    document.getElementById("cmp-empty").hidden = rows.length > 0;
    document.getElementById("cmp-body").hidden = rows.length === 0;
    if (!rows.length) return;
    const sorted = rows.slice().sort(byTotal);
    const max = Math.max.apply(null, sorted.filter(r => r.v).map(r => r.v.total).concat([state.target])) * 1.02;
    const best = sorted[0].v ? sorted[0].v.total : null;
    const tpos = state.target / max * 100;
    const tmark = lab => `<div class="tmark${tpos > 88 ? " edge" : ""}" style="left:${tpos.toFixed(2)}%">${lab ? `<span>${esc(fmtS(state.target))} target</span>` : ""}</div>`;
    const bar = (v, kind) => {
      const w = v.total / max * 100;
      return `<div class="bar${w > 100 ? " clip" : ""}" style="width:${Math.min(100, Math.max(0.6, w)).toFixed(2)}%"><i class="bg-${kind} p-read" style="flex:${v.read_s}"></i><i class="bg-${kind}" style="flex:${v.write_s}"></i></div>`;
    };
    document.getElementById("cmp-rows").innerHTML = sorted.map((r, i) => {
      const name = `<div class="name">${swatch(r.mc)}<span>${esc(r.mc.name)}<small>${r.lane ? "on " + esc(inMachine(r.lane)) + " · " + esc(how(r.lane)) : "no device on the backends switched on"}</small></span><button class="x" type="button" data-remove="${esc(r.mc.id)}" aria-label="Remove ${esc(r.mc.name)}">×</button></div>`;
      if (!r.v) return `<div class="mrow"><div class="row">${name}<div class="track fine">${r.lane ? "not measurable for this task" : ""}</div><div class="end"></div></div></div>`;
      const ratio = best ? r.v.total / best : null;
      const main = `<div class="row">${name}<div class="track">${bar(r.v, r.mc.kind)}${tmark(i === 0)}</div>
        <div class="end"><b>${r.v.est ? "~" : ""}${esc(fmtS(r.v.total))}</b><small>${i === 0 ? (rows.length > 1 ? "fastest" : "") : ratio ? (ratio >= 9.95 ? Math.round(ratio) : ratio.toFixed(1)) + "× the fastest" : ""}${meets(r.v) ? "" : (i === 0 && rows.length < 2 ? "" : " · ") + "over target"}</small></div></div>`;
      const subs = r.opts.slice(1).filter(d => d.v).map(d => `<div class="row sub"><div class="name">${esc(inMachine(d.lane))} · ${esc(how(d.lane))}</div><div class="track">${bar(d.v, r.mc.kind)}${tmark(false)}</div><div class="end">${d.v.est ? "~" : ""}${esc(fmtS(d.v.total))}</div></div>`).join("");
      return `<div class="mrow">${main}${subs}</div>`;
    }).join("");
    document.getElementById("cmp-key").innerHTML = `<span><i class="swatch" style="border-radius:2px;background:var(--ink2);opacity:.5"></i>reading the prompt</span><span><i class="swatch" style="border-radius:2px;background:var(--ink2)"></i>writing the reply</span><span><i class="swatch" style="border-radius:2px;background:var(--ink2);height:5px;opacity:.6"></i>the machine's other devices, cut off at the edge when slower</span>`;

    /* numbers: one column per machine, best value in bold */
    const col = sorted;
    const metric = (label, get, fmt, better) => {
      const vals = col.map(get), nums = vals.filter(v => v !== null && v !== undefined && isFinite(v));
      const top = nums.length > 1 && better ? (better === "low" ? Math.min.apply(null, nums) : Math.max.apply(null, nums)) : null;
      return `<tr><td>${label}</td>${vals.map(v => `<td class="num">${v === null || v === undefined ? "–" : `<span class="${v === top ? "best" : ""}">${esc(fmt(v))}</span>`}</td>`).join("")}</tr>`;
    };
    const text = (label, get) => `<tr><td>${label}</td>${col.map(r => `<td class="txt">${esc(get(r))}</td>`).join("")}</tr>`;
    const sec = name => `<tr class="sec"><td colspan="${col.length + 1}">${name}</td></tr>`;
    const onCls = cls => r => { const d = devices(r.mc, state.model, p, true).find(o => o.lane.cls === cls); return d && d.v ? d.v.total : null; };
    document.getElementById("cmp-table").innerHTML = `<thead><tr><th></th>${col.map(r => `<th class="dev">${esc(r.mc.name)} <button class="x" type="button" data-remove="${esc(r.mc.id)}" aria-label="Remove ${esc(r.mc.name)}">×</button><small>${esc(KIND[r.mc.kind].label)}</small></th>`).join("")}</tr></thead><tbody>`
      + sec("This task on " + esc(state.model))
      + text("Runs on", r => r.lane ? inMachine(r.lane) + " · " + how(r.lane) : "–")
      + metric("Whole task", r => r.v && r.v.total, fmtS, "low")
      + metric("First word after", r => r.v && r.v.read_s, fmtS, "low")
      + metric("Reads at", r => r.v && r.v.read, v => fmtN(v) + " tok/s", "high")
      + metric("Writes at", r => r.v && r.v.gen, v => fmtN(v) + " tok/s", "high")
      + metric("Memory reserved", r => r.rec ? reserve(r.rec, p) : null, fmtGB, "low")
      + sec("Whole task on each of its devices")
      + metric("CPU", onCls("CPU"), fmtS, "low")
      + metric("Integrated GPU", onCls("integrated GPU"), fmtS, "low")
      + metric("Discrete GPU", onCls("discrete GPU"), fmtS, "low")
      + sec("Whole task on each model")
      + D.models.map(mo => metric(esc(mo.id), r => { const o = machineRow(r.mc, mo.id, p); return o.v && o.v.total; }, fmtS, "low")).join("")
      + sec("Once per launch")
      + metric("Model load, warm disk", r => r.rec && r.rec.load_s, fmtS, "low")
      + metric("First-ever launch adds", r => r.rec && r.rec.compile_s, v => v < 0.1 ? "under 0.1 s" : fmtS(v), null)
      + sec("Probes of the device it runs on, no model loaded")
      + metric("Compute, f16 GEMM", r => r.lane && r.lane.gemm, v => fmtN(v) + " TFLOP/s", "high")
      + metric("Copy", r => r.lane && r.lane.d2d, v => fmtN(v) + " GB/s", "high")
      + sec("Machine")
      + text("System", r => r.mc.full)
      + text("CPU", r => r.mc.cores + " cores / " + r.mc.threads + " threads")
      + text("RAM", r => r.mc.ram === null ? "–" : r.mc.ram + " GB") + text("OS", r => r.mc.os) + `</tbody>`;
    renderScale(sorted);
  }
  function renderScale(rows) {
    const sw = SWEEPS[state.sweep], p0 = task(), N = 48;
    const hi = Math.max(sw.hi, p0[state.sweep]);
    const xs = [];
    for (let i = 0; i <= N; i++) xs.push(Math.round(sw.lo + (hi - sw.lo) * i / N));
    /* a machine's line is its fastest device at every length, so it can switch device along the way */
    const series = rows.filter(r => r.lane).map(r => ({ mc: r.mc, ys: xs.map(x => { const p = Object.assign({}, p0); p[state.sweep] = x; const o = machineRow(r.mc, state.model, p); return o.v ? o.v.total : null; }) }))
      .filter(s => s.ys.some(y => y > 0));
    const el = document.getElementById("scale");
    const W = Math.max(300, el.clientWidth), H = Math.round(Math.min(380, Math.max(260, W * 0.42)));
    const gutter = W < 560 ? 124 : 226, m = { l: 52, r: gutter, t: 14, b: 44 };
    const cap = Math.floor((gutter - 22) / 6.7), fit = s => s.length > cap ? s.slice(0, cap - 1) + "…" : s;
    if (!series.length) { el.innerHTML = `<p class="fine">Nothing to plot for this task.</p>`; return; }
    const flat = [].concat.apply([], series.map(s => s.ys)).filter(y => y > 0).concat([state.target]);
    const dy = padDomain(flat, 1.3);
    const sx = v => m.l + (v - sw.lo) / (hi - sw.lo) * (W - m.l - m.r), sy = logScale(dy[0], dy[1], H - m.b, m.t);
    let svg = "";
    const step = hi / 4 > 1500 ? 2048 : hi / 4 > 400 ? 512 : 128;
    for (let t = 0; t <= hi; t += step) if (t >= sw.lo) svg += `<line x1="${sx(t)}" x2="${sx(t)}" y1="${m.t}" y2="${H - m.b}" class="grid"/><text x="${sx(t)}" y="${H - m.b + 15}" text-anchor="middle">${int(t)}</text>`;
    timeTicks(dy[0], dy[1], v => -sy(v), 28).forEach(t => { svg += `<line x1="${m.l}" x2="${W - m.r}" y1="${sy(t)}" y2="${sy(t)}" class="grid"/><text x="${m.l - 7}" y="${sy(t) + 4}" text-anchor="end">${fmtTick(t)}</text>`; });
    svg += `<text x="${(m.l + W - m.r) / 2}" y="${H - 8}" text-anchor="middle" class="t-ink">${esc(sw.axis)}</text>`;
    svg += `<text transform="translate(13 ${(m.t + H - m.b) / 2}) rotate(-90)" text-anchor="middle" class="t-ink">whole task, log scale</text>`;
    svg += `<line x1="${m.l}" x2="${W - m.r}" y1="${sy(state.target)}" y2="${sy(state.target)}" class="target"/>`;
    const path = ys => { let d = "", pen = false; ys.forEach((y, i) => { if (y > 0) { d += (pen ? "L" : "M") + sx(xs[i]).toFixed(1) + " " + sy(y).toFixed(1); pen = true; } else pen = false; }); return d; };
    const ends = [{ y: sy(state.target), text: fmtS(state.target) + " target", target: true }];
    series.forEach(s => {
      svg += `<path d="${path(s.ys)}" style="fill:none;stroke:var(--bg);stroke-width:5"/><path d="${path(s.ys)}" class="ln k-${s.mc.kind}" style="fill:none"/>`;
      if (s.ys[N] > 0) ends.push({ y: sy(s.ys[N]), text: fit(s.mc.name), kind: s.mc.kind });
    });
    ends.sort((a, b) => a.y - b.y);
    for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 14) ends[i].y = ends[i - 1].y + 14;
    ends.forEach(e => {
      const x = W - m.r + 6;
      svg += (e.kind ? dot(x + 4, e.y.toFixed(1), 3.5, "k-" + e.kind) : "") + `<text x="${x + 12}" y="${(e.y + 4).toFixed(1)}" class="${e.target ? "t-target" : "t-lab"}">${esc(e.text)}</text>`;
    });
    const cur = p0[state.sweep];
    if (cur >= sw.lo && cur <= hi) {
      const flip = sx(cur) > W - m.r - 70;
      svg += `<line x1="${sx(cur)}" x2="${sx(cur)}" y1="${m.t}" y2="${H - m.b}" class="marker"/><text x="${sx(cur) + (flip ? -5 : 5)}" y="${m.t + 10}" text-anchor="${flip ? "end" : "start"}" class="t-ink t-ref">this task</text>`;
    }
    svg += `<g id="scale-hover"></g><rect x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="transparent" id="scale-hit"/>`;
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Whole-task time against ${esc(sw.axis)}, one line per machine">${svg}</svg>`;
    const hit = document.getElementById("scale-hit"), layer = document.getElementById("scale-hover");
    hit.addEventListener("mousemove", ev => {
      const box = hit.getBoundingClientRect(), k = (W - m.l - m.r) / box.width;
      const px = m.l + (ev.clientX - box.left) * k;
      const i = Math.max(0, Math.min(N, Math.round((px - m.l) / (W - m.l - m.r) * N)));
      let html = `<i>${int(xs[i])} tokens</i>`, dots = `<line x1="${sx(xs[i])}" x2="${sx(xs[i])}" y1="${m.t}" y2="${H - m.b}" class="marker"/>`;
      series.slice().sort((a, b) => (a.ys[i] || 0) - (b.ys[i] || 0)).forEach(s => {
        if (!(s.ys[i] > 0)) return;
        dots += dot(sx(xs[i]).toFixed(1), sy(s.ys[i]).toFixed(1), 4.5, "dot k-" + s.mc.kind);
        html += `<br><span class="num">${esc(fmtS(s.ys[i]))}</span> ${esc(s.mc.name)}`;
      });
      layer.innerHTML = dots;
      showTip(ev, html);
    });
    hit.addEventListener("mouseleave", () => { layer.innerHTML = ""; hideTip(); });
  }

  /* ---------- data: machines and their devices, searchable and sortable ---------- */
  const COLS = [
    { key: "machine", label: "Machine", text: true, get: r => r.mc.name },
    { key: "device", label: "Runs on", text: true, get: r => inMachine(r.lane) },
    { key: "backend", label: "Backend", text: true, get: r => FAM[r.lane.family].label },
    { key: "model", label: "Model", text: true, get: r => r.model, multi: true },
    { key: "total", label: "Whole task", get: r => r.v && r.v.total, fmt: (v, r) => (r.v.est ? "~" : "") + fmtS(v) },
    { key: "target", label: "Target", get: r => r.v ? r.v.total - state.target : null, fmt: v => v <= 0 ? "within" : "over by " + fmtS(v), pill: true },
    { key: "first", label: "First word", get: r => r.v && r.v.read_s, fmt: fmtS },
    { key: "read", label: "Reads tok/s", get: r => r.v && r.v.read, fmt: fmtN },
    { key: "gen", label: "Writes tok/s", get: r => r.v && r.v.gen, fmt: fmtN },
    { key: "mem", label: "Memory", get: r => reserve(r.rec, task()), fmt: fmtGB },
    { key: "load", label: "Load", get: r => r.rec.load_s, fmt: fmtS },
    { key: "compile", label: "First launch", get: r => r.rec.compile_s, fmt: fmtS },
    { key: "gemm", label: "Compute TFLOP/s", get: r => r.lane.gemm, fmt: fmtN },
    { key: "d2d", label: "Copy GB/s", get: r => r.lane.d2d, fmt: fmtN },
    { key: "kind", label: "Class", text: true, get: r => KIND[r.mc.kind].label },
    { key: "ram", label: "RAM GB", get: r => r.mc.ram, fmt: fmtN },
    { key: "os", label: "OS", text: true, get: r => r.mc.os },
  ];
  let dataRows = { rows: [], cols: [] };
  function renderData() {
    const p = task(), models = state.allModels ? D.models.map(m => m.id) : [state.model];
    let rows = [];
    models.forEach(mo => D.machines.forEach(mc => devices(mc, mo, p).forEach((d, i) => {
      if (!state.bestOnly || i === 0) rows.push({ mc: mc, lane: d.lane, rec: d.rec, v: d.v, model: mo, top: i === 0 });
    })));
    const terms = state.q.toLowerCase().split(/\s+/).filter(Boolean);
    rows = rows.filter(r => {
      if (state.onlyOk && !meets(r.v)) return false;
      const hay = [r.mc.full, r.lane.short, r.lane.cls, FAM[r.lane.family].label, r.lane.family, KIND[r.mc.kind].label, r.mc.os, r.model].join(" ").toLowerCase();
      return terms.every(t => hay.indexOf(t) >= 0);
    });
    const cols = COLS.filter(c => !c.multi || state.allModels);
    const sc = COLS.find(c => c.key === state.sort.key) || COLS[4];
    rows.sort((a, b) => {
      const x = sc.get(a), y = sc.get(b), nx = x === null || x === undefined, ny = y === null || y === undefined;
      if (nx || ny) return nx === ny ? 0 : nx ? 1 : -1;
      return (sc.text ? String(x).localeCompare(String(y)) : x - y) * state.sort.dir;
    });
    dataRows = { rows: rows, cols: cols };
    const off = FAMILIES.filter(f => !state.on[f.key]).map(f => f.label);
    document.getElementById("data-count").textContent = rows.length + (rows.length === 1 ? " row" : " rows") + (off.length ? " · " + listOf(off) + " switched off" : "");
    document.getElementById("data").innerHTML = `<thead><tr><th class="t">Compare</th>${cols.map(c => `<th class="${c.text ? "t" : ""}"${c.key === state.sort.key ? ` aria-sort="${state.sort.dir > 0 ? "ascending" : "descending"}"` : ""}><button type="button" data-sort="${c.key}">${esc(c.label)}${c.key === state.sort.key ? (state.sort.dir > 0 ? " ↑" : " ↓") : ""}</button></th>`).join("")}</tr></thead><tbody>`
      + (rows.length ? rows.map(r => `<tr><td class="t"><input type="checkbox" data-toggle="${esc(r.mc.id)}" aria-label="Compare ${esc(r.mc.name)}"${state.sel.indexOf(r.mc.id) >= 0 ? " checked" : ""}></td>${cols.map(c => {
        const v = c.get(r);
        if (v === null || v === undefined) return `<td class="${c.text ? "t" : "num"}">–</td>`;
        if (c.key === "machine") return `<td class="t${r.top ? " dev" : ""}">${swatch(r.mc)} ${esc(v)}</td>`;
        if (c.pill) return `<td><span class="pill${v <= 0 ? " ok" : ""}">${v <= 0 ? "✓ " : ""}${esc(c.fmt(v, r))}</span></td>`;
        return `<td class="${c.text ? "t" : "num"}">${esc(c.text ? v : c.fmt(v, r))}</td>`;
      }).join("")}</tr>`).join("") : `<tr><td class="t fine" colspan="${cols.length + 1}">Nothing matches. Clear the search or the target filter, or switch a backend on.</td></tr>`) + `</tbody>`;
  }
  function copyTable() {
    const btn = document.getElementById("copy");
    const lines = [dataRows.cols.map(c => c.label).join("\t")].concat(dataRows.rows.map(r => dataRows.cols.map(c => { const v = c.get(r); return v === null || v === undefined ? "" : c.text ? v : (typeof v === "number" ? +v.toPrecision(4) : v); }).join("\t")));
    const done = msg => { btn.textContent = msg; setTimeout(() => { btn.textContent = "Copy as TSV"; }, 1600); };
    try { navigator.clipboard.writeText(lines.join("\n")).then(() => done("Copied " + dataRows.rows.length + " rows"), () => done("Copy was blocked")); }
    catch (e) { done("Copy was blocked"); }
  }

  /* ---------- selection, tooltip ---------- */
  function toggleSel(id) {
    const i = state.sel.indexOf(id);
    if (i >= 0) state.sel.splice(i, 1);
    else if (state.sel.length < MAX_SEL) state.sel.push(id);
    render();
  }
  function renderSel() {
    document.getElementById("sel-label").textContent = `${state.sel.length} of up to ${MAX_SEL} machines`;
    document.getElementById("tab-compare").textContent = state.sel.length ? `Compare (${state.sel.length})` : "Compare";
    const add = document.getElementById("sel-add");
    add.innerHTML = `<option value="">Add a machine…</option>` + KINDS.map(k => `<optgroup label="${esc(k.label)}">${D.machines.filter(mc => mc.kind === k.key).map(mc => `<option value="${esc(mc.id)}"${state.sel.indexOf(mc.id) >= 0 ? " disabled" : ""}>${esc(mc.name)}</option>`).join("")}</optgroup>`).join("");
    add.disabled = state.sel.length >= MAX_SEL;
  }
  const tip = document.getElementById("tip");
  function showTip(ev, html) {
    tip.innerHTML = html; tip.hidden = false;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    let x = ev.clientX + 14, y = ev.clientY - h - 10;
    if (x + w > window.innerWidth - 8) x = ev.clientX - w - 14;
    if (y < 8) y = ev.clientY + 16;
    tip.style.left = Math.max(8, x) + "px"; tip.style.top = y + "px";
  }
  function hideTip() { tip.hidden = true; }
  function kindTip(key) {
    const c = FLEET.find(x => x.key === key);
    let html = `<b>${esc(KIND[key].label)}</b><span class="num">${pct(c.share)}</span> of the fleet<br><span class="num">${pct(c.opt)}</span> optimal · <span class="num">${Math.round(c.cov * 100) - Math.round(c.opt * 100)}%</span> tolerable`;
    if (!c.curve) return html;
    html += `<br><span class="num">${esc(fmtS(c.fast))}</span> to <span class="num">${esc(fmtS(c.slow))}</span>`;
    TIERS.slice().reverse().forEach((t, j) => { html += `<br>${t.label.toLowerCase()} end: typical <span class="num">${esc(fmtS(c.tiers[TIERS.length - 1 - j].typical))}</span>`; });
    return html;
  }
  function machineTip(id) {
    const r = machineRow(MACHINE[id], state.model, task());
    let html = `<b>${esc(r.mc.full)}</b>`;
    if (!r.lane) html += "no device on the backends switched on";
    if (r.v) html += `<span class="num">${r.v.est ? "~" : ""}${esc(fmtS(r.v.total))}</span> on ${esc(inMachine(r.lane))} · ${esc(how(r.lane))}, ${meets(r.v) ? "within target" : "over by " + esc(fmtS(r.v.total - state.target))}<br><span class="num">${fmtN(r.v.read)}</span> tok/s reading · <span class="num">${fmtN(r.v.gen)}</span> tok/s writing`;
    return html + `<br><i>${state.sel.indexOf(id) >= 0 ? "click to remove from the comparison" : "click to compare"}</i>`;
  }
  const hit = (ev, sel) => ev.target.closest && ev.target.closest(sel);
  document.addEventListener("mousemove", ev => { const c = hit(ev, "[data-kind]"); if (c) showTip(ev, kindTip(c.dataset.kind)); });
  document.addEventListener("mouseover", ev => { const t = hit(ev, "[data-machine]"); if (t) showTip(ev, machineTip(t.dataset.machine)); });
  document.addEventListener("mouseout", ev => {
    if (hit(ev, "[data-kind]") || hit(ev, "[data-machine]")) hideTip();
  });
  function onPick(ev) {
    if (ev.type === "keydown" && ev.key !== "Enter" && ev.key !== " ") return;
    const t = hit(ev, "[data-machine], [data-remove], [data-sort]");
    if (!t || (ev.type === "keydown" && !t.dataset.machine)) return;
    if (t.dataset.machine) { ev.preventDefault(); hideTip(); toggleSel(t.dataset.machine); }
    else if (t.dataset.remove) toggleSel(t.dataset.remove);
    else if (t.dataset.sort) { state.sort = { key: t.dataset.sort, dir: state.sort.key === t.dataset.sort ? -state.sort.dir : 1 }; render(); }
  }
  document.addEventListener("click", onPick);
  document.addEventListener("keydown", onPick);
  document.addEventListener("change", ev => {
    const d = ev.target.dataset || {};
    if (d.toggle) toggleSel(d.toggle);
    else if (d.family) { state.on[d.family] = ev.target.checked; render(); }
  });

  /* ---------- controls ---------- */
  /* the length sliders are finer at the short end, where a few tokens matter */
  const toSlider = (v, max) => Math.round(Math.pow(Math.min(v, max) / max, 1 / 2.5) * 1000);
  function fromSlider(t, max) {
    const v = max * Math.pow(t / 1000, 2.5), step = v < 256 ? 8 : v < 2048 ? 32 : 128;
    return Math.round(v / step) * step;
  }
  function syncInputs(skip) {
    Object.keys(SLIDERS).forEach(k => {
      if (skip !== "in-" + k) document.getElementById("in-" + k).value = state[k];
      if (skip !== "r-" + k) document.getElementById("r-" + k).value = toSlider(state[k], SLIDERS[k]);
    });
  }
  /* a row of buttons where exactly one is pressed */
  function segment(id, items, current, onChoose) {
    const el = document.getElementById(id);
    el.innerHTML = items.map(it => `<button type="button" data-key="${esc(it.key)}" aria-pressed="${it.key === current()}">${esc(it.label)}</button>`).join("");
    el.addEventListener("click", ev => { const b = ev.target.closest("button"); if (!b) return; onChoose(b.dataset.key); render(); });
    return () => el.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.key === current())));
  }
  /* A split bar: one rectangle shared out between its items, with a separator to
     drag (or move with the arrow keys) between each pair, and the names and values
     listed underneath. `bar(el, legend, items, cuts, set, values)`:
       cuts()     the separators' places in percent, ascending
       set(i, at) move separator i to `at` percent, or by `step` arrow presses
       values()   the text under each item */
  function splitBar(el, legend, items, cuts, set, values) {
    el.innerHTML = `<div class="sp-track">${items.map(it => `<div class="sp-seg ${it.cls}"></div>`).join("")}</div>`
      + items.slice(1).map((it, i) => `<div class="sp-h" role="slider" tabindex="0" data-i="${i}" aria-orientation="horizontal" aria-valuemin="0" aria-valuemax="100" aria-label="Boundary between ${esc(items[i].label)} and ${esc(it.label)}"></div>`).join("");
    legend.innerHTML = items.map(it => `<span><i class="sp-sw ${it.cls}"></i>${esc(it.label)}<b></b></span>`).join("");
    el.querySelectorAll(".sp-h").forEach(h => {
      const i = +h.dataset.i;
      h.addEventListener("pointerdown", ev => { ev.preventDefault(); h.setPointerCapture(ev.pointerId); h.focus(); });
      h.addEventListener("pointermove", ev => {
        if (!h.hasPointerCapture(ev.pointerId)) return;
        const b = el.getBoundingClientRect();
        set(i, Math.max(0, Math.min(100, (ev.clientX - b.left) / b.width * 100)), 0);
        render();
      });
      h.addEventListener("keydown", ev => {
        const d = ev.key === "ArrowLeft" || ev.key === "ArrowDown" ? -1 : ev.key === "ArrowRight" || ev.key === "ArrowUp" ? 1 : 0;
        if (d) { ev.preventDefault(); set(i, null, d * (ev.shiftKey ? 5 : 1)); render(); }
      });
    });
    return () => {
      const c = [0].concat(cuts(), [100]), text = values();
      items.forEach((it, j) => {
        const seg = el.firstElementChild.children[j];
        seg.style.left = c[j] + "%"; seg.style.width = (c[j + 1] - c[j]) + "%";
        legend.children[j].querySelector("b").textContent = text[j];
        if (j < items.length - 1) {
          const h = el.querySelector(`.sp-h[data-i="${j}"]`);
          h.style.left = c[j + 1] + "%";
          h.setAttribute("aria-valuenow", Math.round(c[j + 1]));
          h.setAttribute("aria-valuetext", items.map((x, n) => x.label + " " + text[n]).join(", "));
        }
      });
    };
  }
  /* shares of a whole: a separator moves share between its two neighbours, one percent at a time */
  function shareBar(el, legend, group, items) {
    const shares = () => norm(items.map(it => state.fleet[group][it.key])).map(v => v * 100);
    const edge = (sh, i) => sh.slice(0, i + 1).reduce((a, b) => a + b, 0);
    return splitBar(el, legend, items,
      () => { const sh = shares(); return items.slice(1).map((it, i) => edge(sh, i)); },
      (i, at, step) => {
        const sh = shares(), lo = i > 0 ? edge(sh, i - 1) : 0, hi = edge(sh, i + 1);
        const pos = Math.round(Math.max(lo, Math.min(hi, at === null ? edge(sh, i) + step : at)));
        items.forEach((it, j) => { state.fleet[group][it.key] = j === i ? pos - lo : j === i + 1 ? hi - pos : sh[j]; });
      },
      () => shares().map(v => Math.round(v) + "%"));
  }
  /* the two time limits on a log scale: optimal up to the first separator, tolerable
     up to the second, unusable beyond. A limit snaps as finely as it is shown. */
  function limitBar(el, legend) {
    const keys = ["optimal", "target"], span = Math.log(LIMIT_RANGE[1] / LIMIT_RANGE[0]);
    const grain = v => v < 10 ? 0.1 : v < 60 ? 1 : 5;
    return splitBar(el, legend, [{ label: "Optimal", cls: "z0" }, { label: "Tolerable", cls: "z1" }, { label: "Unusable", cls: "z2" }],
      () => keys.map(k => Math.log(state[k] / LIMIT_RANGE[0]) / span * 100),
      (i, at, step) => {
        const was = state[keys[i]];
        let v = at === null ? was + step * grain(step < 0 ? was - 1e-9 : was) : LIMIT_RANGE[0] * Math.exp(at / 100 * span);
        v = Math.max(LIMIT_RANGE[0], Math.min(LIMIT_RANGE[1], Math.round(v / grain(v)) * grain(v)));
        if (i === 0) state.optimal = Math.min(v, state.target); else state.target = Math.max(v, state.optimal);
      },
      () => ["up to " + fmtS(state.optimal), "up to " + fmtS(state.target), ""]);
  }

  const refresh = [];
  function setup() {
    const models = document.getElementById("models");
    models.innerHTML = D.models.map(m => `<option value="${esc(m.id)}">${esc(m.id)}${m.params_b ? " · " + m.params_b + " B" : ""}${m.file_gb ? " · " + m.file_gb + " GB" : ""}</option>`).join("");
    models.addEventListener("change", () => { state.model = models.value; render(); });
    refresh.push(() => { models.value = state.model; });
    refresh.push(segment("scale-var", Object.keys(SWEEPS).map(k => ({ key: k, label: SWEEPS[k].label })), () => state.sweep, k => { state.sweep = k; }));
    Object.keys(SLIDERS).forEach(k => {
      document.getElementById("r-" + k).addEventListener("input", ev => { state[k] = fromSlider(+ev.target.value, SLIDERS[k]); syncInputs("r-" + k); render(); });
      document.getElementById("in-" + k).addEventListener("input", ev => { state[k] = Math.max(0, Math.round(+ev.target.value || 0)); syncInputs("in-" + k); render(); });
      const st = document.getElementById("s-" + k);
      st.innerHTML = STOPS[k].map(s => `<button type="button" data-v="${s.v}">${esc(s.l)}</button>`).join("");
      st.addEventListener("click", ev => { const b = ev.target.closest("button"); if (!b) return; state[k] = +b.dataset.v; syncInputs(); render(); });
    });
    refresh.push(limitBar(document.getElementById("split-zone"), document.getElementById("legend-zone")));
    document.getElementById("backends").innerHTML = FAMILIES.map(f => `<label><input type="checkbox" data-family="${esc(f.key)}"${state.on[f.key] ? " checked" : ""}> ${esc(f.label)}</label>`).join("");
    refresh.push(shareBar(document.getElementById("split-kind"), document.getElementById("legend-kind"), "kind", KINDS.map(k => ({ key: k.key, label: k.label, cls: "bg-" + k.key }))));
    refresh.push(shareBar(document.getElementById("split-tier"), document.getElementById("legend-tier"), "tier", TIERS.slice().reverse().map((t, i) => ({ key: t.key, label: t.label, cls: "t" + i }))));
    document.getElementById("sel-add").addEventListener("change", ev => { if (ev.target.value) toggleSel(ev.target.value); });
    document.getElementById("tabs").addEventListener("click", ev => { const b = ev.target.closest("button"); if (!b) return; state.view = b.dataset.view; render(); });
    document.getElementById("q").addEventListener("input", ev => { state.q = ev.target.value; renderData(); });
    document.getElementById("best-only").addEventListener("change", ev => { state.bestOnly = ev.target.checked; renderData(); });
    document.getElementById("all-models").addEventListener("change", ev => { state.allModels = ev.target.checked; renderData(); });
    document.getElementById("only-ok").addEventListener("change", ev => { state.onlyOk = ev.target.checked; renderData(); });
    document.getElementById("copy").addEventListener("click", copyTable);
    syncInputs();
    let timer = null;
    window.addEventListener("resize", () => { clearTimeout(timer); timer = setTimeout(render, 120); });
  }
  function render() {
    document.querySelectorAll("#tabs button").forEach(b => b.setAttribute("aria-selected", String(b.dataset.view === state.view)));
    ["coverage", "compare", "data"].forEach(v => { document.getElementById("view-" + v).hidden = v !== state.view; });
    const all = priced(state.model, task());
    FLEET = fleetNow(task());
    refresh.forEach(f => f());
    renderSel();
    renderAnswer(all);
    if (state.view === "coverage") renderMap(all);
    if (state.view === "compare") renderCompare();
    if (state.view === "data") renderData();
  }
  setup();
  render();
})();
