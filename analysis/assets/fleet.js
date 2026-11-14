/* The fleet model: how much of a fleet of machines finishes a task in time.
 *
 * Pure functions — no DOM, no globals besides the one export. Coverage needs one
 * number per machine: how long the task takes on it. Reading and writing speed rise
 * together across machines, so the model works on that one number, in log time.
 *
 * A fleet is a share per machine class, and inside each class a low / mid / high
 * split. A class is a stretch of time, from its slowest machine to its fastest; the
 * split spreads the class over the stretch as one smooth curve (a Beta density).
 * The stretch's thirds are the low, mid and high tiers, and the curve is the one
 * whose share in each third is the share set. The split never moves the ends of
 * the stretch.
 *
 * The measurements set the middle and the two ends:
 *   - `typical`, the times of the CPUs the class draws on, each as a typical
 *     machine has it. Their mean, in log time, is the middle of the stretch.
 *   - `cpu`, every time those CPUs can show, the typical ones among them. With no
 *     GPU in use the stretch reaches PAD past the one furthest from the middle,
 *     and as far on the other side: the fleet has machines as far below the
 *     typical one as the fastest measured is above it.
 *   - `gpu`, one entry per machine of the class: its time on its fastest GPU in
 *     use, null without one. A GPU only pulls an end in. The fast end is the
 *     fastest GPU's where that beats the fastest CPU. The slow end is the slowest
 *     GPU's where every machine has one and it beats the slowest CPU: the slowest
 *     machine of a class has the slowest of each and runs on the better of the two.
 * With no CPU time the GPUs set both ends, over the machines that have one. A
 * stretch shorter than MIN_SPAN is widened about its middle.
 *
 * Classes handed the same CPU times cover the same stretch until a GPU moves one
 * of them, and each leans inside it toward where its own machines sit: a class
 * whose machines are faster than the CPUs it draws on has more of its weight at
 * the fast end. The lean moves the mean of the curve and keeps how concentrated the
 * split made it, so a class that leans holds other shares in its thirds than the
 * split set. It counts for n / (n + SHRINK) with n machines, and it is the same
 * whatever backends are in use.
 *
 * A machine never gets slower for having more devices to choose from, so with the
 * CPU in use neither end moves later when a backend is added, and coverage never
 * falls.
 *
 * A class's coverage is the share of its curve at or under the time limit, and its
 * curve is returned scaled by the class's share of the fleet, so areas compare
 * across classes: area is share of the fleet, and the area under a limit is the
 * share of the fleet within it.
 */
window.fleetMath = (function () {
  "use strict";

  const PAD = Math.log(1.25);        /* how far a stretch reaches past its slowest and fastest time */
  const MIN_SPAN = Math.log(4);      /* the shortest stretch: a class is never one point */
  const SHRINK = 4;                  /* the machines at which a class's own lean counts for half */
  const CURVE_POINTS = 120;
  /* the three tiers as stretches of [0, 1], running from the slow end to the fast */
  const TIERS = [[0, 1 / 3], [1 / 3, 2 / 3], [2 / 3, 1]];

  const sum = a => a.reduce((x, y) => x + y, 0);
  const mean = a => sum(a) / a.length;
  const most = a => Math.max.apply(null, a), least = a => Math.min.apply(null, a);
  const clamp = v => Math.max(0, Math.min(1, v));
  function norm(a) {
    const s = sum(a);
    return s > 0 ? a.map(v => v / s) : a.map(() => 1 / a.length);
  }

  function lgamma(x) {
    const c = [76.18009172947146, -86.50532032941677, 24.01409824083091,
               -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
    let y = x, tmp = x + 5.5, ser = 1.000000000190015;
    tmp -= (x + 0.5) * Math.log(tmp);
    for (let j = 0; j < 6; j++) ser += c[j] / ++y;
    return -tmp + Math.log(2.5066282746310005 * ser / x);
  }
  /* the continued fraction of the incomplete beta function (Lentz's method) */
  function betaCf(x, a, b) {
    const tiny = 1e-30, fix = v => Math.abs(v) < tiny ? tiny : v;
    const qab = a + b, qap = a + 1, qam = a - 1;
    let c = 1, d = 1 / fix(1 - qab * x / qap), h = d;
    for (let m = 1; m <= 300; m++) {
      const m2 = 2 * m;
      let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
      d = 1 / fix(1 + aa * d); c = fix(1 + aa / c); h *= d * c;
      aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
      d = 1 / fix(1 + aa * d); c = fix(1 + aa / c);
      const del = d * c;
      h *= del;
      if (Math.abs(del - 1) < 3e-10) break;
    }
    return h;
  }
  /* the share of a Beta(a, b) below x */
  function betaBelow(x, a, b) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    const bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
    return x < (a + 1) / (a + b + 2) ? bt * betaCf(x, a, b) / a : 1 - bt * betaCf(1 - x, b, a) / b;
  }
  /* the density of a Beta(a, b) at x */
  function betaAt(x, a, b) {
    return Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + (a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x));
  }
  /* the x below which a Beta(a, b) holds the share q */
  function betaPoint(q, a, b) {
    let lo = 0, hi = 1;
    for (let n = 0; n < 44; n++) {
      const mid = (lo + hi) / 2;
      if (betaBelow(mid, a, b) < q) lo = mid; else hi = mid;
    }
    return (lo + hi) / 2;
  }

  /* The curve a tier split asks for: the Beta(a, b) on [0, 1] whose share below 1/3
   * is the low share and above 2/3 the high share. `split` is [low, mid, high], in
   * any unit. Two nested bisections; a share of zero is held just above it, where a
   * Beta still exists. */
  const curves = {};
  function tierCurve(split) {
    const s = norm(split.map(v => Math.max(0.004, v))), key = s.map(v => v.toFixed(4)).join("/");
    if (curves[key]) return curves[key];
    const LO = Math.log(0.12), HI = Math.log(120);
    const bFor = a => {
      let lo = LO, hi = HI;
      for (let n = 0; n < 40; n++) {
        const m = (lo + hi) / 2;
        if (betaBelow(1 / 3, a, Math.exp(m)) < s[0]) lo = m; else hi = m;
      }
      return Math.exp((lo + hi) / 2);
    };
    let lo = LO, hi = HI;
    for (let n = 0; n < 40; n++) {
      const m = (lo + hi) / 2, a = Math.exp(m);
      if (betaBelow(2 / 3, a, bFor(a)) < 1 - s[2]) lo = m; else hi = m;
    }
    const a = Math.exp((lo + hi) / 2);
    return (curves[key] = { a: a, b: bFor(a) });
  }

  /* The CPU stretch in log time: centred on the typical times, reaching PAD past
   * the time furthest from there and as far on the other side. */
  function cpuStretch(typical, cpu) {
    const all = cpu.map(Math.log), middle = mean(typical.length ? typical.map(Math.log) : all);
    const reach = Math.max(most(all) - middle, middle - least(all)) + PAD;
    return { late: middle + reach, early: middle - reach };
  }

  /* How far a class leans inside its CPU stretch, as a part of the stretch's length:
   * positive toward the fast end. `own` is the typical times of the class's own
   * machines, `typical` and `cpu` those of every CPU it draws on. */
  function lean(typical, cpu, own) {
    if (!own.length || !typical.length || !cpu.length) return 0;
    const s = cpuStretch(typical, cpu), gap = mean(typical.map(Math.log)) - mean(own.map(Math.log));
    return own.length / (own.length + SHRINK) * gap / (s.late - s.early);
  }

  /* A curve with its mean moved by `by` and its concentration kept. */
  function leaned(tc, by) {
    const n = tc.a + tc.b, m = Math.max(0.05, Math.min(0.95, tc.a / n + by));
    return by ? { a: n * m, b: n * (1 - m) } : tc;
  }

  /* The fleet for one task.
   *   classes: [{key, share, typical: [seconds…], cpu: [seconds…],
   *              gpu: [seconds or null, one per machine], lean}]
   *   split:   [low, mid, high] shares inside every class, before its `lean`
   *   optimal, target: the two time limits in seconds, optimal <= target
   * Returns one entry per class, in order: its normalised `share`; `cov` and `opt`,
   * the share of the class within `target` and within `optimal`; `fast` and `slow`,
   * the ends of its stretch in seconds; `curve`, `[{t, d}]` from slow to fast, and
   * `cuts`, the same at the two tier boundaries; `tiers`, slowest first, each with
   * its share of the fleet `w`, the share of it within target `frac`, and its
   * median time `typical`. A class with no time to place has `curve` null and
   * covers nothing. */
  function fleet(classes, split, optimal, target) {
    const shares = norm(classes.map(c => c.share)), set = tierCurve(split);
    return classes.map((c, i) => {
      const tc = leaned(set, c.lean || 0), edges = [0, betaBelow(1 / 3, tc.a, tc.b), betaBelow(2 / 3, tc.a, tc.b), 1];
      const out = { key: c.key, share: shares[i], cov: 0, opt: 0, fast: null, slow: null, curve: null, cuts: null,
                    tiers: TIERS.map((t, j) => ({ w: shares[i] * (edges[j + 1] - edges[j]), frac: 0, typical: null })) };
      const cpu = c.cpu.map(Math.log), gpu = c.gpu.filter(t => t !== null).map(Math.log);
      if (!cpu.length && !gpu.length) return out;
      let late = Infinity, early = Infinity;
      if (cpu.length) {
        const s = cpuStretch(c.typical, c.cpu);
        late = s.late;
        early = s.early;
      }
      if (gpu.length) {
        early = Math.min(early, least(gpu) - PAD);
        if (!cpu.length || gpu.length === c.gpu.length) late = Math.min(late, most(gpu) + PAD);
      }
      const span = Math.max(late - early, MIN_SPAN), slow = (late + early + span) / 2, timeAt = u => Math.exp(slow - u * span);
      const below = limit => betaBelow(clamp((slow - Math.log(limit)) / span), tc.a, tc.b);
      const over = below(target);
      out.cov = 1 - over;
      out.opt = 1 - below(optimal);
      out.fast = timeAt(1);
      out.slow = timeAt(0);
      out.tiers.forEach((t, j) => {
        const mass = edges[j + 1] - edges[j];
        t.frac = mass > 1e-9 ? (edges[j + 1] - Math.max(edges[j], Math.min(edges[j + 1], over))) / mass : 0;
        t.typical = timeAt(betaPoint((edges[j] + edges[j + 1]) / 2, tc.a, tc.b));
      });
      /* per unit of log time, scaled by the class's share; the open ends stop short
         of 0 and 1, where a split heavy at an end has no finite height */
      const at = u => ({ t: timeAt(u), d: shares[i] * betaAt(u, tc.a, tc.b) / span });
      out.curve = [];
      for (let n = 0; n <= CURVE_POINTS; n++) out.curve.push(at(0.004 + 0.992 * n / CURVE_POINTS));
      out.cuts = [at(1 / 3), at(2 / 3)];
      return out;
    });
  }

  return { betaBelow: betaBelow, betaAt: betaAt, betaPoint: betaPoint, tierCurve: tierCurve, lean: lean, fleet: fleet };
})();
