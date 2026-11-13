/* The fleet model: how much of a fleet of machines finishes a task in time.
 *
 * Pure functions — no DOM, no globals besides the one export. Coverage needs one
 * number per machine: how long the task takes on it. Reading and writing speed rise
 * together across machines, so the model works on that one number, in log time.
 *
 * A fleet is a share per machine class, and inside each class a low / mid / high
 * split. A class is a fixed stretch of time; the split spreads the class over the
 * stretch as one smooth curve (a Beta density). The stretch's thirds are the low,
 * mid and high tiers, and the curve is the one whose share in each third is the
 * share set. The split never moves the ends of the stretch.
 *
 * What the measurements set:
 *   - `ref`, the class's machines each on its fastest device: the length of the
 *     stretch, from the slowest to the fastest machine plus PAD at each end, pulled
 *     toward the median class's length
 *   - `now`, the same machines on the backends in use: where the stretch sits, from
 *     their mean log time
 * Taking a backend away slides a class without reshaping it, and a machine never
 * gets faster for having fewer devices to choose from, so coverage never rises when
 * a backend is switched off.
 *
 * A class's coverage is the share of its curve at or under the time limit, and its
 * curve is returned scaled by the class's share of the fleet, so areas compare
 * across classes: area is share of the fleet, and the area under a limit is the
 * share of the fleet within it.
 */
window.fleetMath = (function () {
  "use strict";

  const SHRINK = 2;                   /* weight of the pooled length, counted in machines */
  const PAD = 0.25;                   /* of a class's measured span, added at each end */
  const DEFAULT_SPAN = Math.log(6);   /* a class's length when nothing can be pooled */
  const CURVE_POINTS = 120;
  /* the three tiers as stretches of [0, 1], running from the slow end to the fast */
  const TIERS = [[0, 1 / 3], [1 / 3, 2 / 3], [2 / 3, 1]];

  const sum = a => a.reduce((x, y) => x + y, 0);
  const mean = a => sum(a) / a.length;
  const clamp = v => Math.max(0, Math.min(1, v));
  function norm(a) {
    const s = sum(a);
    return s > 0 ? a.map(v => v / s) : a.map(() => 1 / a.length);
  }
  function median(a) {
    const s = a.slice().sort((x, y) => x - y), n = s.length;
    return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
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

  /* The fleet for one task.
   *   classes: [{key, share, ref: [seconds…], now: [seconds…]}]
   *   split:   [low, mid, high] shares inside every class
   *   optimal, target: the two time limits in seconds, optimal <= target
   * Returns one entry per class, in order: its normalised `share`; `cov` and `opt`,
   * the share of the class within `target` and within `optimal`; `fast` and `slow`,
   * the ends of its stretch in seconds; `curve`, `[{t, d}]` from slow to fast, and
   * `cuts`, the same at the two tier boundaries; `tiers`, slowest first, each with
   * its share of the fleet `w`, the share of it within target `frac`, and its
   * median time `typical`. A class with no machine to place has `curve` null and
   * covers nothing. */
  function fleet(classes, split, optimal, target) {
    const shares = norm(classes.map(c => c.share)), parts = norm(split), tc = tierCurve(split);
    const logs = classes.map(c => ({ ref: c.ref.map(Math.log), now: c.now.map(Math.log) }));
    const own = logs.map(l => l.ref.length > 1
      ? (Math.max.apply(null, l.ref) - Math.min.apply(null, l.ref)) * (1 + 2 * PAD) : null);
    const measured = own.filter(v => v !== null);
    const pooled = measured.length ? median(measured) : DEFAULT_SPAN;
    const edges = [0, betaBelow(1 / 3, tc.a, tc.b), betaBelow(2 / 3, tc.a, tc.b), 1];
    return classes.map((c, i) => {
      const ref = logs[i].ref, now = logs[i].now;
      const out = { key: c.key, share: shares[i], cov: 0, opt: 0, fast: null, slow: null, curve: null, cuts: null,
                    tiers: TIERS.map((t, j) => ({ w: shares[i] * parts[j], frac: 0, typical: null })) };
      if (!ref.length || !now.length) return out;
      const span = own[i] === null ? pooled
        : (SHRINK * pooled + (ref.length - 1) * own[i]) / (SHRINK + ref.length - 1);
      /* the stretch keeps its place relative to the class's mean, and the mean follows the backends */
      const middle = mean(now) + (Math.max.apply(null, ref) + Math.min.apply(null, ref)) / 2 - mean(ref);
      const slow = middle + span / 2, timeAt = u => Math.exp(slow - u * span);
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

  return { betaBelow: betaBelow, betaAt: betaAt, betaPoint: betaPoint, tierCurve: tierCurve, fleet: fleet };
})();
