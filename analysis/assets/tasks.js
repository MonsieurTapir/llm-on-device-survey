/* The task calculator: what one prompt and one reply cost on one device.
 *
 * Pure functions over the records `site._records` embeds — no DOM, no globals
 * besides the one export. A record carries, per (lane, model), the sweep's two
 * measured cost curves: cumulative time to first token over prompt depth, and the
 * decode ladder. A task is a prompt read from an empty context and a reply written
 * after it.
 *
 * Both curves are measurements, never fits of them: inside the measured range a
 * price is the sweep's own clock, interpolated, so a lane's nonlinearities price
 * themselves. Past the deepest point either curve reached, its last measured rate
 * is carried onward and the price is marked as an estimate (`est`) — the marginal
 * cost was still rising everywhere measured, so that is the optimistic side. A
 * price is refused (null) only where there is nothing to evaluate: a lane that
 * measured no curve, or a task that does not fit the model's own trained context.
 *
 * A CPU record also carries its thread ladder (`scaling`): per phase, the rate at
 * a few thread counts up to the one the lane ran at. A price can be asked for the
 * same CPU held to fewer threads, and is the measured price scaled by the ladder's
 * own ratio between the two widths, reading and writing each by theirs.
 */
window.taskMath = (function () {
  "use strict";

  /* Decode is integrated in steps this size: the rate falls as the cache fills, so
   * a long reply cannot be priced at its starting rate. Small enough that the error
   * against a continuous integral is far under the spread of the measurement. */
  const STEP = 32;

  /* T(d): cumulative milliseconds to read a prompt to depth `d`, read off the
   * sweep's measured curve — `[depth, ms]` pairs ascending from the `[0, 0]` origin —
   * with linear interpolation between the measured points. Past the deepest point
   * the last segment's per-token cost is carried onward. */
  function cumMs(curve, d) {
    if (!(d > 0)) return 0;
    const last = curve[curve.length - 1];
    if (d > last[0]) {
      const prev = curve[curve.length - 2];
      return last[1] + (d - last[0]) * (last[1] - prev[1]) / (last[0] - prev[0]);
    }
    for (let i = 1; i < curve.length; i++) {
      if (d <= curve[i][0]) {
        const a = curve[i - 1], b = curve[i];
        return a[1] + (d - a[0]) * (b[1] - a[1]) / (b[0] - a[0]);
      }
    }
    return last[1];
  }

  /* The rate a ladder of `[x, tok/s]` pairs gives at `fill`: linear between the
   * measured points, held flat outside them. The decode ladder is over the tokens
   * already in the context, a thread ladder over thread counts. */
  function tpsAt(ladder, fill) {
    if (fill <= ladder[0][0]) return ladder[0][1];
    for (let i = 1; i < ladder.length; i++) {
      const lo = ladder[i - 1], hi = ladder[i];
      if (fill <= hi[0]) {
        return hi[0] === lo[0] ? hi[1] : lo[1] + (hi[1] - lo[1]) * (fill - lo[0]) / (hi[0] - lo[0]);
      }
    }
    return ladder[ladder.length - 1][1];
  }

  /* Seconds to write `n` tokens starting from `fill`. */
  function decodeSeconds(ladder, fill, n) {
    let s = 0, done = 0;
    while (done < n) {
      const take = Math.min(STEP, n - done);
      s += take / tpsAt(ladder, fill + done + take / 2);
      done += take;
    }
    return s;
  }

  /* How many times longer one phase takes on `threads` threads than at the width
   * the lane ran it: the ratio of the two rates on the phase's thread ladder,
   * `{threads, points}` with `points` as `[threads, tok/s]` pairs ascending. A lane
   * is never priced wider than it ran, and a phase with no ladder is left as it is. */
  function heldTo(phase, threads) {
    if (!phase || !(threads > 0) || threads >= phase.threads) return 1;
    return tpsAt(phase.points, phase.threads) / tpsAt(phase.points, threads);
  }

  /* One task on one record: seconds to read the prompt, seconds to write the reply,
   * their sum, and the two average speeds. `task` is `{prompt, out}` in tokens.
   * With an empty prompt or reply the matching speed falls back to the rate the
   * curve starts at, so every priced record has both. `threads`, when given, prices
   * a CPU record held to that many threads. */
  function price(rec, task, threads) {
    if (!rec || rec.curve.length < 2 || !rec.ladder.length) return null;
    if (rec.n_ctx_train > 0 && task.prompt + task.out > rec.n_ctx_train) return null;
    const sc = rec.scaling || {}, slowRead = heldTo(sc.prefill, threads), slowWrite = heldTo(sc.decode, threads);
    const read_s = slowRead * cumMs(rec.curve, task.prompt) / 1000;
    const write_s = slowWrite * decodeSeconds(rec.ladder, task.prompt, task.out);
    return {
      read_s: read_s,
      write_s: write_s,
      total: read_s + write_s,
      read: task.prompt > 0 && read_s > 0 ? task.prompt / read_s : 512000 / cumMs(rec.curve, 512) / slowRead,
      gen: task.out > 0 && write_s > 0 ? task.out / write_s : tpsAt(rec.ladder, task.prompt) / slowWrite,
      est: !(rec.pre_max !== null && task.prompt <= rec.pre_max) || rec.ladder.length < 2
        || !(rec.kv_max !== null && task.prompt + task.out <= rec.kv_max),
    };
  }

  return { cumMs: cumMs, tpsAt: tpsAt, decodeSeconds: decodeSeconds, heldTo: heldTo, price: price };
})();
