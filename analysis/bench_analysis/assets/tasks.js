/* The task calculator: the page's only arithmetic.
 *
 * Pure functions over the pack `site._task_pack` embeds — no DOM, no vega, no
 * globals besides the one export. The pack carries, per (lane, model), the sweep's
 * two measured cost curves — cumulative time to first token over depth, and the
 * decode ladder — plus the once-per-launch costs; this turns a task configuration
 * into the three columns the chart draws.
 *
 * Both curves are measurements, never fits of them: inside the measured range a
 * prediction is the sweep's own clock, interpolated, so a lane's nonlinearities
 * price themselves. Anything resting on thin evidence is marked as an estimate
 * rather than quietly averaged in — a lane whose ladder holds a single fill, or a
 * depth past the one the sweep reached, where each curve's last measured rate is
 * carried onward (the marginal cost was still rising everywhere measured, so that
 * is the optimistic side and the mark says so). A number is refused only where
 * there is nothing to evaluate: a lane that measured no curve, or a task that does
 * not fit the model's own trained context.
 */
window.taskMath = (function () {
  "use strict";

  /* Decode is integrated in steps this size: the rate falls as the cache fills, so
   * a long reply cannot be priced at its starting rate. Small enough that the error
   * against a continuous integral is far under the spread of the measurement. */
  var STEP = 32;

  /* The two refusals and the marks. The refusals are facts about the model and the
   * lane — a task longer than the context the model was trained on, a lane whose sweep
   * measured no curve to evaluate. The marks say how far the evidence reached.
   * All of them are labels, not sentences: what a half-lit bar means is explained once,
   * in the section's own copy, and a hover is no place to read a paragraph. */
  var BEYOND_CTX = "beyond this model's context";
  var NO_CURVE = "no prompt reading measured";
  var NO_LADDER = "no generation rate measured";
  var FLAT = "one measured fill";
  var HELD = "rate held at the deepest fill measured";

  /* A token is about ¾ of a word — the only unit a reader has a feel for. Rounded to
   * a step that reads as the estimate it is: 25 words under 500, 50 under 2,000, 250
   * above. Mirrors `site._words`, which writes the same counts into the preset notes
   * at build time; keep the two in step. */
  var WORDS_PER_TOKEN = 0.75;

  function words(tokens) {
    if (!(tokens > 0)) return 0;
    var w = tokens * WORDS_PER_TOKEN;
    var step = w < 500 ? 25 : w < 2000 ? 50 : 250;
    return Math.round(w / step) * step;
  }

  /* Which task the configuration belongs to is the select's choice, never derived:
   * editing the token counts edits the selected task and it stays that task. */

  /* T(d): cumulative milliseconds to read a prompt to depth `d`, read straight off
   * the sweep's measured curve — `[depth, ms]` pairs ascending from the harness's
   * own `[0, 0]` origin — with linear interpolation between the measured points.
   * Past the deepest point the last segment's per-token cost is carried onward:
   * the marginal cost was still rising everywhere measured, so holding it is the
   * optimistic side, and the caller marks anything that reaches out there. */
  function cumMs(curve, d) {
    if (!(d > 0)) return 0;
    var last = curve[curve.length - 1];
    if (d > last[0]) {
      var prev = curve[curve.length - 2];
      return last[1] + (d - last[0]) * (last[1] - prev[1]) / (last[0] - prev[0]);
    }
    for (var i = 1; i < curve.length; i++) {
      if (d <= curve[i][0]) {
        var a = curve[i - 1], b = curve[i];
        return a[1] + (d - a[0]) * (b[1] - a[1]) / (b[0] - a[0]);
      }
    }
    return last[1];
  }

  /* Milliseconds to read `n` new tokens with `d0` tokens already in the cache:
   * T(d0+n) − T(d0). The same `n` costs more deeper in because the curve steepens
   * with depth — `d0 = depth` is a chat turn appending to a live cache, `d0 = 0`
   * a document read into an empty one. */
  function prefillMs(curve, d0, n) {
    if (!curve || curve.length < 2 || !(n > 0)) return null;
    return cumMs(curve, d0 + n) - cumMs(curve, d0);
  }

  /* Generation rate at a given cache fill: linear between the measured fills, flat
   * outside them. Flat rather than continued down is deliberate: the ladder's ends are
   * measurements and its slope past them is not, so a reply that runs deeper than the
   * sweep went is priced at the slowest rate actually seen, and marked as an estimate
   * that may still be optimistic. */
  function tpsAt(ladder, kv) {
    if (!ladder || !ladder.length) return null;
    if (kv <= ladder[0][0]) return ladder[0][1];
    for (var i = 1; i < ladder.length; i++) {
      var lo = ladder[i - 1], hi = ladder[i];
      if (kv <= hi[0]) {
        var span = hi[0] - lo[0];
        if (!span) return hi[1];
        return lo[1] + (hi[1] - lo[1]) * (kv - lo[0]) / span;
      }
    }
    return ladder[ladder.length - 1][1];
  }

  /* Seconds to generate `n` tokens starting from a cache holding `kv0`. The rate
   * falls as the reply grows, so this integrates 1/rate over the fill the reply
   * itself adds, at the midpoint of each step. */
  function decodeSeconds(ladder, kv0, n) {
    if (!ladder || !ladder.length) return null;
    if (!(n > 0)) return 0;
    var seconds = 0, done = 0;
    while (done < n) {
      var take = Math.min(STEP, n - done);
      var rate = tpsAt(ladder, kv0 + done + take / 2);
      if (!(rate > 0)) return null;
      seconds += take / rate;
      done += take;
    }
    return seconds;
  }

  /* Whether a configuration sits inside what this lane actually measured, per phase —
   * the prompt no deeper than the deepest prefill point, the reply ending no further
   * than the deepest fill the ladder reached — and whether it fits the context the
   * model was trained for at all. The first two decide whether a number is marked as
   * an estimate; the third is the one that decides whether there is a number. A model
   * whose trained context was not reported cannot rule anything out, so it doesn't. */
  function envelope(rec, p) {
    var deep = p.depth + p.prompt;
    return {
      prefill: rec.curve.length > 1 && rec.pre_max !== null && deep <= rec.pre_max,
      decode: !!rec.ladder.length && rec.kv_max !== null
        && deep + p.out <= rec.kv_max,
      context: !(rec.n_ctx_train > 0) || deep + p.out <= rec.n_ctx_train,
    };
  }

  /* A measured depth as a reader would say it: "3.5k", to the nearest half-thousand
   * tokens, and the plain count under a thousand where halves would be noise. */
  function depthLabel(tokens) {
    if (!(tokens > 0)) return "0";
    if (tokens < 1000) return String(Math.round(tokens));
    var k = Math.round(tokens / 500) / 2;
    return (k % 1 === 0 ? k.toFixed(0) : k.toFixed(1)) + "k";
  }

  /* Seconds a reader can hold: one decimal under ten seconds, then whole seconds,
   * then minutes, then hours. */
  function fmtSeconds(s) {
    if (s === null || s === undefined || !isFinite(s)) return null;
    if (s < 10) return s.toFixed(1) + " s";
    if (s < 60) return Math.round(s) + " s";
    if (s < 3600) {
      var min = Math.floor(s / 60);
      return min + " min " + Math.round(s - 60 * min) + " s";
    }
    var hours = Math.floor(s / 3600);
    return hours + " h " + Math.round((s - 3600 * hours) / 60) + " min";
  }

  function round(value, digits) {
    if (value === null || value === undefined || !isFinite(value)) return null;
    var f = Math.pow(10, digits);
    return Math.round(value * f) / f;
  }

  function join(notes) {
    var kept = notes.filter(function (n) { return !!n; });
    return kept.length ? kept.join("; ") : null;
  }

  /* Tooltip fields carry an em dash where there is nothing to show: vega prints a
   * null field as the word "null", and these are absent by design most of the time — a
   * bar is either an estimate or a refusal or neither, and there is no measurement to
   * compare against outside the job's own preset. Only the fields a mark encodes stay
   * strictly numeric (a dash cannot be plotted). */
  function dash(value) {
    return value === null || value === undefined ? "—" : value;
  }

  /* One chart row per (record, column). `p` is the configuration: task key, depth,
   * prompt, out, and whether the measured job's own numbers belong on it.
   *
   * The three tasks share one arithmetic — read the prompt, write the reply — and
   * differ in what each column reports of it. A chat turn reads only the new message
   * (the depth is already in the KV cache) and watches the rate; read-&-summarize
   * reads the whole prompt from empty and watches the rate; background extraction
   * reports the phases as seconds and its done-after includes getting the model
   * ready, because nobody is watching and only the end matters.
   *
   * Per column, a number or a reason — never both. The prompt columns need the fit,
   * the generation column needs the ladder, and the total needs both, so one missing
   * piece does not blank the row. Everything the cost functions can be evaluated for
   * gets a number; what carries it past the measured depths is said in the mark (half
   * ink, a `~`) and named on hover. */
  function taskRows(pack, p) {
    var deep = p.depth + p.prompt;
    var background = p.task === "extract";
    var rows = [];

    pack.records.forEach(function (rec) {
      var reach = envelope(rec, p);
      var ttft = null, reply = null, rate = null, total = null;
      var estPre = null, estDec = null;
      var notePre = null, noteDec = null;

      if (!reach.context) {
        /* The one wall: the model was never trained this far, so no measurement of
         * ours has anything to say about it. Both columns carry the same fact, and
         * the caller below prints it once. */
        notePre = BEYOND_CTX;
        noteDec = BEYOND_CTX;
      } else {
        if (rec.curve.length < 2) notePre = NO_CURVE;
        else {
          ttft = prefillMs(rec.curve, p.task === "chat" ? p.depth : 0, p.prompt) / 1000;
          estPre = reach.prefill ? null
            : "past the measured " + depthLabel(rec.pre_max) + " tokens";
        }

        if (!rec.ladder.length) noteDec = NO_LADDER;
        else {
          reply = decodeSeconds(rec.ladder, deep, p.out);
          rate = p.out > 0 && reply > 0 ? p.out / reply : tpsAt(rec.ladder, deep);
          /* A one-point ladder already says the rate is held at its single fill; past
           * the deepest of several, the same holding needs saying for itself. */
          estDec = rec.ladder.length < 2 ? FLAT : (reach.decode ? null : HELD);
        }
      }

      /* Background extraction's done-after includes getting the model ready — the
       * run starts from nothing. That is the weights read and the context
       * allocated, and not the harness's warm pass, which is mostly a prefill of
       * the task's own prompt and is a cost no deployment pays. A lane that scored
       * no job measured no load spans and says so rather than reading as if the
       * load were free. */
      var load = background ? (rec.load_s || 0) : 0;
      var noLoad = background && rec.load_s === null
        ? "model load not measured here" : null;
      if (ttft !== null && reply !== null) total = load + ttft + reply;

      var est = { ttft: estPre, tps: estDec,
                  total: join([estPre, estDec, noLoad]) };
      var value = { ttft: ttft, tps: background ? reply : rate, total: total };
      var note = { ttft: notePre, tps: noteDec, total: notePre || noteDec };
      /* What a row carries is what the chart draws or filters on, plus the one thing
       * a mark cannot say: why its number is marked or missing. The configuration
       * itself is in the controls row above the chart. */
      var shared = {
        lane: rec.lane, dev_class: rec.dev_class, model: rec.model,
        quant: rec.quant, backend: rec.backend, rank: rec.rank,
      };

      /* A reason belongs to the lane, not to a column: "beyond this model's context"
       * is the same sentence about the same model in all three. So each distinct reason
       * is printed once, in the first column that has it, and the columns after it
       * leave the row blank — hover still answers in every column (`reason`). */
      var told = [];

      pack.metrics.forEach(function (metric) {
        var isRate = metric === "tps" && !background;
        var v = round(value[metric], isRate ? 1 : 2);
        var measured = p.measured && rec.measured ? rec.measured[metric] : null;
        var dot = measured === undefined || measured === null ? null : measured;
        /* The human-readable value, printed at the bar's end and repeated on
         * hover — with its ~ when it is an estimate. */
        var label = v === null ? null
          : (est[metric] ? "~" : "")
            + (isRate ? String(Math.round(v)) : fmtSeconds(v));
        var reason = v === null ? note[metric] : null;
        if (reason !== null && told.indexOf(reason) >= 0) reason = null;
        else if (reason !== null) told.push(reason);
        rows.push(Object.assign({}, shared, {
          metric: metric, value: v, label: label, value_label: dash(label),
          /* `note` is the text the chart prints, once per lane. `why` is the same fact
           * for the column hover asks about — whichever applies there, since a column
           * either carries a number resting on something or carries no number at all.
           * `measured` never draws here — it is what accuracyRows grades against. */
          note: reason, est: !!est[metric],
          why: dash(est[metric] || note[metric]),
          measured: dot,
        }));
      });
    });
    return rows;
  }

  /* How the calculator scores against the one thing it can be checked on: the
   * validation job's own shape, priced by the same arithmetic as every other
   * task, read against the numbers the job actually measured. Generation is not
   * graded on its own — where a sweep ran thin the job's rate is part of the
   * lane's ladder (see the pack), so only time to first token (the prompt fit)
   * and the whole task are independent enough to score. Phase strings match
   * site.py's ACCURACY_PHASES — keep the two sides in step. */
  function accuracyRows(pack) {
    var task = null;
    (pack.tasks || []).forEach(function (t) { if (t.measured) task = t; });
    if (!task) return [];
    var p = { task: task.key, depth: task.depth, prompt: task.prompt,
              out: task.out, measured: true };
    var phases = { ttft: "time to first token", total: "whole task" };
    var rows = [];
    taskRows(pack, p).forEach(function (r) {
      if (!phases[r.metric] || r.value === null || r.measured === null) return;
      var err = (r.value - r.measured) / r.measured * 100;
      rows.push({
        lane: r.lane, dev_class: r.dev_class, model: r.model, quant: r.quant,
        backend: r.backend, rank: r.rank, phase: phases[r.metric],
        err_pct: Math.round(err * 10) / 10,
        err_label: (err < 0 ? "−" : "+") + Math.abs(err).toFixed(1) + "%",
        pred_label: r.value_label, meas_label: fmtSeconds(r.measured),
      });
    });
    return rows;
  }

  return {
    prefillMs: prefillMs, tpsAt: tpsAt,
    decodeSeconds: decodeSeconds, envelope: envelope, fmtSeconds: fmtSeconds,
    depthLabel: depthLabel, taskRows: taskRows, accuracyRows: accuracyRows,
    words: words,
  };
})();
