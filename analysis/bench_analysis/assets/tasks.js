/* The task calculator: the page's only arithmetic.
 *
 * Pure functions over the pack `site._task_pack` embeds — no DOM, no vega, no
 * globals besides the one export. The pack carries, per (lane, model), the sweep's
 * prefill cost function, its decode ladder, and the once-per-launch costs; this
 * turns a task configuration into the three columns the chart draws.
 *
 * Two rules run through all of it. Anything resting on thin evidence is marked as an
 * estimate rather than quietly averaged in — a lane whose ladder holds a single fill,
 * a prefill fit the chunks scattered around, or a depth past the one the sweep
 * reached. And a number is refused only where there is nothing to evaluate: a lane
 * that measured no cost function, or a task that does not fit the model's own trained
 * context. Past the sweep's deepest point the fit is still evaluated — its shape is
 * physical, a per-dispatch term plus a depth-linear one — and the generation rate is
 * held at the slowest fill actually measured, which is the conservative end.
 */
window.taskMath = (function () {
  "use strict";

  /* Decode is integrated in steps this size: the rate falls as the cache fills, so
   * a long reply cannot be priced at its starting rate. Small enough that the error
   * against a continuous integral is far under the spread of the measurement. */
  var STEP = 32;

  /* A fit is only trusted this far. r² and the worst residual come straight from
   * the harness; the prefill pass runs once by design, so scatter around the line
   * is the only evidence a chunk was disturbed rather than real. */
  var MIN_R2 = 0.98;
  var MAX_RESID_PCT = 5;

  /* The two refusals and the marks. The refusals are facts about the model and the
   * lane — a task longer than the context the model was trained on, a lane whose sweep
   * produced no cost function to evaluate. The marks say how far the evidence reached.
   * All of them are labels, not sentences: what a half-lit bar means is explained once,
   * in the section's own copy, and a hover is no place to read a paragraph. */
  var BEYOND_CTX = "beyond this model's context";
  var NO_FIT = "no prompt cost function measured";
  var NO_LADDER = "no generation rate measured";
  var FLAT = "one measured fill";
  var HELD = "rate held at the deepest fill measured";
  var LOOSE = "loose prompt fit";

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

  /* Milliseconds to read `n` new tokens with `d0` tokens already in the cache.
   *
   * The pass dispatches in chunks of the fit's own width, and the fit prices one
   * full-width chunk at depth d as `b + m·d/1k` — a per-dispatch term plus the
   * attention term over everything already there. So a prompt is the sum over its
   * chunks, each at the depth it starts from, which is why the same `n` costs more
   * deeper in: `d0 = depth` is a chat turn appending to a live cache, `d0 = 0` is a
   * document read into an empty one. A trailing partial chunk is prorated by its
   * token share (checked against the sweep's own ragged chunks: 1,593 ms predicted
   * against 1,587 measured for a 256-token chunk at depth 1,024). */
  function prefillMs(fit, d0, n) {
    if (!fit || !(n > 0)) return null;
    var total = 0, done = 0;
    while (done < n) {
      var chunk = Math.min(fit.w, n - done);
      total += (fit.b + fit.m * (d0 + done) / 1000) * (chunk / fit.w);
      done += chunk;
    }
    return total;
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
      prefill: !!rec.fit && rec.pre_max !== null && deep <= rec.pre_max,
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

  /* Whether the prefill fit is loose enough that its prediction is an estimate. The
   * numbers behind the verdict are two rows down in the same tooltip, so this says
   * which side of the line the fit fell on and nothing more. */
  function looseFit(fit) {
    var loose = (fit.r2 !== null && fit.r2 < MIN_R2)
      || (fit.resid !== null && fit.resid > MAX_RESID_PCT);
    return loose ? LOOSE : null;
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
        /* The one wall: the model was never trained this far, so no cost function of
         * ours has anything to say about it. Both columns carry the same fact, and the
         * caller below prints it once. */
        notePre = BEYOND_CTX;
        noteDec = BEYOND_CTX;
      } else {
        if (!rec.fit) notePre = NO_FIT;
        else {
          ttft = prefillMs(rec.fit, p.task === "chat" ? p.depth : 0, p.prompt) / 1000;
          estPre = join([
            reach.prefill ? null
              : "past the measured " + depthLabel(rec.pre_max) + " tokens",
            looseFit(rec.fit),
          ]);
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
      /* What a row carries is what the chart draws or filters on, plus the two things
       * a mark cannot say: how good the fit under it was, and why its number is marked
       * or missing. The configuration itself is in the controls row above the chart. */
      var shared = {
        lane: rec.lane, dev_class: rec.dev_class, model: rec.model,
        quant: rec.quant, backend: rec.backend, rank: rec.rank,
        r2: dash(rec.fit ? rec.fit.r2 : null),
        resid: dash(rec.fit ? rec.fit.resid : null),
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
        var label = v === null ? null
          : (est[metric] ? "~" : "")
            + (isRate ? String(Math.round(v)) : fmtSeconds(v));
        /* The label rides past the far end of the row's ink, so it never prints over
         * the measured dot when prediction and measurement nearly agree. */
        var anchor = [v, dot].filter(function (n) { return n !== null; });
        var reason = v === null ? note[metric] : null;
        if (reason !== null && told.indexOf(reason) >= 0) reason = null;
        else if (reason !== null) told.push(reason);
        rows.push(Object.assign({}, shared, {
          metric: metric, value: v, label: label, value_label: dash(label),
          label_x: anchor.length ? Math.max.apply(null, anchor) : null,
          /* `note` is the text the chart prints, once per lane. `why` is the same fact
           * for the column hover asks about — whichever applies there, since a column
           * either carries a number resting on something or carries no number at all. */
          note: reason, est: !!est[metric],
          why: dash(est[metric] || note[metric]),
          measured: dot, measured_label: dash(dot),
        }));
      });
    });
    return rows;
  }

  return {
    prefillMs: prefillMs, tpsAt: tpsAt,
    decodeSeconds: decodeSeconds, envelope: envelope, fmtSeconds: fmtSeconds,
    depthLabel: depthLabel, taskRows: taskRows, words: words,
  };
})();
