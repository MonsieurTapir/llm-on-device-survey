/* Mount the vega islands and wire the copy buttons. One control row drives every
 * island: each chart declares the signals it accepts, so a select pushes its value
 * into whichever views know that name. Dark mode re-tints the palette from the
 * light→dark map, then re-embeds on scheme change.
 *
 * The task island is the one chart with no data of its own: its own control row
 * feeds `taskMath.taskRows` (assets/tasks.js) and the rows are pushed in as a vega
 * changeset, so a preset change costs no reload and no rebuild. Those rows are
 * re-pushed after every embed, which is what carries a hand-typed configuration
 * through a light/dark switch. */
(function () {
  "use strict";

  document.querySelectorAll("button.copy").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var text = document.getElementById(btn.dataset.copy).textContent;
      navigator.clipboard.writeText(text).then(function () {
        var old = btn.textContent;
        btn.textContent = "copied!";
        setTimeout(function () { btn.textContent = old; }, 1200);
      });
    });
  });

  /* The appendix tab strip: one panel visible, the rest `hidden`. */
  document.querySelectorAll(".tabs").forEach(function (strip) {
    var tabs = Array.prototype.slice.call(strip.querySelectorAll("button.tab"));
    strip.addEventListener("click", function (event) {
      var picked = event.target.closest("button.tab");
      if (!picked) return;
      tabs.forEach(function (tab) {
        var on = tab === picked;
        tab.classList.toggle("on", on);
        tab.setAttribute("aria-selected", on ? "true" : "false");
        document.getElementById(tab.dataset.panel).hidden = !on;
      });
    });
  });

  var darkMap = JSON.parse(document.getElementById("lane-dark-map").textContent);
  var dark = window.matchMedia("(prefers-color-scheme: dark)");
  var selects = Array.prototype.slice.call(
    document.querySelectorAll("#controls select"));
  var mounted = [];

  /* Every light palette color in a spec, swapped in place for its dark step. The
   * colors that matter sit in scale `range` arrays, so array indices are rewritten
   * exactly like object keys — `Object.keys` covers both. */
  function retint(node) {
    if (!node || typeof node !== "object") return;
    Object.keys(node).forEach(function (k) {
      var v = node[k];
      if (typeof v === "string") { if (darkMap[v]) node[k] = darkMap[v]; }
      else retint(v);
    });
  }

  function ink(spec) {
    /* Gridlines are inked too: vega's default #ddd is a shout on the dark
     * surface. One recessive hairline off whatever the surface is. */
    var axis = { labelColor: "currentColor", titleColor: "currentColor",
                 domainColor: "currentColor", tickColor: "currentColor",
                 gridColor: "currentColor", gridOpacity: 0.14 };
    spec.config = spec.config || {};
    spec.config.axis = Object.assign({}, spec.config.axis, axis);
    spec.config.legend = { labelColor: "currentColor", titleColor: "currentColor" };
    spec.config.title = { color: "currentColor", subtitleColor: "currentColor" };
    spec.config.header = { labelColor: "currentColor", titleColor: "currentColor" };
    spec.config.text = { color: "currentColor" };
    spec.config.background = "transparent";
    return spec;
  }

  /* Keep vega's tooltip inside the window.
   *
   * The tooltip is a fixed-position element on the body, so no island's scroll box can
   * clip it — but vega picks one of four spots beside the cursor and, when none of them
   * fits, falls back to one that can hang off the left edge. On a narrow window a
   * wrapped note is wider than either gap beside the cursor, so that fallback is the
   * normal case rather than the corner one. This nudges the box back in without
   * changing its size or which side of the cursor it took. Writing `left` re-enters the
   * observer once, finds nothing left to move, and stops. */
  function clamp(el) {
    new MutationObserver(function () {
      var box = el.getBoundingClientRect();
      var room = document.documentElement.clientWidth - box.width - 4;
      var left = Math.min(Math.max(4, box.left), Math.max(4, room));
      if (Math.abs(left - box.left) > 0.5) el.style.left = left + "px";
    }).observe(el, { attributeFilter: ["style"] });
  }

  /* The element is vega-tooltip's, created when it first has something to show. */
  function clampWhenItExists() {
    var el = document.getElementById("vg-tooltip-element");
    if (el) return clamp(el);
    var watch = new MutationObserver(function () {
      var found = document.getElementById("vg-tooltip-element");
      if (!found) return;
      watch.disconnect();
      clamp(found);
    });
    watch.observe(document.body, { childList: true });
  }

  /* Redraw one island. An island whose data the page inserts (the task grid) was
   * laid out for an empty set, so it needs its size recomputed rather than just its
   * marks redrawn — and it keeps needing that, because how many lanes it holds
   * depends on both the filters and the rows the page last pushed in. */
  function redraw(entry) {
    if (entry.named) entry.view.resize();
    entry.view.runAsync();
  }

  /* Push the current selection into one island; signals it lacks are skipped. */
  function scope(entry) {
    var touched = false;
    selects.forEach(function (sel) {
      if (entry.signals.indexOf(sel.dataset.signal) < 0) return;
      entry.view.signal(sel.dataset.signal, sel.value);
      touched = true;
    });
    if (touched) redraw(entry);
  }

  /* The task grid's column titles belong to the selected task (a chat turn and a
   * background job answer different questions), and its middle column is a rate for
   * the watched tasks but seconds for the background one — where the reading-speed
   * gridlines would be noise. The spec is re-parsed fresh on every mount, so the
   * patch is idempotent: swap each column's axis title, and strip the anchor
   * gridlines when the middle column is not a rate. */
  function patchTaskSpec(spec, task) {
    if (!task || !spec.hconcat) return spec;
    spec.hconcat.forEach(function (col, i) {
      var axis = col.spec.layer[0].encoding.x.axis;
      var titled = task.columns[i];
      axis.title = titled[1] ? [titled[0], titled[1]] : titled[0];
      if (i === 1 && task.mid_unit !== "tps") {
        axis.grid = false;
        delete axis.values;
        delete axis.labelExpr;
      }
    });
    return spec;
  }

  function mount(el) {
    var raw = document.querySelector(
      'script[data-island="' + el.dataset.spec + '"]');
    if (!raw) return null;
    var spec = ink(JSON.parse(raw.textContent));
    if (dark.matches) retint(spec);
    if (el.dataset.spec === "tasks") patchTaskSpec(spec, currentTask());
    var signals = (spec.params || []).map(function (p) { return p.name; });
    var named = spec.data && spec.data.name;
    return vegaEmbed(el, spec, { actions: false }).then(function (res) {
      var entry = { el: el, view: res.view, signals: signals, named: named };
      mounted = mounted.filter(function (m) { return m.el !== el; });
      mounted.push(entry);
      scope(entry);
      return entry;
    });
  }

  function mountAll() {
    mounted = [];
    var pending = [];
    document.querySelectorAll(".island").forEach(function (el) {
      var p = mount(el);
      if (p) pending.push(p);
    });
    /* The task rows outlive the views that draw them: a re-embed (dark mode) gets
     * the current configuration pushed back in, not the task it started on. */
    Promise.all(pending).then(applyTasks);
  }

  selects.forEach(function (sel) {
    sel.addEventListener("change", function () { mounted.forEach(scope); });
  });

  /* ------------------------------------------------------- the task calculator */
  var packEl = document.getElementById("task-pack");
  var pack = packEl ? JSON.parse(packEl.textContent) : null;
  var noteEl = document.getElementById("task-note");
  var presetEl = document.getElementById("t-preset");
  var numbers = { depth: document.getElementById("t-depth"),
                  prompt: document.getElementById("t-prompt"),
                  out: document.getElementById("t-out") };

  function currentTask() {
    if (!pack || !presetEl) return null;
    var found = pack.tasks.filter(function (t) {
      return t.key === presetEl.value;
    });
    return found[0] || null;
  }

  function num(el) {
    var v = parseInt(el.value, 10);
    return isFinite(v) && v > 0 ? v : 0;
  }

  /* The configuration the controls currently describe: the selected task with
   * whatever token counts sit in the inputs. Editing a count edits the task; it
   * never becomes some other task. */
  function taskParams() {
    var chosen = currentTask() || {};
    return { task: chosen.key, depth: num(numbers.depth),
             prompt: num(numbers.prompt), out: num(numbers.out),
             measured: !!chosen.measured };
  }

  /* What each token count is in words, beside the input it belongs to. Tokens are
   * what the cost functions are fitted in and words are what a reader recognises, so
   * the field keeps the token count and the hint carries the translation — the same
   * ¾-word ratio the preset notes are written in (`taskMath.words`). */
  function showWords() {
    Object.keys(numbers).forEach(function (field) {
      var hint = document.querySelector('.words[data-for="t-' + field + '"]');
      if (!hint) return;
      var count = window.taskMath.words(num(numbers[field]));
      hint.textContent = count ? "≈ " + count.toLocaleString() + " words" : "";
    });
  }

  /* Only the inputs the task uses are on show: context-already-there is a chat
   * idea. */
  function showFields(chosen) {
    var fields = (chosen && chosen.fields) || [];
    document.querySelectorAll("#task-controls [data-field]").forEach(function (el) {
      el.classList.toggle("hidden", fields.indexOf(el.dataset.field) < 0);
    });
  }

  function applyTasks() {
    if (!pack || !presetEl) return;
    var chosen = currentTask();
    showFields(chosen);
    showWords();
    noteEl.textContent = (chosen && chosen.note) || "";
    var rows = window.taskMath.taskRows(pack, taskParams());
    mounted.forEach(function (entry) {
      if (entry.named !== "tasks") return;
      entry.view.change("tasks", vega.changeset().remove(function () {
        return true;
      }).insert(rows));
      redraw(entry);
    });
  }

  if (pack && presetEl) {
    presetEl.addEventListener("change", function () {
      var chosen = currentTask();
      if (chosen) {
        Object.keys(numbers).forEach(function (field) {
          numbers[field].value = chosen[field];
        });
      }
      /* A new task brings its own column titles (and, for the background task,
       * a seconds middle column with no reading-speed gridlines), so the island
       * is re-embedded with the patched spec; the rows follow via applyTasks. */
      var island = document.querySelector('.island[data-spec="tasks"]');
      var p = island && mount(island);
      if (p) p.then(applyTasks);
      else applyTasks();
    });
    Object.keys(numbers).forEach(function (field) {
      numbers[field].addEventListener("input", applyTasks);
    });
  }

  mountAll();
  clampWhenItExists();
  dark.addEventListener("change", mountAll);
})();
