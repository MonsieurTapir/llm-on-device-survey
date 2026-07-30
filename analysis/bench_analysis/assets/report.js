/* Mount the vega islands and wire the copy buttons. One control row drives every
 * island: each chart declares the signals it accepts, so a select pushes its value
 * into whichever views know that name. Dark mode re-tints the palette from the
 * light→dark map, then re-embeds on scheme change. */
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

  function retint(node) {
    if (Array.isArray(node)) { node.forEach(retint); return; }
    if (node && typeof node === "object") {
      Object.keys(node).forEach(function (k) {
        var v = node[k];
        if (typeof v === "string" && darkMap[v]) node[k] = darkMap[v];
        else retint(v);
      });
    }
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

  /* Push the current selection into one island; signals it lacks are skipped. */
  function scope(entry) {
    var touched = false;
    selects.forEach(function (sel) {
      if (entry.signals.indexOf(sel.dataset.signal) < 0) return;
      entry.view.signal(sel.dataset.signal, sel.value);
      touched = true;
    });
    if (touched) entry.view.runAsync();
  }

  function mountAll() {
    mounted = [];
    document.querySelectorAll(".island").forEach(function (el) {
      var raw = document.querySelector(
        'script[data-island="' + el.dataset.spec + '"]');
      if (!raw) return;
      var spec = ink(JSON.parse(raw.textContent));
      if (dark.matches) retint(spec);
      var signals = (spec.params || []).map(function (p) { return p.name; });
      vegaEmbed(el, spec, { actions: false }).then(function (res) {
        var entry = { view: res.view, signals: signals };
        mounted.push(entry);
        scope(entry);
      });
    });
  }

  selects.forEach(function (sel) {
    sel.addEventListener("change", function () { mounted.forEach(scope); });
  });

  mountAll();
  dark.addEventListener("change", mountAll);
})();
