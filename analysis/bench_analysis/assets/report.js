/* Mount the vega islands and wire the copy buttons. Dark mode re-tints the
 * lane palette from the light→dark map, then re-embeds on scheme change. */
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

  var darkMap = JSON.parse(document.getElementById("lane-dark-map").textContent);
  var dark = window.matchMedia("(prefers-color-scheme: dark)");

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
    var axis = { labelColor: "currentColor", titleColor: "currentColor",
                 domainColor: "currentColor", tickColor: "currentColor" };
    spec.config = spec.config || {};
    spec.config.axis = Object.assign({}, spec.config.axis, axis);
    spec.config.legend = { labelColor: "currentColor", titleColor: "currentColor" };
    spec.config.title = { color: "currentColor" };
    spec.config.header = { labelColor: "currentColor", titleColor: "currentColor" };
    spec.config.text = { color: "currentColor" };
    spec.config.background = "transparent";
    return spec;
  }

  function mountAll() {
    document.querySelectorAll(".island").forEach(function (el) {
      var raw = document.querySelector(
        'script[data-island="' + el.dataset.spec + '"]');
      if (!raw) return;
      var spec = ink(JSON.parse(raw.textContent));
      if (dark.matches) retint(spec);
      vegaEmbed(el, spec, { actions: false });
    });
  }

  mountAll();
  dark.addEventListener("change", mountAll);
})();
