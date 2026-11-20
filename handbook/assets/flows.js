/* Animates the workflow lanes on the home page (markup in index.md / index.zh.md, styles in
   flows.css). Each lane's steps light up one per second; a lane with data-delay="n" starts n
   steps late. When a step marked .flow-handoff lights up, a pulse runs down its dashed line
   and the lane marked data-target glows. The loop pauses on hover or focus, while the figure is
   off screen, and never runs under prefers-reduced-motion. Labels are read from the markup,
   so one script serves every language. */
(function () {
  "use strict";
  var STEP = 1000;   // ms per step
  var HOP = 420;     // ms for the token to cross the gap
  var REST = 3;      // steps of pause at the end of the loop

  function setup(fig) {
    if (fig.getAttribute("data-flows-ready")) return;
    fig.setAttribute("data-flows-ready", "1");

    var lanes = Array.prototype.slice.call(fig.querySelectorAll(".flow-lane"));
    var steps = lanes.map(function (l) { return Array.prototype.slice.call(l.querySelectorAll(".flow-steps > li")); });
    var delays = lanes.map(function (l) { return parseInt(l.getAttribute("data-delay") || "0", 10); });
    var target = fig.querySelector(".flow-lane[data-target]");
    var length = Math.max.apply(null, steps.map(function (s, i) { return s.length + delays[i]; }));
    var motion = window.matchMedia ? window.matchMedia("(prefers-reduced-motion: reduce)") : null;

    var t = 0, timer = null, hovered = false, focused = false, visible = true;

    function clearAll() {
      fig.querySelectorAll(".is-active, .is-leaving, .is-handing").forEach(function (el) {
        el.classList.remove("is-active", "is-leaving", "is-handing");
      });
      if (target) target.classList.remove("is-receiving");
    }

    function handoff(li) {
      li.classList.add("is-handing");
      setTimeout(function () { if (target) target.classList.add("is-receiving"); }, 750);
      setTimeout(function () {
        li.classList.remove("is-handing");
        if (target) target.classList.remove("is-receiving");
      }, 1500);
    }

    function tick() {
      timer = null;
      if (!fig.isConnected) return;
      if (hovered || focused || !visible || document.hidden) { schedule(); return; }
      if (t >= length + REST) { t = 0; clearAll(); }
      lanes.forEach(function (lane, i) {
        var s = t - delays[i], list = steps[i];
        if (s >= 1 && s <= list.length) {
          var prev = list[s - 1];
          prev.classList.remove("is-active");
          if (s < list.length) {
            prev.classList.add("is-leaving");
            setTimeout(function () { prev.classList.remove("is-leaving"); }, HOP + 60);
          }
        }
        if (s >= 0 && s < list.length) {
          var cur = list[s];
          setTimeout(function () {
            cur.classList.add("is-active");
            if (cur.classList.contains("flow-handoff")) handoff(cur);
          }, s > 0 ? HOP : 0);
        }
      });
      t += 1;
      schedule();
    }

    function schedule() { if (!timer) timer = setTimeout(tick, STEP); }

    function start() {
      if (motion && motion.matches) { stop(); return; }
      if (!timer) { t = 0; clearAll(); timer = setTimeout(tick, 600); }
    }
    function stop() { if (timer) clearTimeout(timer); timer = null; clearAll(); }

    fig.addEventListener("mouseenter", function () { hovered = true; });
    fig.addEventListener("mouseleave", function () { hovered = false; });
    fig.addEventListener("focusin", function () { focused = true; });
    fig.addEventListener("focusout", function (e) { if (!fig.contains(e.relatedTarget)) focused = false; });
    if ("IntersectionObserver" in window) {
      new IntersectionObserver(function (entries) { visible = entries[0].isIntersecting; }).observe(fig);
    }
    if (motion) {
      var onChange = function () { if (motion.matches) stop(); else start(); };
      if (motion.addEventListener) motion.addEventListener("change", onChange);
      else if (motion.addListener) motion.addListener(onChange);
    }
    start();
  }

  function init() { document.querySelectorAll(".flows").forEach(setup); }

  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(init);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
