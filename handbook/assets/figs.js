/* Animated figures on topic pages (styles in figs.css). A figure is declarative markup:

     <figure class="fx">
       <div class="fx-stage"> ...blocks; any element may carry timing attributes... </div>
       <ol class="fx-say"><li>what frame 0 shows</li><li>what frame 1 shows</li>...</ol>
       <figcaption>Figure: ...</figcaption>
     </figure>

   One <li> per frame; a frame lasts one beat (data-beat on the figure, default 1000 ms),
   or data-hold="2" beats. Timing attributes, in frame numbers counted from 0:
     data-at="k"      shown from frame k on (and emphasised, .is-now, at frame k)
     data-to="k"      hidden from frame k on (with data-at: shown for frames at..k-1)
     data-hot="k 3-5" emphasised (.is-now) at those frames; .is-seen from the first on
   The loop rests on the last frame, then starts again. It pauses while the figure is off
   screen, while the pointer is on its step buttons, while anything in it has keyboard focus
   (a focused step shows its frame and description) and after the pause button. Under
   prefers-reduced-motion it never plays: it shows the last frame and every description,
   and the step buttons still show each frame. Without this script the CSS shows the last
   frame too. Labels are read from the markup, so one script serves every language. */
(function () {
  "use strict";
  var BEAT = 1000;  // ms per frame
  var REST = 3;     // extra beats on the last frame before the loop starts again
  var TEXT = {
    en: { pause: "Pause the animation", play: "Play the animation", step: "Step {i} of {n}: " },
    zh: { pause: "暂停动画", play: "播放动画", step: "第 {i} 步，共 {n} 步：" }
  };

  function frames(spec) {
    var out = [];
    (spec || "").split(/[\s,]+/).forEach(function (part) {
      if (!part) return;
      var m = part.split("-");
      var a = parseInt(m[0], 10), b = m.length > 1 ? parseInt(m[1], 10) : a;
      for (var k = a; k <= b; k++) out.push(k);
    });
    return out;
  }

  function setup(fig) {
    if (fig.getAttribute("data-fx-ready")) return;
    var say = fig.querySelector(".fx-say");
    var lines = say ? Array.prototype.slice.call(say.children) : [];
    var n = lines.length;
    if (!n) return;
    fig.setAttribute("data-fx-ready", "1");

    var lang = (document.documentElement.getAttribute("lang") || "en").slice(0, 2);
    var text = TEXT[lang] || TEXT.en;
    var beat = parseInt(fig.getAttribute("data-beat") || BEAT, 10);
    var motion = window.matchMedia ? window.matchMedia("(prefers-reduced-motion: reduce)") : null;

    var items = Array.prototype.slice.call(fig.querySelectorAll("[data-at], [data-to], [data-hot]")).map(function (el) {
      var hot = frames(el.getAttribute("data-hot"));
      return {
        el: el,
        timed: el.hasAttribute("data-at"),
        at: parseInt(el.getAttribute("data-at") || "0", 10),
        to: el.hasAttribute("data-to") ? parseInt(el.getAttribute("data-to"), 10) : Infinity,
        hot: hot,
        first: hot.length ? Math.min.apply(null, hot) : Infinity
      };
    });

    // children of a sweep pulse one after another
    fig.querySelectorAll(".fx-sweep").forEach(function (sw) {
      Array.prototype.forEach.call(sw.children, function (c, i) { c.style.setProperty("--i", i); });
    });

    // controls: a pause button and one button per frame, next to the description
    var bar = document.createElement("div");
    bar.className = "fx-foot";
    var ctl = document.createElement("div");
    ctl.className = "fx-ctl";
    var play = document.createElement("button");
    play.type = "button";
    play.className = "fx-play";
    ctl.appendChild(play);
    var dots = lines.map(function (li, i) {
      var b = document.createElement("button");
      b.type = "button";
      b.className = "fx-dot";
      b.textContent = String(i + 1);
      b.setAttribute("aria-label", text.step.replace("{i}", i + 1).replace("{n}", n) + li.textContent.trim());
      ctl.appendChild(b);
      return b;
    });
    say.parentNode.insertBefore(bar, say);
    bar.appendChild(ctl);
    bar.appendChild(say);
    fig.classList.add("is-live");

    var t = -1, timer = null, paused = false, pointer = false, focused = false, visible = true;

    function still() { return !!(motion && motion.matches); }

    function render(k) {
      t = k;
      items.forEach(function (o) {
        var on = o.at <= k && k < o.to;
        var now = (o.timed && o.at === k) || o.hot.indexOf(k) >= 0;
        o.el.classList.toggle("is-on", on);
        o.el.classList.toggle("is-now", on && now);
        o.el.classList.toggle("is-seen", k >= o.first);
      });
      lines.forEach(function (li, i) { li.classList.toggle("is-now", i === k); });
      dots.forEach(function (d, i) {
        if (i === k) d.setAttribute("aria-current", "step");
        else d.removeAttribute("aria-current");
      });
      fig.setAttribute("data-frame", String(k));
    }

    function hold(k) {
      var h = parseFloat(lines[k].getAttribute("data-hold") || "1");
      return beat * (isNaN(h) ? 1 : h) + (k === n - 1 ? REST * beat : 0);
    }

    function running() { return !paused && !pointer && !focused && visible && !document.hidden && !still(); }

    function tick() {
      timer = null;
      if (!fig.isConnected || !running()) return;
      render(t >= n - 1 ? 0 : t + 1);
      timer = setTimeout(tick, hold(t));
    }

    function sync() {
      fig.classList.toggle("is-still", still());
      play.hidden = still();
      var p = paused || still();
      play.setAttribute("aria-label", p ? text.play : text.pause);
      play.setAttribute("aria-pressed", p ? "true" : "false");
      play.classList.toggle("is-paused", p);
      if (running()) { if (!timer) timer = setTimeout(tick, hold(Math.max(t, 0))); }
      else if (timer) { clearTimeout(timer); timer = null; }
    }

    // the frame at which an element in the stage first matters
    function frameOf(el) {
      for (; el && el !== fig; el = el.parentElement) {
        if (el.hasAttribute("data-at")) return parseInt(el.getAttribute("data-at"), 10);
        if (el.hasAttribute("data-hot")) return frames(el.getAttribute("data-hot"))[0];
      }
      return -1;
    }

    play.addEventListener("click", function () {
      paused = !paused;
      if (!paused && t === n - 1) render(0);
      sync();
    });
    dots.forEach(function (d, i) {
      d.addEventListener("click", function () { paused = true; render(i); sync(); });
      d.addEventListener("mouseenter", function () { render(i); });
      d.addEventListener("focus", function () { render(i); });
      d.addEventListener("keydown", function (e) {
        var j = { ArrowRight: i + 1, ArrowDown: i + 1, ArrowLeft: i - 1, ArrowUp: i - 1, Home: 0, End: n - 1 }[e.key];
        if (j === undefined) return;
        e.preventDefault();
        dots[Math.max(0, Math.min(n - 1, j))].focus();
      });
    });
    ctl.addEventListener("mouseenter", function () { pointer = true; sync(); });
    ctl.addEventListener("mouseleave", function () { pointer = false; sync(); });
    fig.addEventListener("focusin", function (e) {
      focused = true;
      if (!ctl.contains(e.target)) {
        var k = frameOf(e.target);
        if (k >= 0 && k < n) render(k);
      }
      sync();
    });
    fig.addEventListener("focusout", function (e) {
      if (!fig.contains(e.relatedTarget)) { focused = false; sync(); }
    });
    document.addEventListener("visibilitychange", sync);
    if ("IntersectionObserver" in window) {
      new IntersectionObserver(function (entries) { visible = entries[0].isIntersecting; sync(); }).observe(fig);
    }
    if (motion) {
      var onChange = function () { render(still() ? n - 1 : 0); sync(); };
      if (motion.addEventListener) motion.addEventListener("change", onChange);
      else if (motion.addListener) motion.addListener(onChange);
    }
    render(still() ? n - 1 : 0);
    sync();
  }

  function init() { document.querySelectorAll("figure.fx").forEach(setup); }

  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(init);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
