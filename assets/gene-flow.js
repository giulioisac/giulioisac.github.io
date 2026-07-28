/* Figures for gene-flow.html. Fig 1 is a static diagram and needs no script.
   Fig 2: relaxation of allele frequencies under asymmetric migration, and the
   F2 statistics it induces. No dependencies. */
(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";
  var W = 704, ML = 44, MR = 16, PW = W - ML - MR;

  function el(name, attrs, parent) {
    var e = document.createElementNS(NS, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function randn() {
    var u = 0, v = 0;
    while (!u) u = Math.random();
    while (!v) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function clamp01(x) { return x < 1e-4 ? 1e-4 : (x > 1 - 1e-4 ? 1 - 1e-4 : x); }

  /* Vertical hairline driven by pointer or arrow keys; the readout in the
     figure head keeps the last values, so hover never gates them. */
  function crosshair(svg, read, o) {
    if (!svg || !read) return null;
    function px(x) { return ML + (x - o.xMin) / (o.xMax - o.xMin) * PW; }
    var lines = o.bands.map(function (b) {
      return el("line", { class: "cross", y1: b[0], y2: b[1], "stroke-width": 1, visibility: "hidden", "pointer-events": "none" }, svg);
    });
    var cx = o.xMin;
    function show(x, hide) {
      cx = Math.max(o.xMin, Math.min(o.xMax, x));
      lines.forEach(function (l) {
        l.setAttribute("x1", px(cx)); l.setAttribute("x2", px(cx));
        l.setAttribute("visibility", hide ? "hidden" : "visible");
      });
      read.textContent = o.fmt(cx);
    }
    function fromEvent(e) {
      var r = svg.getBoundingClientRect();
      show(o.xMin + ((e.clientX - r.left) / r.width * W - ML) / PW * (o.xMax - o.xMin));
    }
    svg.addEventListener("pointermove", fromEvent);
    svg.addEventListener("pointerdown", fromEvent);
    svg.addEventListener("pointerleave", function () {
      lines.forEach(function (l) { l.setAttribute("visibility", "hidden"); });
    });
    svg.addEventListener("keydown", function (e) {
      var step = e.shiftKey ? o.step * 5 : o.step;
      if (e.key === "ArrowLeft") show(cx - step);
      else if (e.key === "ArrowRight") show(cx + step);
      else if (e.key === "Home") show(o.xMin);
      else if (e.key === "End") show(o.xMax);
      else return;
      e.preventDefault();
    });
    show(o.xMin, true);
    return { refresh: function () { show(cx, true); } };
  }

  /* ------------------------------------------------------------------ Fig 2
     Two demes, many neutral loci. Migration pulls frequencies together; the
     time-lagged F2 statistics break the symmetry and expose the direction. */
  (function () {
    var svg = document.getElementById("relaxSvg");
    if (!svg) return;

    var TT = 40, LAG = 4, NL = 240, SHOW = 6, NE = 500, FMAX = 0.45;
    var T1 = 22, B1 = 150, T2 = 186, B2 = 274;
    var mAB = 0.005, mBA = 0.04, seedA = null, seedB = null;

    function px(t) { return ML + t / TT * PW; }
    function pyf(v) { return B1 - v * (B1 - T1); }
    function pyF(v) { return B2 - Math.max(0, Math.min(FMAX, v)) / FMAX * (B2 - T2); }

    /* static chrome */
    el("text", { class: "pcap", x: ML, y: 14 }, svg).textContent = "ALLELE FREQUENCIES";
    [0.5].forEach(function (v) {
      el("line", { class: "grid", x1: ML, x2: W - MR, y1: pyf(v), y2: pyf(v), "stroke-width": 1 }, svg);
    });
    el("line", { class: "axis", x1: ML, x2: W - MR, y1: B1, y2: B1, "stroke-width": 1 }, svg);
    el("text", { class: "pcap", "text-anchor": "middle", transform: "rotate(-90)", x: String(-Math.round((T1 + B1) / 2)), y: "10" }, svg).textContent = "FREQUENCY";
    [[0, "0"], [0.5, "0.5"], [1, "1"]].forEach(function (t) {
      el("text", { class: "tick-label", x: ML - 8, y: pyf(t[0]) + 4, "text-anchor": "end" }, svg).textContent = t[1];
    });
    el("text", { class: "pcap", x: ML, y: 172 }, svg).textContent = "GENETIC DISTANCE";
    [0.15, 0.3].forEach(function (v) {
      el("line", { class: "grid", x1: ML, x2: W - MR, y1: pyF(v), y2: pyF(v), "stroke-width": 1 }, svg);
    });
    el("line", { class: "axis", x1: ML, x2: W - MR, y1: B2, y2: B2, "stroke-width": 1 }, svg);
    el("text", { class: "pcap", "text-anchor": "middle", transform: "rotate(-90)", x: String(-Math.round((T2 + B2) / 2)), y: "10" }, svg).textContent = "F\u2082 DISTANCE";
    [[0, "0"], [0.15, "0.15"], [0.3, "0.30"]].forEach(function (t) {
      el("text", { class: "tick-label", x: ML - 8, y: pyF(t[0]) + 4, "text-anchor": "end" }, svg).textContent = t[1];
    });
    [0, 10, 20, 30, 40].forEach(function (t) {
      el("line", { class: "axis", x1: px(t), x2: px(t), y1: B2, y2: B2 + 4, "stroke-width": 1 }, svg);
      el("text", { class: "tick-label", x: px(t), y: B2 + 18, "text-anchor": "middle" }, svg).textContent = String(t);
    });
    el("text", { class: "anno", x: ML + PW / 2, y: B2 + 34, "text-anchor": "middle" }, svg).textContent = "generations";

    var defs = el("defs", {}, svg);
    el("rect", { x: ML, y: T1 - 6, width: PW, height: B1 - T1 + 6 }, el("clipPath", { id: "relaxClipA" }, defs));
    el("rect", { x: ML, y: T2 - 6, width: PW, height: B2 - T2 + 6 }, el("clipPath", { id: "relaxClipB" }, defs));
    var gTraj = el("g", { "clip-path": "url(#relaxClipA)" }, svg);
    var gStat = el("g", { "clip-path": "url(#relaxClipB)" }, svg);
    var trajA = [], trajB = [], i;
    for (i = 0; i < SHOW; i++) trajA.push(el("path", { class: "curve ref thin" }, gTraj));
    for (i = 0; i < SHOW; i++) trajB.push(el("path", { class: "curve acc thin" }, gTraj));
    var pF = el("path", { class: "curve ink" }, gStat);
    var pFab = el("path", { class: "curve ink dash" }, gStat);
    var pFba = el("path", { class: "curve acc" }, gStat);
    el("text", { class: "plabel", x: px(1), y: pyf(0.86) }, svg).textContent = "A";
    el("text", { class: "plabel", x: px(1), y: pyf(0.14) + 10 }, svg).textContent = "B";

    function reseed() {
      seedA = new Float64Array(NL); seedB = new Float64Array(NL);
      for (var m = 0; m < NL; m++) {
        seedA[m] = clamp01(0.8 + 0.06 * randn());
        seedB[m] = clamp01(0.2 + 0.06 * randn());
      }
    }
    var F = [], Fab = [], Fba = [];

    function run() {
      var a = new Float64Array(seedA), b = new Float64Array(seedB), m, t;
      var histA = [new Float64Array(a)], histB = [new Float64Array(b)];
      for (t = 0; t < TT; t++) {
        var na = new Float64Array(NL), nb = new Float64Array(NL);
        for (m = 0; m < NL; m++) {
          var ma = (1 - mAB) * a[m] + mAB * b[m];
          var mb = (1 - mBA) * b[m] + mBA * a[m];
          na[m] = clamp01(ma + Math.sqrt(ma * (1 - ma) / NE) * randn());
          nb[m] = clamp01(mb + Math.sqrt(mb * (1 - mb) / NE) * randn());
        }
        a = na; b = nb;
        histA.push(new Float64Array(a)); histB.push(new Float64Array(b));
      }
      F = []; Fab = []; Fba = [];
      for (t = 0; t <= TT; t++) {
        var s = 0;
        for (m = 0; m < NL; m++) { var d = histA[t][m] - histB[t][m]; s += d * d; }
        F.push(s / NL);
      }
      for (t = 0; t + LAG <= TT; t++) {
        var s1 = 0, s2 = 0;
        for (m = 0; m < NL; m++) {
          var d1 = histA[t + LAG][m] - histB[t][m];
          var d2 = histB[t + LAG][m] - histA[t][m];
          s1 += d1 * d1; s2 += d2 * d2;
        }
        Fab.push(s1 / NL); Fba.push(s2 / NL);
      }
      return { A: histA, B: histB };
    }

    function pathFrom(vals, ymap) {
      var d = [];
      for (var t = 0; t < vals.length; t++) d.push(px(t).toFixed(1) + "," + ymap(vals[t]).toFixed(1));
      return "M" + d.join(" L");
    }
    function seriesOf(hist, mu) {
      var v = [];
      for (var t = 0; t < hist.length; t++) v.push(hist[t][mu]);
      return v;
    }

    var cross = crosshair(svg, document.getElementById("relaxRead"), {
      xMin: 0, xMax: TT, step: 1, bands: [[T1 - 6, B1], [T2 - 6, B2]],
      fmt: function (x) {
        if (!F.length) return "";
        var t = Math.round(x), tl = Math.min(t, TT - LAG);
        return "t " + t + " · F " + F[t].toFixed(3) +
          " · F′AB " + Fab[tl].toFixed(3) + " · F′BA " + Fba[tl].toFixed(3);
      }
    });

    function draw() {
      var h = run(), k;
      for (k = 0; k < SHOW; k++) {
        trajA[k].setAttribute("d", pathFrom(seriesOf(h.A, k), pyf));
        trajB[k].setAttribute("d", pathFrom(seriesOf(h.B, k), pyf));
      }
      pF.setAttribute("d", pathFrom(F, pyF));
      pFab.setAttribute("d", pathFrom(Fab, pyF));
      pFba.setAttribute("d", pathFrom(Fba, pyF));
      cross.refresh();
    }

    var elAB = document.getElementById("relaxAB"), vAB = document.getElementById("relaxABV");
    var elBA = document.getElementById("relaxBA"), vBA = document.getElementById("relaxBAV");
    elAB.addEventListener("input", function () {
      mAB = +elAB.value; vAB.textContent = mAB.toFixed(3); draw();
    });
    elBA.addEventListener("input", function () {
      mBA = +elBA.value; vBA.textContent = mBA.toFixed(3); draw();
    });
    document.getElementById("relaxResample").addEventListener("click", function () {
      reseed(); draw();
    });
    vAB.textContent = mAB.toFixed(3); vBA.textContent = mBA.toFixed(3);
    reseed(); draw();
  })();
})();
