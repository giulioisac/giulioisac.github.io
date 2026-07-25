/* Figures for gene-flow.html.
   Fig 1: relaxation of allele frequencies under asymmetric migration, and the
   F2 statistics it induces. Fig 3: the same constrained regression run on raw
   frequencies vs on bias-corrected F2 statistics, as sampling depth and the
   number of loci are varied. No dependencies. */
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

  /* ------------------------------------------------------------------ Fig 1
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

  /* ------------------------------------------------------------------ Fig 3
     Three demes evolving under a known importation matrix, observed through a
     finite sequencing sample. The same constrained regression is run twice:
     once on the observed frequencies, once on bias-corrected F2 statistics. */
  (function () {
    var svg = document.getElementById("estSvg");
    if (!svg) return;

    var N = 3, TT = 14, NE = 400;
    var ATRUE = [
      [0.90, 0.07, 0.03],
      [0.05, 0.93, 0.02],
      [0.02, 0.03, 0.95]
    ];
    var PAIRS = [];
    for (var a = 0; a < N; a++) for (var b = 0; b < N; b++) if (a !== b) PAIRS.push([a, b]);

    var TOPY = 40, BOTY = 250, YMAX = 1 / N;
    var nLoci = 60, nSeq = 30, hidden = null;

    function xg(g) { return ML + (g + 0.5) / PAIRS.length * PW; }
    function py(v) { return BOTY - Math.max(0, Math.min(YMAX, v)) / YMAX * (BOTY - TOPY); }

    /* static chrome */
    [0.1, 0.2, 0.3].forEach(function (v) {
      el("line", { class: "grid", x1: ML, x2: W - MR, y1: py(v), y2: py(v), "stroke-width": 1 }, svg);
    });
    el("line", { class: "axis", x1: ML, x2: W - MR, y1: BOTY, y2: BOTY, "stroke-width": 1 }, svg);
    [0, 0.1, 0.2, 0.3].forEach(function (v) {
      el("text", { class: "tick-label", x: ML - 8, y: py(v) + 4, "text-anchor": "end" }, svg).textContent = v.toFixed(1);
    });
    el("line", { class: "xline", x1: ML, x2: W - MR, y1: py(1 / N), y2: py(1 / N), "stroke-width": 1, "stroke-dasharray": "5 4" }, svg);
    el("text", { class: "anno", x: W - MR, y: py(1 / N) - 7, "text-anchor": "end" }, svg).textContent = "uniform matrix, 1/n = 0.33";
    el("text", { class: "pcap", x: ML, y: 14 }, svg).textContent = "IMPORTATION RATE";
    var SUB = ["₁", "₂", "₃"];
    PAIRS.forEach(function (p, g) {
      el("line", { class: "grid", x1: xg(g), x2: xg(g), y1: TOPY, y2: BOTY, "stroke-width": 1 }, svg);
      el("text", { class: "tick-label", x: xg(g), y: BOTY + 18, "text-anchor": "middle" }, svg)
        .textContent = "A" + SUB[p[0]] + SUB[p[1]];
    });
    el("text", { class: "anno", x: ML, y: 284 }, svg).textContent =
      "each column: one off-diagonal entry, the fraction population i imports from j per step";

    var truthMarks = [], lsDots = [], f2Dots = [];
    PAIRS.forEach(function (p, g) {
      truthMarks.push(el("line", { class: "curve ref", x1: xg(g) - 15, x2: xg(g) + 15, y1: py(ATRUE[p[0]][p[1]]), y2: py(ATRUE[p[0]][p[1]]) }, svg));
      lsDots.push(el("circle", { class: "dot", cx: xg(g) - 9, cy: py(0), r: 4.5 }, svg));
      f2Dots.push(el("circle", { class: "dot acc", cx: xg(g) + 9, cy: py(0), r: 4.5 }, svg));
    });

    /* ---- simulation ---- */
    function simulate(L) {
      var x = [], t, i, mu, x0 = [];
      for (i = 0; i < N; i++) x0.push(new Float64Array(L));
      for (mu = 0; mu < L; mu++) {
        var p = 0.15 + 0.7 * Math.random();
        for (i = 0; i < N; i++) x0[i][mu] = clamp01(p + 0.25 * randn());
      }
      x.push(x0);
      for (t = 0; t < TT; t++) {
        var prev = x[t], next = [];
        for (i = 0; i < N; i++) next.push(new Float64Array(L));
        for (mu = 0; mu < L; mu++) {
          for (i = 0; i < N; i++) {
            var m = 0;
            for (var j = 0; j < N; j++) m += ATRUE[i][j] * prev[j][mu];
            next[i][mu] = clamp01(m + Math.sqrt(m * (1 - m) / NE) * randn());
          }
        }
        x.push(next);
      }
      return x;
    }
    /* Draw S sequences. Exact Bernoulli when cheap, matched-variance normal
       rounded onto the same k/S grid when not. */
    function sampleFreq(p, S) {
      if (S <= 100) {
        var k = 0;
        for (var s = 0; s < S; s++) if (Math.random() < p) k++;
        return k / S;
      }
      var v = p + Math.sqrt(p * (1 - p) / S) * randn();
      return Math.round(Math.max(0, Math.min(1, v)) * S) / S;
    }
    function observe(x, L, S) {
      var o = [], t, i, mu;
      for (t = 0; t <= TT; t++) {
        var row = [];
        for (i = 0; i < N; i++) {
          var f = new Float64Array(L);
          for (mu = 0; mu < L; mu++) f[mu] = sampleFreq(x[t][i][mu], S);
          row.push(f);
        }
        o.push(row);
      }
      return o;
    }

    /* ---- constrained least squares on the simplex ---- */
    function projSimplex(v) {
      var n = v.length, u = Array.prototype.slice.call(v).sort(function (p, q) { return q - p; });
      var css = 0, theta = 0, i;
      for (i = 0; i < n; i++) {
        css += u[i];
        if (u[i] - (css - 1) / (i + 1) > 0) theta = (css - 1) / (i + 1);
      }
      var out = new Float64Array(n);
      for (i = 0; i < n; i++) out[i] = Math.max(v[i] - theta, 0);
      return out;
    }
    function solveRow(G, b) {
      var A = new Float64Array(N), g = new Float64Array(N), i, j, it, tr = 0;
      for (i = 0; i < N; i++) { A[i] = 1 / N; tr += G[i][i]; }
      var step = 1 / (2 * (tr + 1e-12));
      for (it = 0; it < 3000; it++) {
        for (i = 0; i < N; i++) {
          var s = 0;
          for (j = 0; j < N; j++) s += G[i][j] * A[j];
          g[i] = A[i] - step * 2 * (s - b[i]);
        }
        A = projSimplex(g);
      }
      return A;
    }
    function zeros2() { var m = []; for (var i = 0; i < N; i++) m.push(new Float64Array(N)); return m; }

    function estLS(o, L) {
      var A = [], i, j, k, t, mu;
      for (i = 0; i < N; i++) {
        var G = zeros2(), b = new Float64Array(N);
        for (t = 0; t < TT; t++) for (mu = 0; mu < L; mu++) {
          for (j = 0; j < N; j++) {
            b[j] += o[t][j][mu] * o[t + 1][i][mu];
            for (k = 0; k < N; k++) G[j][k] += o[t][j][mu] * o[t][k][mu];
          }
        }
        A.push(solveRow(G, b));
      }
      return A;
    }
    function estF2(o, L, S) {
      var c = 1 / Math.max(1, S - 1), F = [], Fp = [], t, i, j, k, mu;
      for (t = 0; t <= TT; t++) {
        var Ft = zeros2();
        for (j = 0; j < N; j++) for (k = 0; k < N; k++) {
          var s = 0;
          for (mu = 0; mu < L; mu++) {
            var u = o[t][j][mu], v = o[t][k][mu];
            s += (u - v) * (u - v) - (j === k ? 0 : c * (u * (1 - u) + v * (1 - v)));
          }
          Ft[j][k] = s / L;
        }
        F.push(Ft);
      }
      for (t = 0; t < TT; t++) {
        var Fq = zeros2();
        for (i = 0; i < N; i++) for (k = 0; k < N; k++) {
          var s2 = 0;
          for (mu = 0; mu < L; mu++) {
            var p = o[t + 1][i][mu], q = o[t][k][mu];
            s2 += (p - q) * (p - q) - c * (p * (1 - p) + q * (1 - q));
          }
          Fq[i][k] = s2 / L;
        }
        Fp.push(Fq);
      }
      var A = [];
      for (i = 0; i < N; i++) {
        var G = zeros2(), b = new Float64Array(N);
        for (t = 0; t < TT; t++) for (k = 0; k < N; k++) {
          if (k === i) continue;
          var y = Fp[t][i][k] - Fp[t][i][i], d = new Float64Array(N);
          for (j = 0; j < N; j++) d[j] = F[t][j][k] - F[t][j][i];
          for (j = 0; j < N; j++) {
            b[j] += d[j] * y;
            for (var kk = 0; kk < N; kk++) G[j][kk] += d[j] * d[kk];
          }
        }
        A.push(solveRow(G, b));
      }
      return A;
    }
    function rmse(A) {
      var s = 0;
      PAIRS.forEach(function (p) {
        var e = A[p[0]][p[1]] - ATRUE[p[0]][p[1]];
        s += e * e;
      });
      return Math.sqrt(s / PAIRS.length);
    }

    var readEl = document.getElementById("estRead"), fitEl = document.getElementById("estFit");
    var sEl = document.getElementById("estS"), sV = document.getElementById("estSV");
    var lEl = document.getElementById("estL"), lV = document.getElementById("estLV");

    function draw() {
      var o = observe(hidden, nLoci, nSeq);
      var Als = estLS(o, nLoci), Af2 = estF2(o, nLoci, nSeq);
      PAIRS.forEach(function (p, g) {
        lsDots[g].setAttribute("cy", py(Als[p[0]][p[1]]));
        f2Dots[g].setAttribute("cy", py(Af2[p[0]][p[1]]));
      });
      readEl.textContent = "RMSE · direct " + rmse(Als).toFixed(3) + " · F₂ " + rmse(Af2).toFixed(3);
      fitEl.textContent = "3 demes · " + TT + " steps · Nₑ " + NE;
    }
    var pending = false;
    function schedule() {
      if (pending) return;
      pending = true;
      requestAnimationFrame(function () { pending = false; draw(); });
    }

    sEl.addEventListener("input", function () {
      nSeq = +sEl.value; sV.textContent = String(nSeq); schedule();
    });
    lEl.addEventListener("input", function () {
      nLoci = +lEl.value; lV.textContent = String(nLoci);
      hidden = simulate(nLoci); schedule();
    });
    document.getElementById("estResample").addEventListener("click", function () {
      hidden = simulate(nLoci); schedule();
    });

    sV.textContent = String(nSeq); lV.textContent = String(nLoci);
    hidden = simulate(nLoci);
    draw();
  })();

  /* ------------------------------------------------------------------ Kalman fig
     One allele frequency tracked over time. A forward Kalman filter separates
     genetic drift (variance x(1-x)/Ne per step) from binomial sampling noise
     (variance x(1-x)/S per observation). Sliders control each independently. */
  (function () {
    var svg = document.getElementById("kalmanSvg");
    if (!svg) return;

    var TT = 30, TOP = 22, BOT = 200;
    var Ne = 500, S = 20;

    function pxt(t) { return ML + t / TT * PW; }
    function pyv(v) { return BOT - Math.max(0, Math.min(1, v)) * (BOT - TOP); }

    /* static chrome */
    [0.25, 0.5, 0.75].forEach(function (v) {
      el("line", { class: "grid", x1: ML, x2: W - MR, y1: pyv(v), y2: pyv(v), "stroke-width": 1 }, svg);
    });
    el("line", { class: "axis", x1: ML, x2: W - MR, y1: BOT, y2: BOT, "stroke-width": 1 }, svg);
    el("text", { class: "pcap", "text-anchor": "middle", transform: "rotate(-90)",
      x: String(-Math.round((TOP + BOT) / 2)), y: "10" }, svg).textContent = "FREQUENCY";
    [[0, "0"], [0.25, ".25"], [0.5, ".5"], [0.75, ".75"], [1, "1"]].forEach(function (t) {
      el("text", { class: "tick-label", x: ML - 8, y: pyv(t[0]) + 4, "text-anchor": "end" }, svg).textContent = t[1];
    });
    [0, 10, 20, 30].forEach(function (t) {
      el("line", { class: "axis", x1: pxt(t), x2: pxt(t), y1: BOT, y2: BOT + 4, "stroke-width": 1 }, svg);
      el("text", { class: "tick-label", x: pxt(t), y: BOT + 16, "text-anchor": "middle" }, svg).textContent = String(t);
    });
    el("text", { class: "anno", x: ML + PW / 2, y: BOT + 30, "text-anchor": "middle" }, svg).textContent = "time step";

    var defs = el("defs", {}, svg);
    el("rect", { x: ML, y: TOP - 4, width: PW, height: BOT - TOP + 4 },
      el("clipPath", { id: "kalmanClip" }, defs));
    var g = el("g", { "clip-path": "url(#kalmanClip)" }, svg);
    var bandEl = el("path", { class: "wash acc", style: "opacity:.18" }, g);
    var trueEl = el("path", { class: "curve ref" }, g);
    var filtEl = el("path", { class: "curve acc" }, g);
    var obsEl  = el("g", {}, g);

    var trueTraj = null, obsArr = null;

    function simulate() {
      /* generate true trajectory at a fixed Ne_true independent of the slider */
      var NE_TRUE = 400;
      var x = [clamp01(0.25 + 0.5 * Math.random())];
      for (var t = 1; t <= TT; t++) {
        var prev = x[t - 1];
        x.push(clamp01(prev + Math.sqrt(prev * (1 - prev) / NE_TRUE) * randn()));
      }
      return x;
    }

    function observe(x) {
      var y = [];
      for (var t = 0; t <= TT; t++) {
        var k = 0;
        for (var s = 0; s < S; s++) if (Math.random() < x[t]) k++;
        y.push(k / S);
      }
      return y;
    }

    function kalmanAndDraw() {
      var y = obsArr, mu = [y[0]], P = [Math.max(y[0] * (1 - y[0]) / S, 1e-4)];
      for (var t = 1; t <= TT; t++) {
        var pm = mu[t - 1], pp = P[t - 1];
        var Q  = Math.max(pm * (1 - pm) / Ne, 1e-6);
        var Pp = pp + Q;
        var R  = Math.max(y[t] * (1 - y[t]) / S, 1e-4);
        var K  = Pp / (Pp + R);
        mu.push(clamp01(pm + K * (y[t] - pm)));
        P.push((1 - K) * Pp);
      }

      /* true trajectory */
      var d = [];
      for (var t = 0; t <= TT; t++) d.push(pxt(t).toFixed(1) + "," + pyv(trueTraj[t]).toFixed(1));
      trueEl.setAttribute("d", "M" + d.join(" L"));

      /* filter mean */
      d = [];
      for (var t = 0; t <= TT; t++) d.push(pxt(t).toFixed(1) + "," + pyv(mu[t]).toFixed(1));
      filtEl.setAttribute("d", "M" + d.join(" L"));

      /* ±1σ band */
      var up = [], dn = [];
      for (var t = 0; t <= TT; t++) {
        var sig = Math.sqrt(P[t]);
        up.push(pxt(t).toFixed(1) + "," + pyv(Math.min(1, mu[t] + sig)).toFixed(1));
        dn.push(pxt(t).toFixed(1) + "," + pyv(Math.max(0, mu[t] - sig)).toFixed(1));
      }
      bandEl.setAttribute("d", "M" + up.join(" L") + " L" + dn.reverse().join(" L") + " Z");

      /* observation dots */
      while (obsEl.firstChild) obsEl.removeChild(obsEl.firstChild);
      for (var t = 0; t <= TT; t++) {
        el("circle", { class: "dot ref", cx: pxt(t).toFixed(1), cy: pyv(y[t]).toFixed(1),
          r: "3.5", "stroke-width": "1.5" }, obsEl);
      }
    }

    function resample() { trueTraj = simulate(); obsArr = observe(trueTraj); kalmanAndDraw(); }

    var neEl = document.getElementById("kalmanNe"), neV = document.getElementById("kalmanNeV");
    var sEl  = document.getElementById("kalmanS"),  sV  = document.getElementById("kalmanSV");
    neEl.addEventListener("input", function () {
      Ne = +neEl.value; neV.textContent = String(Ne);
      kalmanAndDraw();  /* same trajectory and observations, new filter */
    });
    sEl.addEventListener("input",  function () {
      S = +sEl.value; sV.textContent = String(S);
      obsArr = observe(trueTraj);  /* re-observe with new S, then re-filter */
      kalmanAndDraw();
    });
    document.getElementById("kalmanResample").addEventListener("click", resample);
    neV.textContent = String(Ne); sV.textContent = String(S);
    resample();
  })();
})();
