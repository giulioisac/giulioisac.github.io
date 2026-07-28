/* Figures for me.html: the crosshair readout on the static figure (Fig 1)
   and the live noise-contrastive-estimation demo (Fig 2). No dependencies. */
(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";
  var W = 704, ML = 44, MR = 16, PW = W - ML - MR;

  function gauss(x, m, s) {
    return Math.exp(-((x - m) * (x - m)) / (2 * s * s)) / (s * Math.sqrt(2 * Math.PI));
  }
  function el(name, attrs, parent) {
    var e = document.createElementNS(NS, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }

  /* Crosshair: a vertical hairline per panel, driven by pointer or arrow keys
     (Left/Right step 0.1, Shift 0.5, Home/End). The readout in the figure
     head always shows the values at the last x, so hover never gates them. */
  function crosshair(svg, read, o) {
    if (!svg || !read) return null;
    function px(x) { return ML + (x - o.xMin) / (o.xMax - o.xMin) * PW; }
    var lines = o.bands.map(function (b) {
      return el("line", { class: "cross", y1: b[0], y2: b[1], "stroke-width": 1, visibility: "hidden", "pointer-events": "none" }, svg);
    });
    var cx = 0;
    function show(x, hide) {
      cx = Math.max(o.xMin, Math.min(o.xMax, x));
      lines.forEach(function (l) {
        l.setAttribute("x1", px(cx)); l.setAttribute("x2", px(cx));
        l.setAttribute("visibility", hide ? "hidden" : "visible");
      });
      read.textContent = "x " + cx.toFixed(2) + " · " + o.series.map(function (s) {
        return s.n + " " + s.f(cx).toFixed(s.d == null ? 3 : s.d);
      }).join(" · ");
    }
    function fromEvent(e) {
      var r = svg.getBoundingClientRect();
      var sx = (e.clientX - r.left) / r.width * W;
      show(o.xMin + (sx - ML) / PW * (o.xMax - o.xMin));
    }
    svg.addEventListener("pointermove", fromEvent);
    svg.addEventListener("pointerdown", fromEvent);
    svg.addEventListener("pointerleave", function () {
      lines.forEach(function (l) { l.setAttribute("visibility", "hidden"); });
    });
    svg.addEventListener("keydown", function (e) {
      var step = e.shiftKey ? 0.5 : 0.1;
      if (e.key === "ArrowLeft") show(cx - step);
      else if (e.key === "ArrowRight") show(cx + step);
      else if (e.key === "Home") show(o.xMin);
      else if (e.key === "End") show(o.xMax);
      else return;
      e.preventDefault();
    });
    show(0, true); // seed the readout without drawing the line
    return { refresh: function () { show(cx, true); } };
  }

  /* ---- Fig 1: exponential tilt + the classifier it defines ---- */
  function qClf(x) { return gauss(x, 0, 1.5); }
  function pClf(x) { return gauss(x, 1.1, 0.75); }
  crosshair(document.getElementById("clfSvg"), document.getElementById("clfRead"), {
    xMin: -4.5, xMax: 4.5, bands: [[22, 154], [178, 274]],
    series: [
      { n: "Q", f: qClf },
      { n: "P", f: pClf },
      { n: "P/Q", f: function (x) { return pClf(x) / qClf(x); }, d: 2 },
      { n: "d", f: function (x) { var r = pClf(x) / qClf(x); return r / (1 + r); }, d: 2 }
    ]
  });

  /* ---- Fig 2: live NCE demo ---- */
  var svg = document.getElementById("demoSvg");
  if (!svg) return;

  var XMIN = -5, XMAX = 5, TOP = 20, BASE = 196, YMAX = 0.55;
  var N = 400, NF = 9, RW = 1.3, LR = 0.5, LAM = 1e-3, MAXSTEP = 8000;
  var CENTERS = [];
  for (var ci = 0; ci < 8; ci++) CENTERS.push(-4 + ci * 8 / 7);
  var MIX = [[0.55, -1.3, 0.6], [0.45, 1.4, 0.8]];
  var REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;

  function px(x) { return ML + (x - XMIN) / (XMAX - XMIN) * PW; }
  function py(v) { return BASE - Math.max(0, v) / YMAX * (BASE - TOP); }
  function target(x) {
    return MIX[0][0] * gauss(x, MIX[0][1], MIX[0][2]) + MIX[1][0] * gauss(x, MIX[1][1], MIX[1][2]);
  }

  // static chrome
  [0.25, 0.5].forEach(function (v) {
    el("line", { class: "grid", x1: ML, x2: W - MR, y1: py(v), y2: py(v), "stroke-width": 1 }, svg);
  });
  el("line", { class: "axis", x1: ML, x2: W - MR, y1: BASE, y2: BASE, "stroke-width": 1 }, svg);
  [[0, "0"], [0.25, "0.25"], [0.5, "0.5"]].forEach(function (t) {
    el("text", { class: "tick-label", x: ML - 8, y: py(t[0]) + 4, "text-anchor": "end" }, svg).textContent = t[1];
  });
  [-4, -2, 0, 2, 4].forEach(function (x) {
    el("text", { class: "tick-label", x: px(x), y: 228, "text-anchor": "middle" }, svg).textContent = String(x);
  });
  var defs = el("defs", {}, svg);
  var clip = el("clipPath", { id: "demoClip" }, defs);
  el("rect", { x: ML, y: 6, width: PW, height: BASE - 6 }, clip);
  var rugData = el("g", {}, svg);
  var rugNoise = el("g", {}, svg);
  var curves = el("g", { "clip-path": "url(#demoClip)" }, svg);
  var pTargetEl = el("path", { class: "curve ink dash" }, curves);
  var pNoiseEl = el("path", { class: "curve ref" }, curves);
  var pModelEl = el("path", { class: "curve acc" }, curves);

  function pathOf(fn) {
    var d = [];
    for (var i = 0; i <= 140; i++) {
      var x = XMIN + i / 140 * (XMAX - XMIN);
      d.push(px(x).toFixed(1) + "," + py(fn(x)).toFixed(1));
    }
    return "M" + d.join(" L");
  }
  pTargetEl.setAttribute("d", pathOf(target));

  // model: logistic classifier over RBF features; logit s(x) = theta . phi(x)
  function feats(x) {
    var f = new Float64Array(NF);
    f[0] = 1;
    for (var i = 0; i < 8; i++) {
      var t = x - CENTERS[i];
      f[i + 1] = Math.exp(-(t * t) / (2 * RW * RW));
    }
    return f;
  }
  function featM(xs) { return xs.map(feats); }
  function dot(a, b) { var s = 0; for (var i = 0; i < NF; i++) s += a[i] * b[i]; return s; }
  function softplus(z) { return z > 30 ? z : Math.log(1 + Math.exp(z)); }

  function randn() {
    var u = 0, v = 0;
    while (!u) u = Math.random();
    while (!v) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function sampleData() {
    var xs = [];
    for (var i = 0; i < N; i++) {
      var c = Math.random() < MIX[0][0] ? MIX[0] : MIX[1];
      xs.push(c[1] + c[2] * randn());
    }
    return xs;
  }
  function sampleNoise() {
    var xs = [];
    for (var i = 0; i < N; i++) xs.push(muQ + sgQ * randn());
    return xs;
  }
  function drawRug(g, xs, cls, y1, y2) {
    while (g.firstChild) g.removeChild(g.firstChild);
    xs.forEach(function (x) {
      if (x < XMIN || x > XMAX) return;
      el("line", { class: cls, x1: px(x).toFixed(1), x2: px(x).toFixed(1), y1: y1, y2: y2, "stroke-width": 1 }, g);
    });
  }

  var muEl = document.getElementById("demoMu"), muV = document.getElementById("demoMuV");
  var sgEl = document.getElementById("demoSg"), sgV = document.getElementById("demoSgV");
  var fitEl = document.getElementById("demoFit");

  var muQ = 0, sgQ = 1.8;
  var theta = new Float64Array(NF), grad = new Float64Array(NF);
  var D = sampleData(), G = sampleNoise();
  var FD = featM(D), FG = featM(G);
  var step = 0, loss = 0, gmax = Infinity, Z = 1, ess = N, running = false;

  function gdSteps(k) {
    for (var t = 0; t < k; t++) {
      grad.fill(0);
      loss = 0;
      var i, j, s, q;
      for (i = 0; i < N; i++) {
        s = dot(theta, FD[i]); q = 1 / (1 + Math.exp(-s));
        loss += softplus(-s);
        for (j = 0; j < NF; j++) grad[j] -= (1 - q) * FD[i][j];
      }
      for (i = 0; i < N; i++) {
        s = dot(theta, FG[i]); q = 1 / (1 + Math.exp(-s));
        loss += softplus(s);
        for (j = 0; j < NF; j++) grad[j] += q * FG[i][j];
      }
      loss /= N;
      gmax = 0;
      for (j = 0; j < NF; j++) {
        grad[j] /= N;
        if (j > 0) grad[j] += LAM * theta[j];
        theta[j] -= LR * grad[j];
        gmax = Math.max(gmax, Math.abs(grad[j]));
      }
      step++;
      if (step >= MAXSTEP) break;
    }
  }
  function noisePdf(x) { return gauss(x, muQ, sgQ); }
  function logitAt(x) { return dot(theta, feats(x)); }
  function modelPdf(x) { // e^{s(x)} Q(x) / Z  with Z importance-sampled on noise
    return Math.exp(Math.min(logitAt(x), 30)) * noisePdf(x) / Z;
  }
  function updateZ() {
    var sw = 0, sw2 = 0, w;
    for (var i = 0; i < N; i++) {
      w = Math.exp(Math.min(dot(theta, FG[i]), 30));
      sw += w; sw2 += w * w;
    }
    Z = sw / N;
    ess = sw2 > 0 ? (sw * sw) / sw2 : 0;
  }

  var cross = crosshair(svg, document.getElementById("demoRead"), {
    xMin: XMIN, xMax: XMAX, bands: [[TOP, BASE]],
    series: [
      { n: "Q", f: noisePdf },
      { n: "P", f: target },
      { n: "P̂", f: modelPdf }
    ]
  });

  function draw() {
    updateZ();
    pNoiseEl.setAttribute("d", pathOf(noisePdf));
    pModelEl.setAttribute("d", pathOf(modelPdf));
    fitEl.textContent = "step " + step + " · loss " + loss.toFixed(3) +
      " · ESS " + Math.round(ess) + "/" + N;
    cross.refresh();
  }

  function frame() {
    if (!running) return;
    gdSteps(30);
    draw();
    if (step >= MAXSTEP || (gmax < 1e-3 && step > 200)) { running = false; return; }
    requestAnimationFrame(frame);
  }
  function start() {
    if (REDUCED) { gdSteps(3000); draw(); return; }
    if (!running) { running = true; requestAnimationFrame(frame); }
  }

  muEl.addEventListener("input", function () {
    muQ = +muEl.value; muV.textContent = muQ.toFixed(1);
    G = sampleNoise(); FG = featM(G);
    drawRug(rugNoise, G, "rug ref", 207, 213);
    step = 0; start();
  });
  sgEl.addEventListener("input", function () {
    sgQ = +sgEl.value; sgV.textContent = sgQ.toFixed(2);
    G = sampleNoise(); FG = featM(G);
    drawRug(rugNoise, G, "rug ref", 207, 213);
    step = 0; start();
  });
  document.getElementById("demoResample").addEventListener("click", function () {
    D = sampleData(); FD = featM(D);
    G = sampleNoise(); FG = featM(G);
    theta.fill(0); step = 0;
    drawRug(rugData, D, "rug acc", 199, 205);
    drawRug(rugNoise, G, "rug ref", 207, 213);
    start();
  });

  drawRug(rugData, D, "rug acc", 199, 205);
  drawRug(rugNoise, G, "rug ref", 207, 213);
  draw();
  start();
})();
