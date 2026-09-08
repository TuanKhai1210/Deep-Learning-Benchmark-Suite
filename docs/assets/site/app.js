(() => {
  "use strict";
  const menu = document.querySelector(".menu-toggle");
  const mobileNav = document.getElementById("mobile-nav");
  if (menu && mobileNav) {
    menu.hidden = false;
    const closeMenu = () => { mobileNav.hidden = true; menu.setAttribute("aria-expanded", "false"); menu.setAttribute("aria-label", "Open navigation"); };
    menu.addEventListener("click", () => {
      const expanded = menu.getAttribute("aria-expanded") === "true";
      mobileNav.hidden = expanded;
      menu.setAttribute("aria-expanded", String(!expanded));
      menu.setAttribute("aria-label", expanded ? "Open navigation" : "Close navigation");
    });
    mobileNav.addEventListener("click", (event) => { if (event.target.closest("a")) closeMenu(); });
    document.addEventListener("keydown", (event) => { if (event.key === "Escape" && !mobileNav.hidden) { closeMenu(); menu.focus(); } });
    matchMedia("(min-width: 761px)").addEventListener("change", (event) => { if (event.matches) closeMenu(); });
  }
  // Model links from team cards open the matching detail, including direct URLs.
  const revealLinkedModel = () => {
    const id = location.hash.slice(1);
    if (!id.startsWith("model-")) return;
    const target = document.getElementById(id);
    if (target instanceof HTMLDetailsElement) target.open = true;
  };
  revealLinkedModel();
  window.addEventListener("hashchange", revealLinkedModel);
  const models = {
    linear: { index: "01 / A LINEAR BASELINE", name: "Linear classifier", label: "LINEAR PROJECTION", description: "Flatten the image and learn a direct mapping to class logits. A simple baseline for the value of added complexity.", layers: [5, 4] },
    mlp: { index: "02 / NONLINEAR FEATURES", name: "Multilayer perceptron", label: "HIDDEN LAYERS", description: "Hidden layers turn flattened pixels into learned nonlinear features. A step beyond the linear baseline.", layers: [4, 6, 6, 4] },
    cnn: { index: "03 / SPATIAL STRUCTURE", name: "Convolutional network", label: "SPATIAL FEATURES", description: "Shared filters build local features across an image. Spatial structure becomes part of the architecture.", layers: [] },
    rnn: { index: "04 / SEQUENTIAL REPRESENTATION", name: "LSTM / GRU", label: "RECURRENT STATES", description: "Read an image as a sequence of rows, columns, or patches. Hidden states carry information across timesteps.", layers: [] },
    transformer: { index: "05 / ATTENTION OVER PATCHES", name: "Transformer encoder", label: "SELF-ATTENTION", description: "Project image patches into tokens, add position information, and learn relationships through self-attention.", layers: [] }
  };
  const diagram = document.getElementById("diagram-content");
  if (!diagram) return;
  const NS = "http://www.w3.org/2000/svg";
  const draw = (tag, attrs, text) => {
    const el = document.createElementNS(NS, tag);
    for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, String(value));
    if (text) el.textContent = text;
    diagram.appendChild(el);
    return el;
  };
  const line = (x1, y1, x2, y2, strong = false) => draw("line", { x1, y1, x2, y2, stroke: strong ? "#d9fa66" : "#69806c", "stroke-width": strong ? 1.5 : .75, opacity: strong ? .7 : .4 });
  const node = (x, y, active = false, radius = 7) => draw("circle", { cx: x, cy: y, r: radius, fill: active ? "#d9fa66" : "#25392a", stroke: active ? "#d9fa66" : "#a3b99e", "stroke-width": 1.2 });
  const text = (x, y, value, fill = "#a9bca3", size = 14) => draw("text", { x, y, fill, "font-family": "Consolas, monospace", "font-size": size, "text-anchor": "middle" }, value);
  const network = (layers) => {
    const positions = layers.map((count, i) => Array.from({ length: count }, (_, j) => [45 + i * 390 / (layers.length - 1), 135 + (j - (count - 1) / 2) * 32]));
    for (let i = 0; i < positions.length - 1; i++) for (const a of positions[i]) for (const b of positions[i + 1]) line(...a, ...b, a === positions[i][1] && b === positions[i + 1][2]);
    positions.forEach((layer, i) => layer.forEach(([x, y], j) => node(x, y, (i === positions.length - 1 && j === 1) || (i === 1 && j === 2))));
    text(45, 252, "784 pixels"); text(435, 252, "10 classes");
  };
  const box = (x, y, width, height, active = false) => draw("rect", { x, y, width, height, rx: 4, fill: active ? "#d9fa6622" : "#263a2b", stroke: active ? "#d9fa66" : "#9bb794", "stroke-width": 1.2 });
  const render = (key) => {
    const model = models[key];
    diagram.replaceChildren();
    if (model.layers.length) network(model.layers);
    if (key === "cnn") {
      line(85, 135, 175, 135, true); line(240, 135, 310, 135, true); line(359, 135, 435, 135, true);
      box(30, 91, 62, 86); for (let i = 1; i < 5; i++) { line(30 + i * 12, 92, 30 + i * 12, 176); line(31, 91 + i * 17, 91, 91 + i * 17); }
      for (let i = 0; i < 3; i++) box(149 + i * 13, 70 + i * 13, 57, 97, i === 2);
      for (let i = 0; i < 3; i++) box(289 + i * 10, 98 + i * 10, 39, 58);
      for (let i = 0; i < 4; i++) node(435, 87 + i * 32, i === 1);
      text(61, 237, "image"); text(190, 237, "convolution"); text(325, 237, "pooling"); text(435, 237, "head");
    }
    if (key === "rnn") {
      const xs = [64, 169, 274, 379];
      xs.forEach((x, i) => {
        if (i < 3) line(x + 26, 120, xs[i + 1] - 26, 120, true);
        line(x, 187, x, 144); box(x - 23, 95, 46, 49, i === 2);
        text(x, 125, "h" + (i + 1), i === 2 ? "#d9fa66" : "#c8d7c4", 13);
        box(x - 22, 186, 44, 13); text(x, 230, "row " + (i + 1));
      });
      line(405, 120, 441, 120, true); node(447, 120, true); text(224, 55, "shared recurrent cell");
    }
    if (key === "transformer") {
      for (let i = 0; i < 4; i++) {
        const y = 64 + i * 43; box(25, y, 27, 27, i === 1); line(52, y + 13, 154, y + 13);
        for (let j = 0; j < 4; j++) line(171, y + 13, 316, 77 + j * 43, i === j);
        node(165, y + 13, i === 1, 6); node(323, y + 13, i === 2, 6); line(330, y + 13, 428, 137);
      }
      node(437, 137, true, 9); text(42, 252, "patches"); text(164, 252, "tokens + pos"); text(315, 252, "attention"); text(435, 252, "head");
    }
    document.getElementById("model-index").textContent = model.index;
    document.getElementById("model-name").textContent = model.name;
    document.getElementById("model-description").textContent = model.description;
    document.getElementById("representation-label").textContent = model.label;
    document.getElementById("diagram-title").textContent = model.name + " conceptual diagram";
    document.querySelectorAll("[data-model]").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.model === key)));
  };
  document.querySelectorAll("[data-model]").forEach(button => button.addEventListener("click", () => render(button.dataset.model)));
  render("mlp");
})();
