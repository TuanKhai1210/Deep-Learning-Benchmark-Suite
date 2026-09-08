---
layout: default
title: Beyond accuracy
body_class: home-page
description: Five neural architectures. One shared protocol. Exploring the trade-offs behind image classification with reproducible deep learning experiments.
---
<section class="hero container">
  <div class="hero-copy">
    <div class="eyebrow hero-eyebrow"><span class="live-dot"></span> STUDENT RESEARCH / HCMUT · VNU-HCM</div>
    <h1>Beyond<br><span class="accent">accuracy.</span></h1>
    <p class="hero-description">Different architectures. One shared question.<br>What makes a model worth choosing?</p>
    <p class="hero-detail">A comparative study of neural networks, their representations, and the trade-offs between performance, complexity, and compute.</p>
    <div class="hero-actions"><a class="button primary" href="{{ '/a1.html' | relative_url }}">Explore the benchmark <span aria-hidden="true">↗</span></a><a class="text-link" href="https://github.com/TuanKhai1210/deep-learning-benchmark-suite">View source <span aria-hidden="true">↗</span></a></div>
    <div class="hero-meta"><span><i class="live-dot"></i>A1 · In development</span><span>PyTorch / Fashion-MNIST</span></div>
  </div>
  <div class="explorer">
    <div class="explorer-top"><span class="eyebrow">ARCHITECTURE EXPLORER</span><span class="figure-number">FIG. 01</span></div>
    <div class="diagram-wrap">
      <div class="diagram-labels"><span>INPUT</span><span id="representation-label">HIDDEN LAYERS</span><span>LOGITS</span></div>
      <svg id="architecture-diagram" viewBox="0 0 480 270" role="img" aria-labelledby="diagram-title diagram-desc"><title id="diagram-title">Multilayer perceptron diagram</title><desc id="diagram-desc">Illustration of input, hidden layers and class outputs. This diagram does not represent measured results or a finalized model configuration.</desc><g id="diagram-content"></g></svg>
      <noscript><p class="no-script-note">Explore five approaches: flattened pixels, nonlinear features, spatial filters, sequential states, and attention over patches.</p></noscript>
      <span class="diagram-caption">CONCEPTUAL VIEW · NOT EXPERIMENTAL RESULTS</span>
    </div>
    <div class="model-picker" role="group" aria-label="Explore an architecture">
      <button type="button" data-model="linear" aria-pressed="false">Linear</button>
      <button type="button" data-model="mlp" aria-pressed="true">MLP</button>
      <button type="button" data-model="cnn" aria-pressed="false">CNN</button>
      <button type="button" data-model="rnn" aria-pressed="false">LSTM / GRU</button>
      <button type="button" data-model="transformer" aria-pressed="false">Transformer</button>
    </div>
    <div class="explorer-description" aria-live="polite" aria-atomic="true"><div><span id="model-index" class="eyebrow">02 / NONLINEAR FEATURES</span><h2 id="model-name">Multilayer perceptron</h2></div><p id="model-description">Hidden layers turn flattened pixels into learned nonlinear features. A step beyond the linear baseline.</p></div>
  </div>
</section>

<div class="metrics-strip"><div class="container metrics-inner"><div><strong>05</strong><span>Architecture families</span></div><div><strong>03</strong><span>Training seeds</span></div><div><strong>15</strong><span>Planned main runs</span></div><div class="metric-statement"><span class="eyebrow">ONE SHARED STANDARD</span><p>Reproducibility<br>comes first.</p></div></div></div>

<section class="section container" id="research">
  <div class="section-heading"><div><span class="eyebrow">01 / RESEARCH PROGRAM</span><h2>Three studies.<br>A wider perspective.</h2></div><p>From the foundations of image classification to specialized tasks and multimodal learning.</p></div>
  <div class="project-grid">
    <a class="project-card featured" href="{{ '/a1.html' | relative_url }}">
      <div class="card-top"><span class="card-number">01</span><span class="status-tag dark-tag"><i class="live-dot"></i>IN DEVELOPMENT</span></div>
      <div class="project-icon icon-layers" aria-hidden="true"><i></i><i></i><i></i></div>
      <span class="eyebrow">FASHION-MNIST / IMAGE CLASSIFICATION</span><h3>Foundations &amp;<br>architecture benchmarks</h3><p>Linear to Transformer. Compare five model families under one data and evaluation protocol.</p>
      <div class="card-bottom"><span>Explore Assignment 1</span><span class="circle-arrow" aria-hidden="true">↗</span></div>
    </a>
    <a class="project-card" href="{{ '/a2.html' | relative_url }}">
      <div class="card-top"><span class="card-number">02</span><span class="status-tag">PLANNED</span></div>
      <div class="project-icon icon-grid" aria-hidden="true"><i></i><i></i><i></i><i></i></div>
      <span class="eyebrow">DATASET &amp; TASK TO BE CONFIRMED</span><h3>Scale &amp;<br>specialized learning</h3><p>Extend the study to larger-scale data and a specialized task, subject to proposal approval.</p>
      <div class="card-bottom"><span>View research scope</span><span class="circle-arrow" aria-hidden="true">↗</span></div>
    </a>
    <a class="project-card" href="{{ '/a3.html' | relative_url }}">
      <div class="card-top"><span class="card-number">03</span><span class="status-tag">PLANNED</span></div>
      <div class="project-icon icon-fusion" aria-hidden="true"><i></i><i></i></div>
      <span class="eyebrow">MULTIMODAL / PROPOSAL PENDING</span><h3>Multiple signals.<br>Shared understanding.</h3><p>Explore how combining different modalities changes what a model can learn.</p>
      <div class="card-bottom"><span>View research scope</span><span class="circle-arrow" aria-hidden="true">↗</span></div>
    </a>
  </div>
</section>

<section class="protocol-section" id="protocol"><div class="container">
  <div class="section-heading"><div><span class="eyebrow">02 / THE EXPERIMENTAL STANDARD</span><h2>A fair comparison<br>starts with the protocol.</h2></div><a class="text-link" href="{{ '/a1-experiment-contract.html' | relative_url }}">Read the experiment contract ↗</a></div>
  <div class="protocol-grid"><div class="protocol-copy"><span class="status-tag">A1-V0 · DRAFT FOR REVIEW</span><p class="large-copy">Same data.<br>Independent runs.<br><span class="accent">Traceable evidence.</span></p><p>Each reported result will connect to its configuration, saved split, checkpoint, environment, and code revision.</p><div class="seed-block"><div><span class="eyebrow">FIXED SPLIT SEED</span><strong>36</strong></div><div><span class="eyebrow">TRAINING SEEDS</span><div class="seed-list"><code>69420</code><code>67</code><code>69</code></div></div></div></div>
  <div class="protocol-steps"><div class="protocol-step"><span>01</span><div><h3>Understand the data</h3><p>Fashion-MNIST, class distributions, representative samples, and explicit leakage checks.</p></div><span class="step-symbol" aria-hidden="true">↗</span></div><div class="protocol-step"><span>02</span><div><h3>Build a shared pipeline</h3><p>One saved partition. Normalization fitted on training data. Consistent model inputs and evaluation.</p></div><span class="step-symbol" aria-hidden="true">↗</span></div><div class="protocol-step"><span>03</span><div><h3>Measure the trade-offs</h3><p>Accuracy, macro-F1, parameter counts, training time, and inference time.</p></div><span class="step-symbol" aria-hidden="true">↗</span></div><div class="protocol-step"><span>04</span><div><h3>Explain the behavior</h3><p>Learning curves, confusion patterns, prediction examples, and architectural inductive bias.</p></div><span class="step-symbol" aria-hidden="true">↗</span></div></div></div>
</div></section>

<section class="section container">
  <div class="section-heading"><div><span class="eyebrow">03 / CURRENT FOCUS</span><h2>Building the evidence.</h2></div><p>Assignment 1 is in development. Verified measurements will be published as experiments are completed.</p></div>
  <div class="progress-grid"><div class="progress-card"><span class="eyebrow">AGREED</span><h3>The experiment’s foundations</h3><ul class="check-list"><li>Fashion-MNIST as the main dataset</li><li>Team responsibilities and model ownership</li><li>Split seed 36 and three training seeds</li></ul><a class="text-link" href="{{ '/a1.html' | relative_url }}">See the full study ↗</a></div><div class="progress-card"><span class="eyebrow">IN DEVELOPMENT</span><h3>From pipeline to comparison</h3><p>Data preparation, model implementations, training and validation, and experiment logging.</p><div class="release-note"><span class="live-dot"></span><span>Benchmark results are not available yet.</span></div></div></div>
</section>

<section class="team-section section container" id="team">
  <div class="section-heading"><div><span class="eyebrow">04 / THE TEAM</span><h2>Three people.<br>One shared investigation.</h2></div><p>CO3133 · Semester 261<br>Guided by Lê Thành Sách.<br>Ho Chi Minh City University of Technology.</p></div>
  <div class="team-grid">
    <article class="team-card"><div class="team-top"><span class="avatar avatar-mint" aria-hidden="true">HT</span><span class="eyebrow">A / DATA &amp; VISION</span></div><h3>Nguyễn Hạo Thiên</h3><span class="student-id">2453194</span><p>Dataset, EDA, preprocessing, and visual error analysis.</p><div class="model-tags"><a href="{{ '/a1.html#model-linear' | relative_url }}">Linear</a><a href="{{ '/a1.html#model-cnn' | relative_url }}">CNN</a></div><a href="https://github.com/MrzThien1105">GitHub / MrzThien1105 <span aria-hidden="true">↗</span></a></article>
    <article class="team-card"><div class="team-top"><span class="avatar avatar-blue" aria-hidden="true">AK</span><span class="eyebrow">B / TRAINING &amp; EVALUATION</span></div><h3>Nguyễn Anh Khoa</h3><span class="student-id">2452539</span><p>Shared training pipeline, metrics, checkpoints, and timing.</p><div class="model-tags"><a href="{{ '/a1.html#model-mlp' | relative_url }}">MLP</a><a href="{{ '/a1.html#model-rnn' | relative_url }}">LSTM / GRU</a></div><a href="https://github.com/TCL03-HCMUT">GitHub / TCL03-HCMUT <span aria-hidden="true">↗</span></a></article>
    <article class="team-card"><div class="team-top"><span class="avatar avatar-peach" aria-hidden="true">TK</span><span class="eyebrow">C / REPRODUCIBILITY &amp; INTEGRATION</span></div><h3>Tạ Tuấn Khải</h3><span class="student-id">2452515</span><p>Configuration, reproducibility, system integration, and documentation.</p><div class="model-tags"><a href="{{ '/a1.html#model-transformer' | relative_url }}">Transformer</a><a href="{{ '/a1-experiment-contract.html' | relative_url }}">Integration</a></div><a href="https://github.com/TuanKhai1210">GitHub / TuanKhai1210 <span aria-hidden="true">↗</span></a></article>
  </div><p class="team-note">Each member implements, tests, and explains their own contribution. Review rotation: Khoa → Thiên · Khải → Khoa · Thiên → Khải. Group ID: pending confirmation.</p>
</section>

<section class="container disclosure-section"><div class="disclosure-box"><span class="disclosure-symbol" aria-hidden="true">↳</span><div><span class="eyebrow">TRANSPARENCY, BY DESIGN</span><h2>AI-assisted. Review required.</h2><p>AI has supported planning, interfaces, configuration, documentation, and this website’s design. Team review is required; experimental claims will be supported by measured, reproducible evidence.</p></div><a class="text-link" href="https://github.com/TuanKhai1210/deep-learning-benchmark-suite/blob/main/AI_USAGE.md">Read the disclosure ↗</a></div></section>
