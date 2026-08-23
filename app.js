/**
 * DeepPaper Studio — Interactive Neural Architecture Explorer
 * Production Suite: Comprehensive Research Architecture Workbench
 * Features:
 * - 7 Deep Neural Architectures with Interactive Graphical Diagrams
 * - Interactive Token Flow Forward Simulation with animated step cadence
 * - Deep Multi-Tab Micro-Layers Catalog with Live Dynamic Tensor Calculator,
 *   Hyperparameters, Hardware Kernel Execution Notes, Failure Modes, and PyTorch code
 * - Drill-Down Inner-Circuit Modals mapping step-by-step internal mechanisms
 * - Phase 3: Cross-Model Layer-by-Layer & Trade-off Comparison Engine
 * - Phase 4: Live Computational Complexity Simulator & Memory Divergence Visualizer
 */

let papers = [];
let currentPaperIndex = 0;
let masteryState = {};
let currentBatchSize = 2;
let currentSeqLen = 2048;
let isSimulatingFlow = false;
let simulationTimeout = null;

// DOM Elements — Global & Single Paper Explorer
const paperNavTabs = document.getElementById('paperNavTabs');
const paperTitle = document.getElementById('paperTitle');
const paperAuthors = document.getElementById('paperAuthors');
const paperBreakthrough = document.getElementById('paperBreakthrough');
const heroTags = document.getElementById('heroTags');
const diagramTitle = document.getElementById('diagramTitle');
const diagramCanvasViewport = document.getElementById('diagramCanvasViewport');
const layerCardsGrid = document.getElementById('layerCardsGrid');
const microSectionTitle = document.getElementById('microSectionTitle');
const prosList = document.getElementById('prosList');
const consList = document.getElementById('consList');
const reportedMetricsTableBody = document.getElementById('reportedMetricsTableBody');
const paperStudyStatusBtn = document.getElementById('paperStudyStatusBtn');
const paperStudyStatusText = document.getElementById('paperStudyStatusText');
const masteryPercentText = document.getElementById('masteryPercentText');
const themeToggleBtn = document.getElementById('themeToggleBtn');
const simulateFlowBtn = document.getElementById('simulateFlowBtn');
const expandAllCardsBtn = document.getElementById('expandAllCardsBtn');
const tensorLiveDimIndicator = document.getElementById('tensorLiveDimIndicator');

// Modal Elements
const layerModalOverlay = document.getElementById('layerModalOverlay');
const closeLayerModalBtn = document.getElementById('closeLayerModalBtn');
const modalLayerBadge = document.getElementById('modalLayerBadge');
const modalTitle = document.getElementById('modalTitle');
const modalCircuitText = document.getElementById('modalCircuitText');
const modalFormulaSection = document.getElementById('modalFormulaSection');
const modalFormulaContent = document.getElementById('modalFormulaContent');
const modalUndertonesText = document.getElementById('modalUndertonesText');
const modalCitationTag = document.getElementById('modalCitationTag');
const modalInnerDiagramContainer = document.getElementById('modalInnerDiagramContainer');

// Citation Drawer
const citationDrawer = document.getElementById('citationDrawer');
const citationDrawerTitle = document.getElementById('citationDrawerTitle');
const citationDrawerText = document.getElementById('citationDrawerText');
const closeCitationDrawerBtn = document.getElementById('closeCitationDrawerBtn');

// Comparison Engine Elements
const compareModelASelect = document.getElementById('compareModelASelect');
const compareModelBSelect = document.getElementById('compareModelBSelect');
const comparisonBody = document.getElementById('comparisonBody');

// Simulator Elements
const simSeqLen = document.getElementById('simSeqLen');
const simSeqLenVal = document.getElementById('simSeqLenVal');
const simHiddenDim = document.getElementById('simHiddenDim');
const simHiddenDimVal = document.getElementById('simHiddenDimVal');
const simNumLayers = document.getElementById('simNumLayers');
const simNumLayersVal = document.getElementById('simNumLayersVal');
const simBatchSize = document.getElementById('simBatchSize');
const simBatchSizeVal = document.getElementById('simBatchSizeVal');
const simStateDim = document.getElementById('simStateDim');
const simStateDimVal = document.getElementById('simStateDimVal');
const simPrecision = document.getElementById('simPrecision');
const simPrecisionVal = document.getElementById('simPrecisionVal');
const computedMetricsGrid = document.getElementById('computedMetricsGrid');
const memoryDivergenceChart = document.getElementById('memoryDivergenceChart');


// ──────────────────────────────────────────────────────────────────
// BOOTSTRAP INITIALIZATION
// ──────────────────────────────────────────────────────────────────
async function initApp() {
  loadMasteryState();
  initTheme();
  initViewTabs();
  initModalEvents();
  initCitationDrawer();
  initSimulationEvents();

  try {
    const res = await fetch('papers_data.json');
    if (!res.ok) throw new Error('Failed to load papers_data.json');
    papers = await res.json();
  } catch (err) {
    console.error('Error loading papers_data.json:', err);
  }

  if (papers && papers.length > 0) {
    renderPaperNav();
    selectPaper(0);
    initComparisonSelectors();
    initSimulator();
  }
}


// ──────────────────────────────────────────────────────────────────
// MASTERY TRACKER
// ──────────────────────────────────────────────────────────────────
function loadMasteryState() {
  try {
    const s = localStorage.getItem('deeppaper_mastery');
    if (s) masteryState = JSON.parse(s);
  } catch (e) {
    masteryState = {};
  }
  updateMasteryBadge();
}

function togglePaperMastery(pid) {
  masteryState[pid] = !masteryState[pid];
  try {
    localStorage.setItem('deeppaper_mastery', JSON.stringify(masteryState));
  } catch (e) {}
  updateMasteryBadge();
  updateMasteryButton(pid);
}

function updateMasteryBadge() {
  if (!papers || papers.length === 0) return;
  const c = Object.values(masteryState).filter(Boolean).length;
  masteryPercentText.textContent = `${Math.round((c / papers.length) * 100)}% Mastered (${c}/${papers.length})`;
}

function updateMasteryButton(pid) {
  const m = !!masteryState[pid];
  paperStudyStatusBtn.classList.toggle('mastered', m);
  paperStudyStatusText.textContent = m ? 'Mastered ✓' : 'Mark Paper as Mastered';
}

paperStudyStatusBtn.addEventListener('click', () => {
  if (papers[currentPaperIndex]) togglePaperMastery(papers[currentPaperIndex].paper_id);
});


// ──────────────────────────────────────────────────────────────────
// THEME SWITCHER
// ──────────────────────────────────────────────────────────────────
function initTheme() {
  document.body.className = localStorage.getItem('deeppaper_theme') || 'theme-dark';
  themeToggleBtn.addEventListener('click', () => {
    const n = document.body.classList.contains('theme-dark') ? 'theme-light' : 'theme-dark';
    document.body.className = n;
    localStorage.setItem('deeppaper_theme', n);
  });
}


// ──────────────────────────────────────────────────────────────────
// VIEW NAVIGATION TABS
// ──────────────────────────────────────────────────────────────────
function initViewTabs() {
  const viewTabs = document.querySelectorAll('.view-tab');
  const views = {
    'graphical-architecture': document.getElementById('graphicalArchitectureView'),
    'layer-breakdown': document.getElementById('layerBreakdownView'),
    'pros-cons': document.getElementById('prosConsView'),
    'reported-metrics': document.getElementById('reportedMetricsView'),
    'model-comparison': document.getElementById('modelComparisonView'),
    'complexity-simulator': document.getElementById('complexitySimulatorView')
  };

  viewTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      viewTabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      const target = tab.getAttribute('data-view');
      Object.entries(views).forEach(([k, el]) => {
        if (el) el.classList.toggle('active', k === target);
      });
      if (target === 'model-comparison') {
        renderComparisonView();
      } else if (target === 'complexity-simulator') {
        updateSimulator();
      } else if (target === 'layer-breakdown') {
        triggerKaTeX();
      }
      triggerKaTeX();
    });
  });
}


// ──────────────────────────────────────────────────────────────────
// CITATIONS
// ──────────────────────────────────────────────────────────────────
function initCitationDrawer() {
  closeCitationDrawerBtn.addEventListener('click', () => citationDrawer.classList.remove('open'));
}

function showCitation(title, src) {
  citationDrawerTitle.textContent = title;
  citationDrawerText.textContent = `Primary Paper Reference: ${src}`;
  citationDrawer.classList.add('open');
}


// ──────────────────────────────────────────────────────────────────
// MODAL EVENTS
// ──────────────────────────────────────────────────────────────────
function initModalEvents() {
  closeLayerModalBtn.addEventListener('click', closeModal);
  layerModalOverlay.addEventListener('click', e => {
    if (e.target === layerModalOverlay) closeModal();
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
      closeModal();
      citationDrawer.classList.remove('open');
    }
  });
}

function openModal() {
  layerModalOverlay.classList.add('open');
  layerModalOverlay.setAttribute('aria-hidden', 'false');
  triggerKaTeX();
}

function closeModal() {
  layerModalOverlay.classList.remove('open');
  layerModalOverlay.setAttribute('aria-hidden', 'true');
}


// ──────────────────────────────────────────────────────────────────
// TOP GLOBAL PAPER NAV
// ──────────────────────────────────────────────────────────────────
function renderPaperNav() {
  paperNavTabs.innerHTML = '';
  papers.forEach((p, i) => {
    const btn = document.createElement('button');
    btn.className = `paper-nav-btn ${i === currentPaperIndex ? 'active' : ''}`;
    let sn = p.title.split(':')[0].trim();
    if (sn.length > 22) sn = sn.substring(0, 20) + '...';
    btn.innerHTML = `<span>${sn}</span><span class="badge-pill">${p.computed_metric_role.toUpperCase()}</span>`;
    btn.addEventListener('click', () => selectPaper(i));
    paperNavTabs.appendChild(btn);
  });
}


// ──────────────────────────────────────────────────────────────────
// SELECT ACTIVE PAPER
// ──────────────────────────────────────────────────────────────────
function selectPaper(index) {
  currentPaperIndex = index;
  const paper = papers[index];
  document.querySelectorAll('.paper-nav-btn').forEach((b, i) => b.classList.toggle('active', i === index));
  paperTitle.textContent = paper.title;
  paperAuthors.textContent = paper.authors_venue_year;
  paperBreakthrough.textContent = paper.core_breakthrough;
  heroTags.innerHTML = `
    <span class="tag-badge">ID: ${paper.paper_id}</span>
    <span class="tag-badge">Category: ${paper.computed_metric_role.toUpperCase()}</span>
    <span class="tag-badge">${paper.layers.length} Macro Layers</span>
  `;
  updateMasteryButton(paper.paper_id);
  renderGraphicalDiagram(paper);
  renderMicroLayers(paper);
  renderProsCons(paper);
  renderReportedMetrics(paper);
  triggerKaTeX();
}


// ──────────────────────────────────────────────────────────────────
// HELPERS: Render Clickable Block & Flow Arrows
// ──────────────────────────────────────────────────────────────────
function block(label, nodeId, colorClass, paperId, extra = '') {
  return `<div class="graph-block ${colorClass}" id="block_${nodeId}" onclick="drillDownNode('${paperId}','${nodeId}')" ${extra}>${label}</div>`;
}

function arrow() {
  return `<div class="flow-arrow"><svg width="16" height="22" viewBox="0 0 16 22"><path d="M8 0 L8 16 M2 12 L8 18 L14 12" stroke="currentColor" stroke-width="2" fill="none"/></svg></div>`;
}

function arrowUp() {
  return `<div class="flow-arrow"><svg width="16" height="22" viewBox="0 0 16 22"><path d="M8 22 L8 6 M2 10 L8 4 L14 10" stroke="currentColor" stroke-width="2" fill="none"/></svg></div>`;
}

function getCC(c) {
  return { pink: 'pink-block', orange: 'orange-block', yellow: 'yellow-block', blue: 'blue-block', purple: 'purple-block', green: 'green-block' }[c] || 'orange-block';
}


// ──────────────────────────────────────────────────────────────────
// VIEW 1: GRAPHICAL ARCHITECTURE CANVAS
// ──────────────────────────────────────────────────────────────────
function renderGraphicalDiagram(paper) {
  const diag = paper.graphical_diagram;
  if (!diag) return;
  diagramTitle.textContent = diag.diagram_title || `${paper.title.split(':')[0]} Architecture`;
  diagramCanvasViewport.innerHTML = '';

  if (diag.layout_type === 'dual_column_encoder_decoder') {
    renderTransformerDualColumn(paper);
  } else if (diag.layout_type === 'mamba_block_split') {
    renderMambaSplitBlock(paper);
  } else {
    renderGenericPipeline(paper, diag);
  }
}

// 1. TRANSFORMER
function renderTransformerDualColumn(paper) {
  const pid = paper.paper_id;
  const container = document.createElement('div');
  container.className = 'transformer-canvas';

  container.innerHTML = `
    <!-- ENCODER STACK -->
    <div class="canvas-column">
      <div class="column-label">Encoder Stack</div>

      <div class="nx-container-box">
        <div class="nx-badge">N×</div>
        ${block('Add & Norm', 'trans_enc_add_norm2', 'yellow-block', pid)}
        ${arrowUp()}
        ${block('Feed Forward', 'trans_enc_ffn', 'blue-block', pid)}
        ${arrowUp()}
        ${block('Add & Norm', 'trans_enc_add_norm1', 'yellow-block', pid)}
        ${arrowUp()}
        ${block('Multi-Head Attention', 'trans_enc_mha', 'orange-block', pid)}
      </div>

      ${arrowUp()}
      ${block('Positional Encoding + Embedding', 'trans_pos_enc_enc', 'pink-block', pid, 'style="width:210px"')}
      ${arrowUp()}
      <div class="graph-block io-block" id="block_trans_inputs">Inputs</div>
    </div>

    <!-- Cross-Attention Bridge -->
    <div class="cross-bridge">
      <div class="bridge-line"></div>
      <div class="bridge-label">K, V →</div>
      <div class="bridge-line"></div>
    </div>

    <!-- DECODER STACK -->
    <div class="canvas-column">
      <div class="column-label">Decoder Stack</div>

      <div class="graph-block io-block" id="block_trans_output_prob">Output Probabilities</div>
      ${arrow()}
      ${block('Softmax', 'trans_softmax', 'green-block', pid)}
      ${arrow()}
      ${block('Linear', 'trans_linear', 'purple-block', pid)}
      ${arrow()}

      <div class="nx-container-box">
        <div class="nx-badge">N×</div>
        ${block('Add & Norm', 'trans_dec_add_norm3', 'yellow-block', pid)}
        ${arrowUp()}
        ${block('Feed Forward', 'trans_dec_ffn', 'blue-block', pid)}
        ${arrowUp()}
        ${block('Add & Norm', 'trans_dec_add_norm2', 'yellow-block', pid)}
        ${arrowUp()}
        ${block('Multi-Head Attention (Cross)', 'trans_dec_cross_mha', 'orange-block cross-attn-highlight', pid)}
        ${arrowUp()}
        ${block('Add & Norm', 'trans_dec_add_norm1', 'yellow-block', pid)}
        ${arrowUp()}
        ${block('Masked Multi-Head Attention', 'trans_dec_masked_mha', 'orange-block', pid)}
      </div>

      ${arrowUp()}
      ${block('Positional Encoding + Embedding', 'trans_pos_enc_dec', 'pink-block', pid, 'style="width:210px"')}
      ${arrowUp()}
      <div class="graph-block io-block" id="block_trans_outputs">Outputs (shifted right)</div>
    </div>
  `;

  diagramCanvasViewport.appendChild(container);
}

// 2. MAMBA
function renderMambaSplitBlock(paper) {
  const pid = paper.paper_id;
  const container = document.createElement('div');
  container.className = 'mamba-canvas';

  container.innerHTML = `
    <div class="graph-block io-block" id="block_mamba_in">Input Token x (B, L, D)</div>
    ${arrow()}
    ${block('Linear Expansion (D → 2E) & Dual-Branch Split', 'mamba_expand_split', 'purple-block', pid, 'style="width:280px"')}
    
    <div class="mamba-split-indicator">
      <div class="split-line left-split"></div>
      <div class="split-label">Split into two branches</div>
      <div class="split-line right-split"></div>
    </div>

    <div class="mamba-branches-row">
      <!-- SSM Branch (Left) -->
      <div class="mamba-branch mamba-branch-left">
        <div class="branch-label ssm-label">SSM Processing Branch</div>
        ${block('1D Causal Conv (k=4)', 'mamba_conv1d', 'blue-block', pid)}
        ${arrow()}
        ${block('SiLU (Swish) σ(x)·x', 'mamba_silu1', 'green-block', pid)}
        ${arrow()}
        ${block('Selective Param Gen (Δ, B, C)', 'mamba_param_gen', 'orange-block', pid)}
        ${arrow()}
        ${block('ZOH Discretization & Parallel Scan', 'mamba_discretize_scan', 'orange-block thick-border', pid)}
      </div>

      <!-- Gate Branch (Right) -->
      <div class="mamba-branch mamba-branch-right">
        <div class="branch-label gate-label">Multiplicative Gate</div>
        <div class="mamba-spacer"></div>
        ${block('Gate SiLU Activation', 'mamba_silu2', 'green-block', pid)}
      </div>
    </div>

    <div class="mamba-merge-indicator">
      <div class="merge-line left-merge"></div>
      <div class="merge-label">⊙ Merge</div>
      <div class="merge-line right-merge"></div>
    </div>

    ${block('⊙ Hadamard Product — Multiplicative Gating', 'mamba_mult_gate', 'yellow-block merge-block', pid)}
    ${arrow()}
    ${block('Output Linear Projection (E → D)', 'mamba_linear_out', 'purple-block', pid)}
    ${arrow()}
    
    <div class="residual-add-row">
      ${block('⊕ Residual Sum (+) with Input x', 'mamba_res_add', 'yellow-block', pid, 'style="width:240px"')}
      <div class="residual-skip-label">← skip from input</div>
    </div>
    ${arrow()}
    <div class="graph-block io-block">Block Output (B, L, D)</div>
  `;

  diagramCanvasViewport.appendChild(container);
}

// 3. GENERIC PIPELINES (MoD, MoE, SWAT, Titans, TransMamba)
function renderGenericPipeline(paper, diag) {
  const pid = paper.paper_id;
  const container = document.createElement('div');
  container.className = 'generic-graph-canvas';

  const isMoD = pid === 'mixture_of_depths' || pid === 'mod';
  const nodes = diag.nodes.filter(n => n.color !== 'io');

  let html = `<div class="graph-block io-block">Input Stream</div>${arrow()}`;

  if (isMoD && nodes.length >= 4) {
    const router = nodes[0];
    html += `${block(router.label, router.id, getCC(router.color), pid)}`;
    html += `<div class="mamba-split-indicator"><div class="split-line left-split"></div><div class="split-label">Router Decision</div><div class="split-line right-split"></div></div>`;
    html += `<div class="mamba-branches-row">`;
    html += `<div class="mamba-branch mamba-branch-left"><div class="branch-label" style="color:var(--block-blue)">Selected Tokens (Top-k)</div>`;
    for (let i = 1; i < nodes.length; i++) {
      html += block(nodes[i].label, nodes[i].id, getCC(nodes[i].color), pid);
      if (i < nodes.length - 1) html += arrow();
    }
    html += `</div>`;
    html += `<div class="mamba-branch mamba-branch-right"><div class="branch-label" style="color:var(--text-muted)">Unselected Tokens</div>`;
    html += `<div class="bypass-pipe"><div class="bypass-arrow-flow"></div><span>Identity Bypass</span></div>`;
    html += `</div></div>`;
    html += `<div class="mamba-merge-indicator"><div class="merge-line left-merge"></div><div class="merge-label">Merge</div><div class="merge-line right-merge"></div></div>`;
  } else {
    nodes.forEach((node, i) => {
      html += block(node.label, node.id, getCC(node.color), pid);
      if (i < nodes.length - 1) html += arrow();
    });
  }

  html += `${arrow()}<div class="graph-block io-block">Layer Output Stream</div>`;
  container.innerHTML = html;
  diagramCanvasViewport.appendChild(container);
}


// ──────────────────────────────────────────────────────────────────
// TOKEN FORWARD PASS SIMULATION ENGINE
// ──────────────────────────────────────────────────────────────────
function initSimulationEvents() {
  if (simulateFlowBtn) {
    simulateFlowBtn.addEventListener('click', toggleTokenSimulation);
  }
  if (expandAllCardsBtn) {
    expandAllCardsBtn.addEventListener('click', toggleExpandAllLayers);
  }
}

function toggleTokenSimulation() {
  if (isSimulatingFlow) {
    stopTokenSimulation();
  } else {
    startTokenSimulation();
  }
}

function startTokenSimulation() {
  isSimulatingFlow = true;
  simulateFlowBtn.classList.add('running');
  simulateFlowBtn.innerHTML = `
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
      <rect x="4" y="4" width="16" height="16" rx="2"></rect>
    </svg>
    <span>Stop Simulation</span>`;

  const blocks = Array.from(diagramCanvasViewport.querySelectorAll('.graph-block'));
  if (blocks.length === 0) return;

  // Clear any existing active class
  blocks.forEach(b => b.classList.remove('sim-active'));

  let step = 0;
  function stepSimulation() {
    if (!isSimulatingFlow) return;
    blocks.forEach(b => b.classList.remove('sim-active'));
    if (step < blocks.length) {
      blocks[step].classList.add('sim-active');
      step++;
      simulationTimeout = setTimeout(stepSimulation, 550);
    } else {
      // Loop or finish
      setTimeout(() => {
        if (isSimulatingFlow) {
          step = 0;
          stepSimulation();
        }
      }, 800);
    }
  }
  stepSimulation();
}

function stopTokenSimulation() {
  isSimulatingFlow = false;
  if (simulationTimeout) clearTimeout(simulationTimeout);
  simulateFlowBtn.classList.remove('running');
  simulateFlowBtn.innerHTML = `
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
      <polygon points="5 3 19 12 5 21 5 3"></polygon>
    </svg>
    <span>Simulate Token Flow</span>`;
  const blocks = diagramCanvasViewport.querySelectorAll('.graph-block');
  blocks.forEach(b => b.classList.remove('sim-active'));
}

function toggleExpandAllLayers() {
  const cards = document.querySelectorAll('.layer-accordion-card');
  const anyClosed = Array.from(cards).some(c => !c.classList.contains('active'));
  cards.forEach(c => c.classList.toggle('active', anyClosed));
  expandAllCardsBtn.textContent = anyClosed ? 'Collapse All Layers' : 'Expand All Layers';
}


// ──────────────────────────────────────────────────────────────────
// DYNAMIC TENSOR SHAPE PLAYGROUND
// ──────────────────────────────────────────────────────────────────
function setTensorShapePreset(b, l) {
  currentBatchSize = b;
  currentSeqLen = l;

  // Update preset buttons
  document.querySelectorAll('.tensor-preset-pill').forEach(btn => {
    const text = btn.textContent;
    btn.classList.toggle('active', text.includes(`B=${b}`) && text.includes(`L=${l.toLocaleString()}`));
  });

  // Update live indicator
  if (tensorLiveDimIndicator) {
    tensorLiveDimIndicator.textContent = `Active Dimensions: [Batch: ${b}, SeqLen: ${l.toLocaleString()}, Dim: 512]`;
  }

  // Re-render micro-layers with updated numerical tensor shapes
  if (papers[currentPaperIndex]) {
    renderMicroLayers(papers[currentPaperIndex]);
  }
}

function formatConcreteTensorShape(symbolicStr, b = currentBatchSize, l = currentSeqLen) {
  if (!symbolicStr) return `[${b}, ${l}, 512]`;
  let s = symbolicStr.replace(/\bB\b/g, b.toString()).replace(/\bL\b/g, l.toLocaleString()).replace(/\bT\b/g, l.toLocaleString());
  s = s.replace(/\bL_target\b/g, l.toLocaleString()).replace(/\bL_source\b/g, l.toLocaleString()).replace(/\bL_tgt\b/g, l.toLocaleString()).replace(/\bL_src\b/g, l.toLocaleString());
  return s;
}


// ──────────────────────────────────────────────────────────────────
// COMPREHENSIVE INNER CIRCUIT REGISTRY (ALL 7 ARCHITECTURES)
// ──────────────────────────────────────────────────────────────────
const INNER_CIRCUITS = {
  // ── TRANSFORMER ──
  trans_enc_mha: { title: 'Inside Multi-Head Self-Attention', steps: [
    { label: 'Input X (B, L, d_model)', color: 'io', desc: 'Token embeddings enter the attention mechanism' },
    { label: 'Linear Projections → Q, K, V', color: 'purple', desc: 'Three separate weight matrices W_Q, W_K, W_V project input to query/key/value spaces' },
    { label: 'Split into h=8 Parallel Heads', color: 'blue', desc: 'Reshape (B, L, d_model) → (B, h, L, d_k) where d_k = d_model/h = 64' },
    { label: 'Scaled Dot-Product: QKᵀ / √d_k', color: 'orange', desc: 'Each head computes (L × L) pairwise attention score matrix' },
    { label: 'Softmax (row-wise)', color: 'green', desc: 'Normalizes scores into attention weight distribution per query' },
    { label: 'Weighted Sum: Attention × V', color: 'orange', desc: 'Aggregate value vectors weighted by attention probabilities' },
    { label: 'Concat All Heads', color: 'yellow', desc: 'Concatenate h head outputs: (B, h, L, d_k) → (B, L, d_model)' },
    { label: 'Output Projection W_O', color: 'purple', desc: 'Final linear layer mixes information across heads' },
  ]},
  trans_dec_masked_mha: { title: 'Inside Masked Multi-Head Attention', steps: [
    { label: 'Decoder Input Tokens', color: 'io', desc: 'Previous decoder sublayer output' },
    { label: 'Linear → Q, K, V', color: 'purple', desc: 'Same triple projection as encoder MHA' },
    { label: 'Apply Causal Mask M', color: 'orange', desc: 'M[i,j] = -∞ for j > i — blocks future token access' },
    { label: 'Masked QKᵀ / √d_k', color: 'orange', desc: 'Attention scores with mask ensure autoregressive property' },
    { label: 'Softmax → Causal Weights', color: 'green', desc: 'Lower-triangular attention matrix — each token sees only past' },
    { label: 'Weighted V + Concat + W_O', color: 'purple', desc: 'Aggregate, merge heads, project to d_model' },
  ]},
  trans_dec_cross_mha: { title: 'Inside Cross-Attention (Encoder→Decoder)', steps: [
    { label: 'Decoder Hidden State → Q', color: 'io', desc: 'Query comes from previous decoder sublayer' },
    { label: 'Encoder Final Output → K, V', color: 'pink', desc: 'Keys and Values come from encoder stack output' },
    { label: 'Q_dec, K_enc, V_enc Projections', color: 'purple', desc: 'Q projected from decoder, K & V from encoder via separate weight matrices' },
    { label: 'Cross-Attention: Q_dec × K_enc^T', color: 'orange', desc: 'Decoder queries attend to every encoder position' },
    { label: 'Softmax → Cross Weights', color: 'green', desc: 'How much each decoder position attends to each encoder position' },
    { label: 'Weighted V_enc + Project', color: 'purple', desc: 'Aggregated encoder information flows into decoder stream' },
  ]},
  trans_pos_enc_enc: { title: 'Inside Sinusoidal Positional Encoding', steps: [
    { label: 'Position Index pos ∈ {0, 1, ..., L-1}', color: 'io', desc: 'Integer position of each token in source sequence' },
    { label: 'sin(pos / 10000^(2i/d))', color: 'orange', desc: 'Even dimensions: sine with geometrically spaced wavelengths' },
    { label: 'cos(pos / 10000^(2i/d))', color: 'blue', desc: 'Odd dimensions: cosine with same frequencies' },
    { label: 'PE Vector ∈ ℝ^d_model', color: 'yellow', desc: 'Interleaved sin/cos creates unique position fingerprint' },
    { label: 'X_emb + PE → Position-Aware Embeddings', color: 'green', desc: 'Element-wise addition (not concatenation) — preserves dimension' },
  ]},
  trans_pos_enc_dec: { title: 'Inside Decoder Positional Encoding', steps: [
    { label: 'Target Position Index', color: 'io', desc: 'Position in target/output sequence' },
    { label: 'sin/cos Frequency Encoding', color: 'orange', desc: 'Same sinusoidal formula as encoder PE' },
    { label: 'Target_emb + PE', color: 'green', desc: 'Injected into target token embeddings' },
  ]},
  trans_enc_ffn: { title: 'Inside Position-Wise Feed-Forward Network', steps: [
    { label: 'Input x (d_model = 512)', color: 'io', desc: 'Per-position vector from attention sublayer' },
    { label: 'Linear W₁: 512 → 2048', color: 'purple', desc: 'First expansion layer (4× wider)' },
    { label: 'ReLU: max(0, x)', color: 'green', desc: 'Introduces non-linearity — sparsifies activations' },
    { label: 'Linear W₂: 2048 → 512', color: 'purple', desc: 'Compress back to model dimension' },
    { label: 'Dropout (p=0.1)', color: 'yellow', desc: 'Regularization applied during training' },
  ]},
  trans_dec_ffn: { title: 'Inside Decoder FFN', steps: [
    { label: 'Input from Cross-Attn Add&Norm', color: 'io', desc: 'Output of cross-attention residual block' },
    { label: 'Linear → d_ff=2048', color: 'purple', desc: '4× expansion' },
    { label: 'ReLU', color: 'green', desc: 'Element-wise ReLU activation' },
    { label: 'Linear → d_model=512', color: 'purple', desc: 'Project back' },
  ]},
  trans_enc_add_norm1: { title: 'Inside Add & Norm (Post-Attention)', steps: [
    { label: 'Pre-Attention Input X', color: 'io', desc: 'Skip connection source' },
    { label: 'MultiHead(X) Output', color: 'orange', desc: 'Attention sublayer output' },
    { label: 'Residual: X + MultiHead(X)', color: 'yellow', desc: 'Identity shortcut preserves gradient flow' },
    { label: 'LayerNorm: (z-μ)/√(σ²+ε) · γ + β', color: 'green', desc: 'Normalize across d_model dimension' },
  ]},
  trans_enc_add_norm2: { title: 'Inside Add & Norm (Post-FFN)', steps: [
    { label: 'Post-Attention Output', color: 'io', desc: 'Input from first Add&Norm' },
    { label: 'FFN(X) Output', color: 'blue', desc: 'Feed-forward network output' },
    { label: 'X + FFN(X)', color: 'yellow', desc: 'Residual addition' },
    { label: 'LayerNorm', color: 'green', desc: 'Channel-wise normalization' },
  ]},
  trans_dec_add_norm1: { title: 'Inside Decoder Add & Norm 1', steps: [
    { label: 'Input + Masked MHA', color: 'yellow', desc: 'Residual around masked self-attention' },
    { label: 'LayerNorm', color: 'green', desc: 'Normalize for cross-attention' },
  ]},
  trans_dec_add_norm2: { title: 'Inside Decoder Add & Norm 2', steps: [
    { label: 'Cross-Attn Input + Cross-Attn Output', color: 'yellow', desc: 'Residual around cross-attention' },
    { label: 'LayerNorm', color: 'green', desc: 'Normalize for FFN' },
  ]},
  trans_dec_add_norm3: { title: 'Inside Decoder Add & Norm 3', steps: [
    { label: 'FFN Input + FFN Output', color: 'yellow', desc: 'Residual around decoder FFN' },
    { label: 'LayerNorm', color: 'green', desc: 'Final normalization in decoder block' },
  ]},
  trans_input_emb: { title: 'Inside Input Embedding', steps: [
    { label: 'Token IDs ∈ {0..V-1}', color: 'io', desc: 'Integer indices into vocabulary' },
    { label: 'Embedding Lookup: W_e[token_id]', color: 'pink', desc: 'Learnable V × d_model weight matrix' },
    { label: 'Scale by √d_model', color: 'yellow', desc: 'Prevent embeddings from being too small relative to PE' },
  ]},
  trans_output_emb: { title: 'Inside Output Embedding', steps: [
    { label: 'Target Token IDs', color: 'io', desc: 'Shifted-right target sequence' },
    { label: 'Shared W_e Lookup', color: 'pink', desc: 'Weight-tied with input embedding' },
    { label: 'Scale √d_model', color: 'yellow', desc: 'Same scaling factor' },
  ]},
  trans_softmax: { title: 'Inside Softmax Output', steps: [
    { label: 'Raw Logits (V dimensions)', color: 'io', desc: 'Unnormalized scores per vocab token' },
    { label: 'exp(z_i) for each logit', color: 'orange', desc: 'Exponentiate' },
    { label: '/ Σ exp(z_j)', color: 'green', desc: 'Normalize to probability distribution' },
    { label: 'P(next token | context)', color: 'yellow', desc: 'Valid probability distribution over vocabulary' },
  ]},
  trans_linear: { title: 'Inside Linear Projection Head', steps: [
    { label: 'Decoder Final Output (d_model)', color: 'io', desc: 'Last decoder layer representation' },
    { label: 'W_v^T: d_model → V', color: 'purple', desc: 'Project to vocabulary size (weight-tied with embedding)' },
    { label: 'Logits ∈ ℝ^V', color: 'yellow', desc: 'One score per vocabulary token' },
  ]},

  // ── MAMBA ──
  mamba_expand_split: { title: 'Inside Linear Expansion & Split', steps: [
    { label: 'Input x ∈ ℝ^D', color: 'io', desc: 'Token with model dimension D' },
    { label: 'Linear Projection: D → 2E', color: 'purple', desc: 'Expand to 2× inner dimension' },
    { label: 'Channel Split → Branch A, Branch B', color: 'yellow', desc: 'First E channels → SSM, second E → Gate' },
  ]},
  mamba_conv1d: { title: 'Inside 1D Causal Convolution', steps: [
    { label: 'SSM Branch Input (E dims)', color: 'io', desc: 'E-dimensional token sequence' },
    { label: 'Causal Padding (k-1 zeros on left)', color: 'yellow', desc: 'Ensure output at time t only depends on t and earlier' },
    { label: '1D Conv (kernel_size=4)', color: 'blue', desc: 'Local context mixing over 4 adjacent positions' },
    { label: 'Locally-Mixed Features', color: 'green', desc: 'Short-range dependencies captured before SSM' },
  ]},
  mamba_silu1: { title: 'Inside SiLU (Swish) Activation', steps: [
    { label: 'Post-Conv Features x', color: 'io', desc: 'Convolution output' },
    { label: 'σ(x) = 1/(1+e^(-x))', color: 'green', desc: 'Sigmoid produces smooth gate ∈ (0,1)' },
    { label: 'SiLU = x · σ(x)', color: 'orange', desc: 'Self-gated: negative values suppressed, positive amplified' },
  ]},
  mamba_param_gen: { title: 'Inside Selective Parameter Generation', steps: [
    { label: 'Activated Features', color: 'io', desc: 'Post-SiLU features' },
    { label: 'Linear_B → B(t) ∈ ℝ^N', color: 'orange', desc: 'Input-dependent state input matrix' },
    { label: 'Linear_C → C(t) ∈ ℝ^N', color: 'orange', desc: 'Input-dependent state output matrix' },
    { label: 'Linear_Δ + Softplus → Δ(t)', color: 'orange', desc: 'Input-dependent step size (controls discretization rate)' },
  ]},
  mamba_discretize_scan: { title: 'Inside ZOH Discretization & Parallel Scan', steps: [
    { label: 'Continuous SSM Params (A, B, C, Δ)', color: 'io', desc: 'A is learned diagonal, B,C,Δ are input-dependent' },
    { label: 'ZOH: Ā = exp(ΔA)', color: 'orange', desc: 'Zero-order hold discretizes continuous A' },
    { label: 'B̄ = (ΔA)⁻¹(exp(ΔA) - I) · ΔB', color: 'orange', desc: 'Discretize B using ZOH' },
    { label: 'Parallel Scan: hₜ = Āhₜ₋₁ + B̄xₜ', color: 'blue', desc: 'GPU-efficient O(L log L) associative parallel prefix sum' },
    { label: 'Readout: yₜ = C · hₜ', color: 'green', desc: 'Project hidden state through C' },
  ]},
  mamba_silu2: { title: 'Inside Gate SiLU', steps: [
    { label: 'Gate Branch Input (E dims)', color: 'io', desc: 'Second half of expansion' },
    { label: 'SiLU: x · σ(x)', color: 'green', desc: 'Smooth gating signal for multiplicative control' },
  ]},
  mamba_mult_gate: { title: 'Inside Hadamard Product Gating', steps: [
    { label: 'SSM Output (E dims)', color: 'io', desc: 'Sequence-modeled left branch' },
    { label: 'Gate Signal (E dims)', color: 'io', desc: 'Activated right branch' },
    { label: 'y = SSM_out ⊙ Gate', color: 'yellow', desc: 'Element-wise multiplication — gate controls information flow' },
  ]},
  mamba_linear_out: { title: 'Inside Output Projection', steps: [
    { label: 'Gated Output (E dims)', color: 'io', desc: 'Post-gating' },
    { label: 'Linear: E → D', color: 'purple', desc: 'Compress back to model dim' },
  ]},
  mamba_res_add: { title: 'Inside Residual Addition', steps: [
    { label: 'Mamba Block Output', color: 'io', desc: 'Processed output' },
    { label: 'Original Input x (skip)', color: 'pink', desc: 'Identity skip connection' },
    { label: 'Output = Block + x', color: 'yellow', desc: 'Gradient highway for deep stacking' },
  ]},

  // ── MIXTURE OF DEPTHS (MoD) ──
  mod_router: { title: 'Inside Per-Token Router MLP', steps: [
    { label: 'Token Representation x_i ∈ ℝ^D', color: 'io', desc: 'Input token from previous layer' },
    { label: 'Linear Weight w_r ∈ ℝ^D', color: 'purple', desc: 'Learned routing projection vector' },
    { label: 'Scalar Routing Score: r_i = w_rᵀ x_i', color: 'orange', desc: 'Unconstrained scalar predicting token computation importance' },
    { label: 'Sigmoid / Softmax Normalization', color: 'green', desc: 'Produces differentiable routing probability' },
  ]},
  mod_topk: { title: 'Inside Top-K Capacity Filter', steps: [
    { label: 'Sequence Scores {r_1, ..., r_T}', color: 'io', desc: 'All token routing scores in sequence' },
    { label: 'Capacity Calculation: C = ⌊c · T⌋', color: 'yellow', desc: 'c is user capacity (e.g. 0.125 = 12.5% tokens)' },
    { label: 'Top-K Selection Algorithm', color: 'orange', desc: 'Finds top C highest scoring tokens along sequence dimension' },
    { label: 'Index Mask & Sorting', color: 'green', desc: 'Preserves spatial order of selected tokens' },
  ]},
  mod_gather: { title: 'Inside Tensor Gather Operation', steps: [
    { label: 'Selected Token Indices I ⊂ {1..T}', color: 'io', desc: 'C indices selected by router' },
    { label: 'Gather: X_selected = X[:, I, :]', color: 'blue', desc: 'Packs sparse tokens into dense tensor (B, C, D)' },
    { label: 'Dense Batch Tensor Ready', color: 'green', desc: 'Ensures standard attention/FFN kernels execute at peak FLOP/s' },
  ]},
  mod_compute_block: { title: 'Inside Self-Attention / MLP Compute Block', steps: [
    { label: 'Dense Sub-Tensor (B, C, D)', color: 'io', desc: 'Only Top-K tokens undergo heavy compute' },
    { label: 'Full Multi-Head Attention / FFN', color: 'pink', desc: 'Attention computed over C tokens instead of T (O(C²) compute)' },
    { label: 'Transformed Output Y_selected', color: 'green', desc: 'High-capacity feature transformations' },
  ]},
  mod_scatter_res: { title: 'Inside Scatter + Weighted Residual Skip', steps: [
    { label: 'Computed Tokens Y_selected & Router r_i', color: 'io', desc: 'Output from compute block' },
    { label: 'Scatter to Original Positions in Sequence', color: 'blue', desc: 'Map C tokens back to their T positions' },
    { label: 'Weighted Residual: y_i = r_i · Y_i + x_i', color: 'yellow', desc: 'Selected tokens receive weighted update' },
    { label: 'Identity Bypass: y_j = x_j', color: 'green', desc: 'Unselected tokens bypass completely with 0 compute FLOPs' },
  ]},

  // ── MIXTURE OF EXPERTS (MoE) ──
  moe_router: { title: 'Inside Gating Router Network', steps: [
    { label: 'Input Token x ∈ ℝ^D', color: 'io', desc: 'Incoming token representation' },
    { label: 'Gating Matrix W_g ∈ ℝ^(D × N_experts)', color: 'purple', desc: 'Linear projection to expert logits' },
    { label: 'Noisy Top-K Gating: H(x) = W_g x + Noise', color: 'orange', desc: 'Exploration noise added during training' },
    { label: 'Softmax over Top-k selected experts', color: 'green', desc: 'Sparse gating weights summing to 1 across active experts' },
  ]},
  moe_shared_expert: { title: 'Inside Dedicated Shared Expert', steps: [
    { label: 'Token x (All Tokens)', color: 'io', desc: 'Every token passes through shared expert unconditionally' },
    { label: 'Shared Dense FFN', color: 'pink', desc: 'Learns common invariant knowledge across domains' },
    { label: 'Reduces Redundancy in Routed Experts', color: 'green', desc: 'Allows routed experts to specialize aggressively' },
  ]},
  moe_expert_bank: { title: 'Inside Sparse Expert Bank', steps: [
    { label: 'Dispatched Tokens per Expert', color: 'io', desc: 'Tokens grouped by assigned expert index' },
    { label: 'Parallel Expert Computation (E_1 .. E_N)', color: 'blue', desc: 'Each expert is an independent 2-layer FFN' },
    { label: 'High Capacity Parameter Bank', color: 'green', desc: 'Trillions of parameters active with only billions of FLOPs' },
  ]},
  moe_weighted_sum: { title: 'Inside Weighted Expert Combination', steps: [
    { label: 'Expert Outputs E_i(x) & Gating Weights G(x)_i', color: 'io', desc: 'Outputs from k active experts' },
    { label: 'Weighted Sum: Σ G(x)_i · E_i(x)', color: 'yellow', desc: 'Linear combination of expert responses' },
    { label: 'Add Shared Expert: + E_shared(x)', color: 'green', desc: 'Fused contextual output representation' },
  ]},

  // ── SLIDING WINDOW ATTENTION (SWAT) ──
  swat_rope_alibi: { title: 'Inside Dual Positional Engine (RoPE + ALiBi)', steps: [
    { label: 'Query & Key Vectors', color: 'io', desc: 'Unpositioned Q, K projections' },
    { label: 'Rotary Position Embedding (RoPE)', color: 'pink', desc: 'Complex plane 2D rotations for relative phase shift' },
    { label: 'Balanced ALiBi Linear Slope Bias', color: 'orange', desc: 'Injects -m·|i-j| distance penalty for long-range stability' },
    { label: 'Dual-Engine Out-of-Distribution Extrapolation', color: 'green', desc: 'Enables generalization to 100k+ token sequences' },
  ]},
  swat_sigmoid_attn: { title: 'Inside Independent Sigmoid Attention', steps: [
    { label: 'Rotated Dot-Product QKᵀ / √d', color: 'io', desc: 'Attention score matrix within window W' },
    { label: 'Independent Sigmoid: σ(score - bias)', color: 'orange', desc: 'Calculates independent probability per pair instead of Softmax' },
    { label: 'Prevents Softmax Probability Dilution', color: 'green', desc: 'Long sequences do not suffer from entropy collapse' },
  ]},
  swat_sliding_cache: { title: 'Inside Sliding Window KV Ring Buffer', steps: [
    { label: 'New Token KV Generated', color: 'io', desc: 'Current step key and value vectors' },
    { label: 'Circular Ring Buffer Update (Size W)', color: 'blue', desc: 'Overwrites oldest token past index (t - W)' },
    { label: 'Constant Memory Footprint O(W)', color: 'green', desc: 'Inference memory stays strictly bounded regardless of generation length' },
  ]},
  swat_stacked_rf: { title: 'Inside Multi-Scale Layer Receptive Field', steps: [
    { label: 'Layer 1 Receptive Field: W tokens', color: 'yellow', desc: 'Direct window coverage' },
    { label: 'Layer 2 Receptive Field: 2W tokens', color: 'blue', desc: 'Composition of overlapping windows' },
    { label: 'Layer N Receptive Field: N × W tokens', color: 'green', desc: 'Full sequence receptive field achieved at top layers' },
  ]},

  // ── TITANS (Test-Time Learning) ──
  titans_nmm_loss: { title: 'Inside Associative Memory Loss Engine', steps: [
    { label: 'Current Context Input x_t', color: 'io', desc: 'Token streaming in' },
    { label: 'Key & Value Generation: k_t = W_k x_t, v_t = W_v x_t', color: 'pink', desc: 'Memory association targets' },
    { label: 'Reconstruction Loss: ‖M_t(k_t) - v_t‖²', color: 'orange', desc: 'Measures how accurately neural memory recalls current token' },
  ]},
  titans_surprise_grad: { title: 'Inside Surprise Gradient Engine', steps: [
    { label: 'Memory Loss ℒ_mem', color: 'io', desc: 'Prediction error on token' },
    { label: 'Compute Gradient: g_t = ∇_{M} ℒ_mem', color: 'orange', desc: 'Surprise metric — how unexpected is this token?' },
    { label: 'High Surprise = Major Memory Update', color: 'green', desc: 'Important unexpected facts trigger strong weight updates' },
  ]},
  titans_momentum_decay: { title: 'Inside Surprise Momentum & Dynamic Decay', steps: [
    { label: 'Raw Surprise Gradient g_t', color: 'io', desc: 'Instantaneous surprise' },
    { label: 'Momentum Accumulation: S_t = η_t S_{t-1} - θ_t g_t', color: 'yellow', desc: 'Smooths memory updates over multi-token concepts' },
    { label: 'Adaptive Forgetting Factor (1 - α_t)', color: 'green', desc: 'Fades transient noise while locking persistent knowledge' },
  ]},
  titans_deep_mlp: { title: 'Inside Deep Neural Long-Term Memory (NMM)', steps: [
    { label: 'Neural Memory Weights M_t', color: 'io', desc: 'Deep Multi-Layer Perceptron acting as dynamic storage' },
    { label: 'Test-Time Weight Update: M_{t+1} = M_t + S_t', color: 'blue', desc: 'Model parameters adapt in real time during inference' },
    { label: 'Infinite Context Recall Capacity', color: 'green', desc: 'Compresses millions of tokens into fixed-size neural weights' },
  ]},
  titans_sw_attn: { title: 'Inside Short-Term Sliding Attention', steps: [
    { label: 'Recent Local Context (W tokens)', color: 'io', desc: 'Immediate conversational tokens' },
    { label: 'Fast Exact Attention', color: 'orange', desc: 'Provides pinpoint precision for immediate syntax and local grammar' },
  ]},
  titans_hyperhead: { title: 'Inside Hyper-Head Integration (MAC/MAG/MAL)', steps: [
    { label: 'Long-Term Memory Output + Short-Term Attention', color: 'io', desc: 'Two memory streams converge' },
    { label: 'Memory as Context (MAC) / Gating (MAG)', color: 'green', desc: 'Dynamic gating mechanism balances persistent facts with local prompt' },
  ]},

  // ── TRANSMAMBA (Hybrid) ──
  tm_shared_w: { title: 'Inside Unified Shared Projection Matrix', steps: [
    { label: 'Input Token Stream X', color: 'io', desc: 'Sequence of tokens' },
    { label: 'Shared Matrix W_proj (Attention & SSM)', color: 'pink', desc: 'Same weights compute Attention Q/K/V and Mamba B/C/Δ' },
    { label: 'Zero-Redundancy Parameter Efficiency', color: 'green', desc: 'Dual-mode capabilities without inflating model parameter footprint' },
  ]},
  tm_transpoint_router: { title: 'Inside Sequence TransPoint Scheduler', steps: [
    { label: 'Sequence Token Index t', color: 'io', desc: 'Current generation position' },
    { label: 'TransPoint Threshold N_trans (e.g. 512 tokens)', color: 'yellow', desc: 'Boundary between prompt encoding and long-form generation' },
    { label: 'Dynamically Routes Mode Execution', color: 'green', desc: 'Decides when to transition from Attention to Mamba scan' },
  ]},
  tm_attn_mode: { title: 'Inside Transformer Attention Regime (t ≤ N_trans)', steps: [
    { label: 'Prompt Tokens [x_1 ... x_{N_trans}]', color: 'io', desc: 'Initial prompt context' },
    { label: 'Full Bidirectional Attention Processing', color: 'orange', desc: 'Captures dense pairwise relations for maximum reasoning fidelity' },
  ]},
  tm_mcc_bridge: { title: 'Inside Memory Compression Cache (MCC)', steps: [
    { label: 'Dense Key-Value Cache at TransPoint', color: 'io', desc: 'KV representations of the prompt' },
    { label: 'Nonlinear Compression Projection', color: 'blue', desc: 'Compresses KV matrices into initial Mamba hidden state h_0' },
    { label: 'Discards Heavy KV Cache', color: 'green', desc: 'Frees 90%+ of GPU VRAM before generation begins' },
  ]},
  tm_ssm_mode: { title: 'Inside Mamba2 SSM Linear Scan (t > N_trans)', steps: [
    { label: 'Initial State h_0 + New Generated Tokens', color: 'io', desc: 'Seamless handover from Attention regime' },
    { label: 'Linear Time Recurrent Scan', color: 'orange', desc: 'O(1) inference speed per generated token' },
    { label: 'High Speed & Low Memory Output Stream', color: 'green', desc: 'Combines Transformer quality with Mamba throughput' },
  ]}
};


// ──────────────────────────────────────────────────────────────────
// DRILL-DOWN NODE HANDLER (Modal Trigger)
// ──────────────────────────────────────────────────────────────────
function drillDownNode(paperId, nodeId) {
  const paper = papers.find(p => p.paper_id === paperId);
  if (!paper || !paper.graphical_diagram) return;

  const node = paper.graphical_diagram.nodes.find(n => n.id === nodeId);
  if (!node) return;

  modalTitle.textContent = node.label;
  modalLayerBadge.textContent = `${paper.title.split(':')[0]} • Component Deep-Dive`;
  modalCircuitText.textContent = node.description || "Internal layer computation circuit.";
  modalUndertonesText.textContent = node.undertones || "No specific undertone noted for this layer.";

  // Find sublayer formula & citation
  let formula = null, citation = "Paper Primary Source";
  if (node.layer_idx !== undefined && paper.layers[node.layer_idx]) {
    const sub = paper.layers[node.layer_idx].sublayers[node.sublayer_idx || 0];
    if (sub) {
      formula = sub.formula_katex;
      citation = sub.source_section;
    }
  }

  if (formula) {
    modalFormulaSection.style.display = 'block';
    modalFormulaContent.innerHTML = `$$\\displaystyle ${formula}$$`;
  } else {
    modalFormulaSection.style.display = 'none';
  }

  modalCitationTag.textContent = citation;
  modalCitationTag.onclick = () => showCitation(node.label, citation);

  renderInnerCircuitDiagram(nodeId);
  openModal();
}

function renderInnerCircuitDiagram(nodeId) {
  const container = modalInnerDiagramContainer;
  if (!container) return;
  container.innerHTML = '';

  const circuit = INNER_CIRCUITS[nodeId];
  if (!circuit) {
    container.innerHTML = `
      <div class="inner-diagram-empty">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
        </svg>
        <span>Internal mechanism step flow diagram not yet mapped for this component.</span>
      </div>`;
    return;
  }

  const colorMap = {
    io: 'inner-io', pink: 'inner-pink', orange: 'inner-orange',
    yellow: 'inner-yellow', blue: 'inner-blue', purple: 'inner-purple', green: 'inner-green'
  };

  let html = `<h4 class="inner-diagram-title">${circuit.title}</h4><div class="inner-circuit-flow">`;
  circuit.steps.forEach((step, i) => {
    html += `
      <div class="inner-circuit-step ${colorMap[step.color] || 'inner-io'}">
        <div class="inner-step-label">${step.label}</div>
        <div class="inner-step-desc">${step.desc}</div>
      </div>`;
    if (i < circuit.steps.length - 1) {
      html += `
        <div class="inner-circuit-arrow">
          <svg width="16" height="20" viewBox="0 0 16 20">
            <path d="M8 0 L8 14 M3 10 L8 16 L13 10" stroke="currentColor" stroke-width="2" fill="none"/>
          </svg>
        </div>`;
    }
  });
  html += `</div>`;
  container.innerHTML = html;
}


// ──────────────────────────────────────────────────────────────────
// VIEW 2: VIBRANT & DEEP MICRO-LAYERS BREAKDOWN
// ──────────────────────────────────────────────────────────────────
function renderMicroLayers(paper) {
  microSectionTitle.textContent = `${paper.title.split(':')[0]} — Micro-Layer Breakdown & Math Catalog`;
  layerCardsGrid.innerHTML = '';

  paper.layers.forEach((layer, lIdx) => {
    const card = document.createElement('div');
    card.className = `layer-accordion-card ${lIdx === 0 ? 'active' : ''}`;
    card.id = `layerCard_${lIdx}`;

    const tIn = formatConcreteTensorShape(layer.tensor_in || "(B, L, 512)");
    const tOut = formatConcreteTensorShape(layer.tensor_out || "(B, L, 512)");
    const flopsW = layer.flops_weight || "O(L · D)";
    const memW = layer.memory_weight || "O(B · L · D)";
    const isLinear = flopsW.includes('O(L ·') || flopsW.includes('O(W ·') || flopsW.includes('O(c ·');

    // Generate flowchart steps
    let flowchartHtml = `<div class="layer-pipeline-flowchart">`;
    layer.sublayers.forEach((sub, sIdx) => {
      flowchartHtml += `
        <div class="flowchart-step-chip">
          <span class="flowchart-step-num">${sIdx + 1}</span>
          <span>${sub.name}</span>
        </div>`;
      if (sIdx < layer.sublayers.length - 1) {
        flowchartHtml += `<span class="flowchart-arrow-icon">➔</span>`;
      }
    });
    flowchartHtml += `</div>`;

    // Render Sublayer Cards
    let sublayersHtml = '';
    layer.sublayers.forEach((sub, sIdx) => {
      const subInShape = formatConcreteTensorShape(sub.in_shape || layer.tensor_in);
      const subOutShape = formatConcreteTensorShape(sub.out_shape || layer.tensor_out);
      const subUid = `sub_${lIdx}_${sIdx}`;

      // Hyperparameters HTML
      let hyperHtml = '';
      if (sub.hyperparameters && sub.hyperparameters.length > 0) {
        hyperHtml = `
          <div class="hyperparams-grid">
            ${sub.hyperparameters.map(h => `
              <div class="hyperparam-card">
                <span class="hyperparam-key">${h.param}:</span>
                <span class="hyperparam-val">${h.val}</span>
                <span class="hyperparam-desc">${h.desc}</span>
              </div>
            `).join('')}
          </div>`;
      } else {
        hyperHtml = `<div style="font-size:0.8rem;color:var(--text-muted)">Standard architectural hyperparameters applied.</div>`;
      }

      // PyTorch Code HTML
      const codeHtml = sub.pytorch_code ? `
        <div class="pytorch-code-wrapper">
          <div class="pytorch-code-header">
            <span>PyTorch 2.0 Module Implementation</span>
            <button class="copy-formula-btn" onclick="copyToClipboard('${sub.pytorch_code.replace(/\n/g, '\\n').replace(/'/g, "\\'")}', this)">Copy Code</button>
          </div>
          <pre class="pytorch-code-pre"><code>${escapeHtml(sub.pytorch_code)}</code></pre>
        </div>` : `<div style="font-size:0.8rem;color:var(--text-muted)">Standard layer implementation in deep learning framework.</div>`;

      // Sublayer Tabs HTML
      const tabsNavHtml = `
        <div class="sublayer-tabs-container">
          <div class="sublayer-tabs-nav">
            <button class="sublayer-tab-btn active" onclick="switchSublayerTab('${subUid}', 'shapes', this)">📐 Shapes</button>
            <button class="sublayer-tab-btn" onclick="switchSublayerTab('${subUid}', 'params', this)">⚙️ Hyperparams</button>
            <button class="sublayer-tab-btn" onclick="switchSublayerTab('${subUid}', 'hardware', this)">⚡ Hardware</button>
            <button class="sublayer-tab-btn" onclick="switchSublayerTab('${subUid}', 'stability', this)">⚠️ Stability</button>
            <button class="sublayer-tab-btn" onclick="switchSublayerTab('${subUid}', 'code', this)">💻 PyTorch</button>
          </div>

          <div class="sublayer-tab-content" id="${subUid}_shapes">
            <div class="hyperparams-grid">
              <div class="hyperparam-card">
                <span class="hyperparam-key">Input:</span>
                <span class="hyperparam-val" style="color:var(--accent-cyan)">${subInShape}</span>
              </div>
              <div class="hyperparam-card">
                <span class="hyperparam-key">Output:</span>
                <span class="hyperparam-val" style="color:var(--accent-emerald)">${subOutShape}</span>
              </div>
            </div>
            <div style="font-size:0.8rem;color:var(--text-secondary);margin-top:8px;">
              ${sub.math_breakdown || sub.explanation}
            </div>
          </div>

          <div class="sublayer-tab-content" id="${subUid}_params" style="display:none;">
            ${hyperHtml}
          </div>

          <div class="sublayer-tab-content" id="${subUid}_hardware" style="display:none;">
            <div class="info-callout-box hardware-box">
              <strong>GPU Kernel & Memory Bandwidth:</strong><br>
              ${sub.hardware_notes || "Standard memory access pattern with standard GPU SRAM tile streaming."}
            </div>
          </div>

          <div class="sublayer-tab-content" id="${subUid}_stability" style="display:none;">
            <div class="info-callout-box failure-box">
              <strong>Potential Failure Modes & Mitigations:</strong><br>
              ${sub.failure_modes || "Stable under normal training hyperparameters and standard learning rate warmup."}
            </div>
          </div>

          <div class="sublayer-tab-content" id="${subUid}_code" style="display:none;">
            ${codeHtml}
          </div>
        </div>`;

      // Hero Formula box
      const formulaHero = sub.formula_katex ? `
        <div class="sublayer-formula-hero">
          <div class="formula-header-bar">
            <span class="formula-tag-label">Mathematical Transformation</span>
            <button class="copy-formula-btn" onclick="copyToClipboard('${sub.formula_katex.replace(/\\/g, '\\\\').replace(/'/g, "\\'")}', this)">Copy Formula</button>
          </div>
          <div class="formula-math-display">$$\\displaystyle ${sub.formula_katex}$$</div>
        </div>` : '';

      sublayersHtml += `
        <div class="sublayer-card">
          <div class="sublayer-top-row">
            <div class="sublayer-name">
              <span class="sublayer-step-dot"></span>
              <span>${lIdx + 1}.${sIdx + 1} ${sub.name}</span>
            </div>
            <div class="sublayer-tags-row">
              <span class="tensor-transform-badge">${subInShape} ➔ ${subOutShape}</span>
              <button class="source-citation-badge" onclick="showCitation('${sub.name.replace(/'/g, "\\'")}','${sub.source_section.replace(/'/g, "\\'")}')" title="View Citation">
                <span>📖</span><span>${sub.source_section}</span>
              </button>
            </div>
          </div>

          ${formulaHero}

          <p class="sublayer-breakdown-text">${sub.explanation}</p>

          ${tabsNavHtml}
        </div>`;
    });

    card.innerHTML = `
      <div class="layer-accordion-header" onclick="toggleLayerCard(${lIdx})">
        <div class="layer-header-left">
          <span class="layer-index-badge">${lIdx + 1}</span>
          <div>
            <div class="layer-title-text">${layer.layer_name}</div>
            <div style="font-size:0.75rem;color:var(--text-muted);font-family:var(--font-mono)">${layer.sublayers.length} Sub-operation${layer.sublayers.length > 1 ? 's' : ''}</div>
          </div>
        </div>
        <div class="layer-header-badges">
          <span class="tensor-flow-tag">${tIn} ➔ ${tOut}</span>
          <span class="complexity-tag ${isLinear ? 'linear-tag' : ''}">${flopsW}</span>
          <span class="tensor-flow-tag" style="color:var(--text-accent)">${memW}</span>
          <svg class="chevron-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
            <polyline points="6 9 12 15 18 9"></polyline>
          </svg>
        </div>
      </div>
      <div class="layer-accordion-body">
        ${flowchartHtml}
        ${sublayersHtml}
      </div>`;

    layerCardsGrid.appendChild(card);
  });

  triggerKaTeX();
}

function toggleLayerCard(i) {
  const c = document.getElementById(`layerCard_${i}`);
  if (c) c.classList.toggle('active');
}

function switchSublayerTab(subUid, tabName, btn) {
  const container = btn.closest('.sublayer-tabs-container');
  if (!container) return;

  // Toggle active tab button
  container.querySelectorAll('.sublayer-tab-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');

  // Toggle panes
  const allPanes = ['shapes', 'params', 'hardware', 'stability', 'code'];
  allPanes.forEach(p => {
    const el = document.getElementById(`${subUid}_${p}`);
    if (el) el.style.display = p === tabName ? 'block' : 'none';
  });
}

function copyToClipboard(text, btn) {
  navigator.clipboard.writeText(text).then(() => {
    const original = btn.textContent;
    btn.textContent = 'Copied! ✓';
    setTimeout(() => { btn.textContent = original; }, 1500);
  }).catch(() => {
    btn.textContent = 'Copied!';
    setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
  });
}

function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}


// ──────────────────────────────────────────────────────────────────
// VIEW 3: PROS & CONS VIEW
// ──────────────────────────────────────────────────────────────────
function renderProsCons(paper) {
  prosList.innerHTML = '';
  paper.pros.forEach(p => {
    const d = document.createElement('div');
    d.className = 'pro-con-item';
    d.innerHTML = `
      <div class="pro-con-claim">${p.claim}</div>
      <button class="source-citation-badge" onclick="showCitation('Advantage','${p.source_section.replace(/'/g, "\\'")}')">
        <span>📖</span><span>${p.source_section}</span>
      </button>`;
    prosList.appendChild(d);
  });

  consList.innerHTML = '';
  paper.cons.forEach(c => {
    const d = document.createElement('div');
    d.className = 'pro-con-item';
    d.innerHTML = `
      <div class="pro-con-claim">${c.claim}</div>
      <button class="source-citation-badge" onclick="showCitation('Limitation','${c.source_section.replace(/'/g, "\\'")}')">
        <span>📖</span><span>${c.source_section}</span>
      </button>`;
    consList.appendChild(d);
  });
}


// ──────────────────────────────────────────────────────────────────
// VIEW 4: REPORTED BENCHMARK METRICS
// ──────────────────────────────────────────────────────────────────
function renderReportedMetrics(paper) {
  reportedMetricsTableBody.innerHTML = '';
  paper.reported_metrics.forEach(m => {
    const r = document.createElement('tr');
    r.innerHTML = `
      <td style="font-weight:600;color:var(--text-primary)">${m.task}</td>
      <td>${m.metric_name}</td>
      <td><span class="score-badge">${m.score_value}</span></td>
      <td>
        <button class="source-citation-badge" onclick="showCitation('${m.task.replace(/'/g, "\\'")}','${m.source_section.replace(/'/g, "\\'")}')">
          <span>📖</span><span>${m.source_section}</span>
        </button>
      </td>
      <td style="font-size:0.8rem;color:var(--text-muted)">${m.comparison_context}</td>`;
    reportedMetricsTableBody.appendChild(r);
  });
}


// ──────────────────────────────────────────────────────────────────
// PHASE 3: CROSS-MODEL COMPARISON ENGINE
// ──────────────────────────────────────────────────────────────────
function initComparisonSelectors() {
  if (!compareModelASelect || !compareModelBSelect) return;
  compareModelASelect.innerHTML = '';
  compareModelBSelect.innerHTML = '';

  papers.forEach(p => {
    const optA = document.createElement('option');
    optA.value = p.paper_id;
    optA.textContent = p.title.split(':')[0];
    compareModelASelect.appendChild(optA);

    const optB = document.createElement('option');
    optB.value = p.paper_id;
    optB.textContent = p.title.split(':')[0];
    compareModelBSelect.appendChild(optB);
  });

  compareModelASelect.value = 'transformer';
  compareModelBSelect.value = 'mamba';

  compareModelASelect.addEventListener('change', renderComparisonView);
  compareModelBSelect.addEventListener('change', renderComparisonView);
}

function setComparisonPreset(mA, mB) {
  if (compareModelASelect && compareModelBSelect) {
    compareModelASelect.value = mA;
    compareModelBSelect.value = mB;
    renderComparisonView();
  }
}

function renderComparisonView() {
  if (!comparisonBody) return;
  const pAId = compareModelASelect.value;
  const pBId = compareModelBSelect.value;
  const pA = papers.find(p => p.paper_id === pAId) || papers[0];
  const pB = papers.find(p => p.paper_id === pBId) || papers[1];

  const nameA = pA.title.split(':')[0];
  const nameB = pB.title.split(':')[0];

  let html = `
    <!-- Executive Breakthrough Comparison -->
    <div class="comp-summary-grid">
      <div class="comp-model-card">
        <div class="comp-model-header">
          <h3 class="comp-model-title">${nameA}</h3>
          <span class="tag-badge">${pA.computed_metric_role.toUpperCase()}</span>
        </div>
        <p class="comp-breakthrough-text"><strong>Core Breakthrough:</strong> ${pA.core_breakthrough}</p>
        <div style="font-size:0.8rem;color:var(--text-muted)">${pA.authors_venue_year}</div>
      </div>

      <div class="comp-model-card">
        <div class="comp-model-header">
          <h3 class="comp-model-title">${nameB}</h3>
          <span class="tag-badge">${pB.computed_metric_role.toUpperCase()}</span>
        </div>
        <p class="comp-breakthrough-text"><strong>Core Breakthrough:</strong> ${pB.core_breakthrough}</p>
        <div style="font-size:0.8rem;color:var(--text-muted)">${pB.authors_venue_year}</div>
      </div>
    </div>

    <!-- Comparative Layer-by-Layer Architectural Mapping -->
    <div class="layer-map-card">
      <h3 class="comp-sub-title">Layer-by-Layer Architectural Alignment</h3>
      <p class="comp-sub-desc">Direct structural mapping of functional building blocks across the two architectures.</p>
      
      <div class="reported-metrics-table-wrapper">
        <table class="comp-mapping-table">
          <thead>
            <tr>
              <th style="width:20%">Functional Stage</th>
              <th style="width:40%">${nameA} Layer Mechanism</th>
              <th style="width:40%">${nameB} Layer Mechanism</th>
            </tr>
          </thead>
          <tbody>
            ${generateLayerMappingRows(pA, pB)}
          </tbody>
        </table>
      </div>
    </div>

    <!-- Advantages vs Advantages -->
    <div class="comp-pro-con-grid">
      <div class="comp-col-box">
        <div class="comp-col-header" style="color:var(--accent-emerald)">
          <span>✓</span>
          <span>${nameA} Advantages & Core Strengths</span>
        </div>
        ${pA.pros.map(p => `<div class="comp-pro-item"><strong>${p.claim}</strong><div style="font-size:0.75rem;color:var(--text-muted);margin-top:4px;">Ref: ${p.source_section}</div></div>`).join('')}
      </div>

      <div class="comp-col-box">
        <div class="comp-col-header" style="color:var(--accent-emerald)">
          <span>✓</span>
          <span>${nameB} Advantages & Core Strengths</span>
        </div>
        ${pB.pros.map(p => `<div class="comp-pro-item"><strong>${p.claim}</strong><div style="font-size:0.75rem;color:var(--text-muted);margin-top:4px;">Ref: ${p.source_section}</div></div>`).join('')}
      </div>
    </div>

    <!-- Limitations vs Limitations -->
    <div class="comp-pro-con-grid">
      <div class="comp-col-box">
        <div class="comp-col-header" style="color:var(--accent-rose)">
          <span>✕</span>
          <span>${nameA} Limitations & Bottlenecks</span>
        </div>
        ${pA.cons.map(c => `<div class="comp-con-item"><strong>${c.claim}</strong><div style="font-size:0.75rem;color:var(--text-muted);margin-top:4px;">Ref: ${c.source_section}</div></div>`).join('')}
      </div>

      <div class="comp-col-box">
        <div class="comp-col-header" style="color:var(--accent-rose)">
          <span>✕</span>
          <span>${nameB} Limitations & Bottlenecks</span>
        </div>
        ${pB.cons.map(c => `<div class="comp-con-item"><strong>${c.claim}</strong><div style="font-size:0.75rem;color:var(--text-muted);margin-top:4px;">Ref: ${c.source_section}</div></div>`).join('')}
      </div>
    </div>

    <!-- Computational & Algorithmic Complexity Matrix -->
    <div class="layer-map-card">
      <h3 class="comp-sub-title">Algorithmic & Theoretical Complexity Matrix</h3>
      <div class="reported-metrics-table-wrapper">
        <table class="styled-table">
          <thead>
            <tr>
              <th>Complexity Metric</th>
              <th>${nameA}</th>
              <th>${nameB}</th>
              <th>Architectural Impact</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Training Time Complexity</strong></td>
              <td><span class="score-badge">${getComplexity(pA.paper_id, 'train')}</span></td>
              <td><span class="score-badge">${getComplexity(pB.paper_id, 'train')}</span></td>
              <td style="font-size:0.8rem;color:var(--text-muted)">Determines training throughput on long sequences</td>
            </tr>
            <tr>
              <td><strong>Inference Step Time Complexity</strong></td>
              <td><span class="score-badge">${getComplexity(pA.paper_id, 'step')}</span></td>
              <td><span class="score-badge">${getComplexity(pB.paper_id, 'step')}</span></td>
              <td style="font-size:0.8rem;color:var(--text-muted)">Token-by-token generation latency overhead</td>
            </tr>
            <tr>
              <td><strong>KV Cache / State Memory Space</strong></td>
              <td><span class="score-badge">${getComplexity(pA.paper_id, 'space')}</span></td>
              <td><span class="score-badge">${getComplexity(pB.paper_id, 'space')}</span></td>
              <td style="font-size:0.8rem;color:var(--text-muted)">VRAM requirements during autoregressive rollout</td>
            </tr>
            <tr>
              <td><strong>Associative In-Context Recall</strong></td>
              <td><span class="score-badge">${getComplexity(pA.paper_id, 'recall')}</span></td>
              <td><span class="score-badge">${getComplexity(pB.paper_id, 'recall')}</span></td>
              <td style="font-size:0.8rem;color:var(--text-muted)">Ability to perform multi-hop reasoning and copy tasks</td>
            </tr>
            <tr>
              <td><strong>Long-Context Horizon</strong></td>
              <td><span class="score-badge">${getComplexity(pA.paper_id, 'horizon')}</span></td>
              <td><span class="score-badge">${getComplexity(pB.paper_id, 'horizon')}</span></td>
              <td style="font-size:0.8rem;color:var(--text-muted)">Maximum sequence length without performance cliff</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  `;

  comparisonBody.innerHTML = html;
  triggerKaTeX();
}

function generateLayerMappingRows(pA, pB) {
  const stages = [
    { title: '1. Position & Input Embedding', desc: 'How spatial coordinates & discrete tokens are projected' },
    { title: '2. Core Sequence Mixing Engine', desc: 'The mathematical mechanism routing context across tokens' },
    { title: '3. Channel / Feature Transformation', desc: 'How representations are expanded non-linearly' },
    { title: '4. Residual Highway & Normalization', desc: 'Gradient preservation and stabilization dynamics' },
    { title: '5. Inference State Retention', desc: 'How past context is cached during autoregression' }
  ];

  return stages.map((st, i) => {
    const lA = pA.layers[i] ? `<strong>${pA.layers[i].layer_name}</strong><br><span style="font-size:0.82rem;color:var(--text-secondary)">${pA.layers[i].sublayers.map(s => s.name).join(' → ')}</span>` : '<span style="color:var(--text-muted)">Implicit in architecture</span>';
    const lB = pB.layers[i] ? `<strong>${pB.layers[i].layer_name}</strong><br><span style="font-size:0.82rem;color:var(--text-secondary)">${pB.layers[i].sublayers.map(s => s.name).join(' → ')}</span>` : '<span style="color:var(--text-muted)">Implicit in architecture</span>';
    return `
      <tr>
        <td><span class="stage-badge">${st.title}</span><div class="stage-desc-text">${st.desc}</div></td>
        <td>${lA}</td>
        <td>${lB}</td>
      </tr>`;
  }).join('');
}

function getComplexity(pid, metric) {
  const table = {
    transformer: { train: 'O(L² · D)', step: 'O(L · D)', space: 'O(B · L · D)', recall: 'Perfect Pairwise (Exact)', horizon: 'Memory-Bound (~8k-32k)' },
    mamba: { train: 'O(L · D · N)', step: 'O(1) Constant', space: 'O(B · D · N)', recall: 'Selective Gated Scan', horizon: 'Extrapolates to 1M+' },
    mixture_of_depths: { train: 'O(c · L² · D)', step: 'O(c · L · D)', space: 'O(c · B · L · D)', recall: 'High on Selected Tokens', horizon: 'Matches Dense (~32k)' },
    moe_survey: { train: 'O(k/E · FLOPs)', step: 'O(k/E · FLOPs)', space: 'O(B · L · D)', recall: 'Specialized Multi-Domain', horizon: 'Matches Baseline' },
    swat: { train: 'O(W · L · D)', step: 'O(W · D)', space: 'O(B · W · D)', recall: 'Local Window + RoPE', horizon: '128k+ Streaming' },
    titans: { train: 'O(W·L + L·D²)', step: 'O(W · D)', space: 'O(B·W·D + M_w)', recall: 'Surprise-Gradient Recall', horizon: '2M+ Needle Recall' },
    transmamba: { train: 'O(N_tr² + L·D·N)', step: 'O(1) after N_tr', space: 'O(B·D·N) [MCC]', recall: 'Exact Prompt + Fast Gen', horizon: '500k+ Hybrid' }
  };
  return (table[pid] && table[pid][metric]) ? table[pid][metric] : 'Linear / Hybrid';
}


// ──────────────────────────────────────────────────────────────────
// PHASE 4: LIVE COMPUTATIONAL SIMULATOR (FIX 1 & FIX 2 IMPLEMENTED)
// ──────────────────────────────────────────────────────────────────
function initSimulator() {
  if (!simSeqLen) return;
  const sliders = [simSeqLen, simHiddenDim, simNumLayers, simBatchSize, simStateDim, simPrecision];
  sliders.forEach(sl => {
    if (sl) sl.addEventListener('input', updateSimulator);
  });
  updateSimulator();
}

function updateSimulator() {
  const L = parseInt(simSeqLen.value, 10);
  const D = parseInt(simHiddenDim.value, 10);
  const N = parseInt(simNumLayers.value, 10);
  const B = parseInt(simBatchSize.value, 10);
  const N_state = parseInt(simStateDim.value, 10);
  const precBytes = parseInt(simPrecision.value, 10);

  // Update slider value badges
  simSeqLenVal.textContent = L >= 1024 ? `${(L / 1024).toFixed(0)}k (${L.toLocaleString()})` : L.toLocaleString();
  simHiddenDimVal.textContent = D.toLocaleString();
  simNumLayersVal.textContent = N.toString();
  simBatchSizeVal.textContent = B.toString();
  simStateDimVal.textContent = N_state.toString();
  simPrecisionVal.textContent = `${precBytes * 8}-bit (${precBytes} Byte${precBytes > 1 ? 's' : ''})`;

  // ==========================================
  // SECTION A: LIVE COMPUTED FORMULAS (JS ENGINE)
  // ==========================================

  // 1. KV Cache Memory (Transformer): 2 * B * L * N * D * precBytes
  const transKvBytes = 2 * B * L * N * D * precBytes;
  const transKvFormatted = formatBytes(transKvBytes);

  // 2. Mamba SSM State Memory: B * N * D * N_state * precBytes (Constant in L!)
  const mambaStateBytes = B * N * D * N_state * precBytes;
  const mambaStateFormatted = formatBytes(mambaStateBytes);

  // 3. Sliding Window Cache (W = 2048): 2 * B * min(L, 2048) * N * D * precBytes
  const W = 2048;
  const swatCacheBytes = 2 * B * Math.min(L, W) * N * D * precBytes;
  const swatCacheFormatted = formatBytes(swatCacheBytes);

  // 4. Mixture of Depths (c = 0.125): 12.5% capacity factor
  const modKvBytes = transKvBytes * 0.125;
  const modKvFormatted = formatBytes(modKvBytes);

  // 5. Total Compute FLOPs per Sequence (Forward Pass):
  // Transformer: 24 * N * L * D^2 + 4 * N * L^2 * D
  const transFlops = (24 * N * L * Math.pow(D, 2)) + (4 * N * Math.pow(L, 2) * D);
  // Mamba: 24 * N * L * D^2 + 6 * N * L * D * N_state
  const mambaFlops = (24 * N * L * Math.pow(D, 2)) + (6 * N * L * D * N_state);

  // 6. Memory compression ratio at current sequence length
  const compressionRatio = (transKvBytes / Math.max(mambaStateBytes, 1)).toFixed(1);

  // Render Section A Computed Cards
  computedMetricsGrid.innerHTML = `
    <!-- Transformer KV Cache Card -->
    <div class="computed-card">
      <div class="comp-card-top">
        <span class="comp-card-title">Transformer KV Cache Memory</span>
        <span class="tag-badge" style="color:var(--accent-rose)">O(L) Linear</span>
      </div>
      <div class="comp-card-val" style="color:var(--accent-rose)">${transKvFormatted}</div>
      <div class="comp-card-formula">Formula: $2 \\cdot B \\cdot L \\cdot N \\cdot D \\cdot \\text{bytes}$</div>
      <p class="comp-card-insight">Grows continuously with sequence length. At long contexts, reloading this cache completely saturates GPU memory bandwidth during decoding.</p>
    </div>

    <!-- Mamba SSM State Memory Card -->
    <div class="computed-card">
      <div class="comp-card-top">
        <span class="comp-card-title">Mamba SSM Hidden State Memory</span>
        <span class="tag-badge" style="color:var(--accent-emerald)">O(1) Constant in L</span>
      </div>
      <div class="comp-card-val" style="color:var(--accent-emerald)">${mambaStateFormatted}</div>
      <div class="comp-card-formula">Formula: $B \\cdot N \\cdot D \\cdot N_{state} \\cdot \\text{bytes}$</div>
      <p class="comp-card-insight"><strong>${compressionRatio}× smaller</strong> than Transformer KV cache! Stays completely fixed in memory regardless of whether generation is 1k or 1M tokens.</p>
    </div>

    <!-- MoD Compute Savings Card -->
    <div class="computed-card">
      <div class="comp-card-top">
        <span class="comp-card-title">Mixture-of-Depths (MoD) Cache</span>
        <span class="tag-badge" style="color:var(--accent-primary)">c = 12.5% Cap</span>
      </div>
      <div class="comp-card-val" style="color:var(--accent-primary)">${modKvFormatted}</div>
      <div class="comp-card-formula">Formula: $c \\cdot (2 \\cdot B \\cdot L \\cdot N \\cdot D \\cdot \\text{bytes})$</div>
      <p class="comp-card-insight">Reduces total attention compute FLOPs by up to 50% by routing only top 12.5% informative tokens into self-attention.</p>
    </div>

    <!-- Sliding Window Ring Buffer Card -->
    <div class="computed-card">
      <div class="comp-card-top">
        <span class="comp-card-title">SWAT Sliding Window Cache (W=2k)</span>
        <span class="tag-badge" style="color:var(--accent-amber)">Bounded O(W)</span>
      </div>
      <div class="comp-card-val" style="color:var(--accent-amber)">${swatCacheFormatted}</div>
      <div class="comp-card-formula">Formula: $2 \\cdot B \\cdot \\min(L, W) \\cdot N \\cdot D \\cdot \\text{bytes}$</div>
      <p class="comp-card-insight">Caps memory at a window of 2,048 tokens. Infinite generation possible on single GPU without OOM.</p>
    </div>

    <!-- Forward FLOPs Comparison Card -->
    <div class="computed-card">
      <div class="comp-card-top">
        <span class="comp-card-title">Total Forward FLOPs (Seq)</span>
        <span class="tag-badge">Compute Scaling</span>
      </div>
      <div class="comp-card-val" style="font-size:1.15rem;">
        <span style="color:var(--accent-rose)">Trans: ${formatFlops(transFlops)}</span><br>
        <span style="color:var(--accent-emerald)">Mamba: ${formatFlops(mambaFlops)}</span>
      </div>
      <div class="comp-card-formula">Trans: $24NLD^2 + 4NL^2D$ | Mamba: $24NLD^2 + 6NLDN_{state}$</div>
      <p class="comp-card-insight">Notice quadratic term $4NL^2D$ exploding in Transformer at $L > 32k$ tokens, while Mamba remains strictly linear.</p>
    </div>

    <!-- Hardware SRAM Scan Efficiency -->
    <div class="computed-card">
      <div class="comp-card-top">
        <span class="comp-card-title">GPU SRAM Fusion Factor</span>
        <span class="tag-badge" style="color:var(--accent-cyan)">Flash / SRAM</span>
      </div>
      <div class="comp-card-val" style="color:var(--accent-cyan)">Fused Scan Kernel</div>
      <div class="comp-card-formula">HBM IO Bound → SRAM Compute Bound</div>
      <p class="comp-card-insight">Mamba avoids materializing intermediate $(B, L, D, N)$ tensors in slow DRAM by keeping the recurrence strictly inside fast on-chip SRAM.</p>
    </div>
  `;

  // Render Memory Divergence Visualizer Bars
  const maxBytes = Math.max(transKvBytes, mambaStateBytes * 1.5, 1024);
  const transPercent = Math.min((transKvBytes / maxBytes) * 100, 100);
  const mambaPercent = Math.min((mambaStateBytes / maxBytes) * 100, 100);
  const modPercent = Math.min((modKvBytes / maxBytes) * 100, 100);
  const swatPercent = Math.min((swatCacheBytes / maxBytes) * 100, 100);

  memoryDivergenceChart.innerHTML = `
    <div class="mem-bar-row">
      <div class="mem-bar-info">
        <span class="mem-model-name">Standard Transformer (Dense Attention)</span>
        <span class="mem-value-label" style="color:var(--accent-rose)">${transKvFormatted} (${transPercent.toFixed(1)}%)</span>
      </div>
      <div class="mem-bar-track">
        <div class="mem-bar-fill bar-transformer" style="width:${transPercent}%"></div>
      </div>
    </div>

    <div class="mem-bar-row">
      <div class="mem-bar-info">
        <span class="mem-model-name">Mamba Selective State Space (SSM)</span>
        <span class="mem-value-label" style="color:var(--accent-emerald)">${mambaStateFormatted} (${Math.max(mambaPercent, 1).toFixed(1)}%)</span>
      </div>
      <div class="mem-bar-track">
        <div class="mem-bar-fill bar-mamba" style="width:${Math.max(mambaPercent, 1.5)}%"></div>
      </div>
    </div>

    <div class="mem-bar-row">
      <div class="mem-bar-info">
        <span class="mem-model-name">Mixture-of-Depths (MoD 12.5% Cap)</span>
        <span class="mem-value-label" style="color:var(--accent-primary)">${modKvFormatted} (${modPercent.toFixed(1)}%)</span>
      </div>
      <div class="mem-bar-track">
        <div class="mem-bar-fill bar-mod" style="width:${modPercent}%"></div>
      </div>
    </div>

    <div class="mem-bar-row">
      <div class="mem-bar-info">
        <span class="mem-model-name">Sliding Window Attention (SWAT W=2048)</span>
        <span class="mem-value-label" style="color:var(--accent-amber)">${swatCacheFormatted} (${swatPercent.toFixed(1)}%)</span>
      </div>
      <div class="mem-bar-track">
        <div class="mem-bar-fill bar-swat" style="width:${swatPercent}%"></div>
      </div>
    </div>
  `;

  triggerKaTeX();
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function formatFlops(flops) {
  if (flops < 1e6) return `${(flops / 1e3).toFixed(1)} KFLOPs`;
  if (flops < 1e9) return `${(flops / 1e6).toFixed(1)} MFLOPs`;
  if (flops < 1e12) return `${(flops / 1e9).toFixed(2)} GFLOPs`;
  if (flops < 1e15) return `${(flops / 1e12).toFixed(2)} TFLOPs`;
  return `${(flops / 1e15).toFixed(2)} PFLOPs`;
}


// ──────────────────────────────────────────────────────────────────
// KATEX AUTO-RENDER TRIGGER
// ──────────────────────────────────────────────────────────────────
function triggerKaTeX() {
  if (window.renderMathInElement) {
    renderMathInElement(document.body, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '$', right: '$', display: false }
      ],
      throwOnError: false
    });
  } else {
    setTimeout(triggerKaTeX, 100);
  }
}


// ──────────────────────────────────────────────────────────────────
// GLOBAL WINDOW EXPORTS
// ──────────────────────────────────────────────────────────────────
window.showCitation = showCitation;
window.drillDownNode = drillDownNode;
window.toggleLayerCard = toggleLayerCard;
window.switchSublayerTab = switchSublayerTab;
window.copyToClipboard = copyToClipboard;
window.setComparisonPreset = setComparisonPreset;
window.setTensorShapePreset = setTensorShapePreset;
window.addEventListener('DOMContentLoaded', initApp);
