/**
 * Synthetica Research Atlas — Core Application Engine (Stitch Edition)
 * Comprehensive Neural Architecture Explorer, Cross-Model Comparator, and Physics Simulator
 */

// Global State
let PAPERS_DATA = [];
let ACTIVE_PAPER_INDEX = 0;
let ACTIVE_PAGE = 'view-atlas-home';
let ACTIVE_EXPLORER_SUBVIEW = 'graphical-architecture';
let MASTERED_PAPERS = new Set();
let CURRENT_TENSOR_PRESET = { b: 2, l: 2048 };
let IS_SIMULATING_FLOW = false;

// Inner Circuit Deep Explanations for Graphical Block Drill-Downs
const INNER_CIRCUITS = {
  'multihead_attention': {
    title: 'Multi-Head Self-Attention Circuit',
    subtitle: 'Joint attention across $h$ parallel representation subspaces',
    category: 'Attention Mechanism',
    formula: '\\text{MultiHead}(Q, K, V) = \\text{Concat}(\\text{head}_1, \\dots, \\text{head}_h)W^O \\quad \\text{where} \\quad \\text{head}_i = \\text{softmax}\\left(\\frac{Q W_i^Q (K W_i^K)^T}{\\sqrt{d_k}}\\right) V W_i^V',
    steps: [
      { step: '1. Linear Projections', desc: 'Project input tensor $X \\in \\mathbb{R}^{B \\times L \\times D}$ into Queries ($Q$), Keys ($K$), and Values ($V$) using learnable projection weights $W^Q, W^K, W^V \\in \\mathbb{R}^{D \\times D}$.' },
      { step: '2. Multi-Head Reshaping', desc: 'Reshape and transpose $(B, L, D) \\rightarrow (B, h, L, d_k)$ where head dimension $d_k = D/h = 64$.' },
      { step: '3. Scaled Dot-Product & Masking', desc: 'Compute compatibility matrix $S = \\frac{QK^T}{\\sqrt{d_k}} \\in \\mathbb{R}^{B \\times h \\times L \\times L}$. For autoregressive generation, apply upper-triangular $-\\infty$ causal mask.' },
      { step: '4. Softmax Normalization', desc: 'Apply row-wise softmax $A = \\text{softmax}(S)$ to produce attention weight probability distribution summing to 1 across sequence length.' },
      { step: '5. Value Aggregation & Output Projection', desc: 'Multiply attention distribution by values: $O_{heads} = A \\cdot V$. Concatenate all $h$ heads and project via $W^O \\in \\mathbb{R}^{D \\times D}$.' }
    ],
    undertones: 'FlashAttention tile streaming fuses the Softmax scaling into GPU SRAM, avoiding the $O(L^2)$ HBM memory read/write bottleneck.',
    pytorch: `def forward(self, q, k, v, mask=None):\n    B, L, _ = q.shape\n    q = self.w_q(q).view(B, L, self.h, self.d_k).transpose(1, 2)\n    k = self.w_k(k).view(B, L, self.h, self.d_k).transpose(1, 2)\n    v = self.w_v(v).view(B, L, self.h, self.d_k).transpose(1, 2)\n    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)\n    if mask is not None:\n        scores = scores.masked_fill(mask == 0, -1e9)\n    attn = F.softmax(scores, dim=-1)\n    out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, self.d_model)\n    return self.w_o(out)`
  },
  'selective_ssm': {
    title: 'Mamba Selective State Space (S6) Circuit',
    subtitle: 'Input-dependent continuous-time dynamical system with Zero-Order Hold discretization',
    category: 'State Space Model',
    formula: 'h_t = \\bar{A}_t h_{t-1} + \\bar{B}_t x_t, \\quad y_t = C_t h_t \\quad \\text{where} \\quad \\bar{A}_t = \\exp(\\Delta_t A), \\; \\bar{B}_t = (\\Delta_t A)^{-1}(\\exp(\\Delta_t A) - I) \\cdot \\Delta_t B_t',
    steps: [
      { step: '1. Dynamic Parameter Generation', desc: 'Compute input-dependent timescale $\\Delta_t = \\text{softplus}(\\text{Linear}(x_t))$, input matrix $B_t = \\text{Linear}(x_t)$, and output matrix $C_t = \\text{Linear}(x_t)$.' },
      { step: '2. Discretization via Zero-Order Hold (ZOH)', desc: 'Transform continuous state matrix $A$ and $B_t$ into discrete step matrices $\\bar{A}_t = \\exp(\\Delta_t A)$ and $\\bar{B}_t \\approx \\Delta_t B_t$.' },
      { step: '3. Hardware-Aware Associative Parallel Scan', desc: 'Compute the prefix scan across the sequence in parallel in GPU SRAM without materializing intermediate $L \\times D \\times N$ tensors in DRAM.' },
      { step: '4. Output Gating & Projection', desc: 'Multiply state outputs by input channel gate using SiLU activation and project back to $d_{model}$.' }
    ],
    undertones: 'Unlike Transformers which store full sequence history in KV cache ($O(B \\cdot L \\cdot D)$), Mamba compresses context into a fixed-size state vector $h_t \\in \\mathbb{R}^{B \\times D \\times N}$, achieving $O(1)$ memory per step during inference.',
    pytorch: `def forward(self, x):\n    # x: (B, L, D)\n    B, L, D = x.shape\n    delta = F.softplus(self.dt_proj(self.x_proj(x)[:, :, :self.dt_rank]))\n    B_mat = self.x_proj(x)[:, :, self.dt_rank:self.dt_rank + self.d_state]\n    C_mat = self.x_proj(x)[:, :, self.dt_rank + self.d_state:]\n    A_bar = torch.exp(delta.unsqueeze(-1) * self.A_log.unsqueeze(0).unsqueeze(0))\n    B_bar = delta.unsqueeze(-1) * B_mat.unsqueeze(2)\n    # Parallel associative scan executes in GPU SRAM\n    y = selective_scan_fn(x, A_bar, B_bar, C_mat)\n    return y`
  },
  'topk_routing': {
    title: 'Mixture-of-Depths Top-K Capacity Router',
    subtitle: 'Dynamic token compute allocation with strict FLOP budget constraint',
    category: 'Dynamic Routing',
    formula: 'R(X) = \\text{TopK}(w_r^T X, \\; k = \\lfloor c \\cdot L \\rfloor), \\quad \\tilde{X} = \\text{LayerNorm}(X_{selected}) \\rightarrow \\text{Block} \\rightarrow \\text{ScatterAdd}',
    steps: [
      { step: '1. Router Scoring', desc: 'Linear projection $r = w_r^T X \\in \\mathbb{R}^{B \\times L}$ assigns scalar computational importance score to every token.' },
      { step: '2. Capacity Budget Thresholding', desc: 'Identify top $k = \\lfloor c \\cdot L \\rfloor$ tokens (e.g. $c = 0.5$ means 50% capacity). Compute scalar cutoff threshold $\\tau$.' },
      { step: '3. Selective Gather & Computation', desc: 'Selected tokens pass through Multi-Head Attention and FFN. Unselected tokens bypass the block entirely via pure residual skip connection.' },
      { step: '4. Weighted Scatter-Add', desc: 'Scale processed tokens by router probability $P(r_i)$ and scatter back into sequence positions.' }
    ],
    undertones: 'Enables 50% reduction in per-layer FLOPs without hurting downstream validation loss because non-critical tokens (e.g. punctuation, common fillers) skip complex attention layers.',
    pytorch: `def forward(self, x):\n    # x: (B, L, D)\n    router_scores = self.router(x).squeeze(-1) # (B, L)\n    k = int(self.capacity_factor * x.size(1))\n    topk_weights, topk_indices = torch.topk(router_scores, k, dim=1)\n    topk_probs = F.softmax(topk_weights, dim=-1)\n    \n    # Gather selected tokens\n    selected_x = torch.gather(x, 1, topk_indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))\n    processed = self.block(selected_x) * topk_probs.unsqueeze(-1)\n    \n    # Scatter back and add residual\n    out = x.clone()\n    out.scatter_add_(1, topk_indices.unsqueeze(-1).expand(-1, -1, x.size(-1)), processed)\n    return out`
  },
  'sigmoid_attention': {
    title: 'SWAT Sigmoid Sliding Window Attention',
    subtitle: 'Softmax-free attention matrix normalization with bounded KV memory',
    category: 'Long-Context Attention',
    formula: 'A_{ij} = \\sigma\\left(\\frac{q_i k_j^T}{\\sqrt{d_k}} - \\gamma |i - j|\\right), \\quad |i - j| \\le W',
    steps: [
      { step: '1. Local Window Partitioning', desc: 'Restrict token attention receptive field to fixed sliding window $W$ (e.g. $W=2048$ tokens).' },
      { step: '2. Sigmoid Activation', desc: 'Replace global exponential softmax normalization with pointwise logistic sigmoid $\\sigma(z) = \\frac{1}{1 + e^{-z}}$.' },
      { step: '3. Decay Bias Injection', desc: 'Add distance-dependent ALiBi/RoPE frequency decay penalty to prioritize proximate tokens.' },
      { step: '4. Bounded KV Cache', desc: 'Evict KV states older than $W$ steps during generation, maintaining constant $O(B \\cdot W \\cdot D)$ memory.' }
    ],
    undertones: 'Eliminates the "attention sink" pathology where early tokens absorb excessive probability mass in standard softmax transformers.',
    pytorch: `def forward(self, q, k, v, window_size=2048):\n    # Local window attention with sigmoid\n    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)\n    scores = scores + self.alibi_bias\n    attn = torch.sigmoid(scores)\n    return torch.matmul(attn, v)`
  }
};

// ==========================================================================
// INITIALIZATION
// ==========================================================================

document.addEventListener('DOMContentLoaded', async () => {
  loadMasteredState();
  await loadPapersData();
  setupGlobalEvents();
  renderAtlasHome();
  renderExplorer();
  renderModelComparison();
  renderSimulator();
  renderDatasetPage();
  updateMasteryUI();
});

// Load Papers Data from JSON
async function loadPapersData() {
  try {
    const res = await fetch('papers_data.json');
    PAPERS_DATA = await res.json();
  } catch (err) {
    console.error('Failed to load papers_data.json:', err);
  }
}

// LocalStorage Mastery Management
function loadMasteredState() {
  try {
    const saved = localStorage.getItem('deep_paper_atlas_mastered');
    if (saved) {
      MASTERED_PAPERS = new Set(JSON.parse(saved));
    }
  } catch (e) {
    console.warn('LocalStorage error:', e);
  }
}

function saveMasteredState() {
  try {
    localStorage.setItem('deep_paper_atlas_mastered', JSON.stringify([...MASTERED_PAPERS]));
  } catch (e) {
    console.warn('LocalStorage save error:', e);
  }
}

function toggleMastery(paperId) {
  if (MASTERED_PAPERS.has(paperId)) {
    MASTERED_PAPERS.delete(paperId);
  } else {
    MASTERED_PAPERS.add(paperId);
  }
  saveMasteredState();
  updateMasteryUI();
  renderAtlasHome();
}

function updateMasteryUI() {
  const total = PAPERS_DATA.length || 7;
  const mastered = MASTERED_PAPERS.size;
  const percent = Math.round((mastered / total) * 100);

  const sidebarPercent = document.getElementById('sidebarMasteryPercent');
  const sidebarBar = document.getElementById('sidebarProgressBar');
  const sidebarCount = document.getElementById('sidebarMasteryCount');
  
  if (sidebarPercent) sidebarPercent.textContent = `${percent}%`;
  if (sidebarBar) sidebarBar.style.width = `${percent}%`;
  if (sidebarCount) sidebarCount.textContent = `${mastered} of ${total} Papers Mastered`;

  const activePaper = PAPERS_DATA[ACTIVE_PAPER_INDEX];
  const paperBtn = document.getElementById('paperStudyStatusBtn');
  const paperText = document.getElementById('paperStudyStatusText');
  if (paperBtn && activePaper) {
    const isMastered = MASTERED_PAPERS.has(activePaper.paper_id);
    paperBtn.classList.toggle('mastered', isMastered);
    if (paperText) paperText.textContent = isMastered ? 'Mastered ✓' : 'Mark as Mastered';
  }
}

// ==========================================================================
// NAVIGATION & EVENT ROUTING
// ==========================================================================

function switchNavPage(pageId) {
  ACTIVE_PAGE = pageId;
  
  // Update sidebar buttons
  document.querySelectorAll('.nav-item').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.page === pageId);
  });

  // Update page views
  document.querySelectorAll('.page-view').forEach(view => {
    view.classList.toggle('active', view.id === pageId);
  });

  // Update breadcrumb
  const breadcrumb = document.getElementById('topHeaderBreadcrumb');
  if (breadcrumb) {
    const titles = {
      'view-atlas-home': 'Atlas Home — Research Explorer',
      'view-paper-explorer': `Paper Explorer — ${PAPERS_DATA[ACTIVE_PAPER_INDEX]?.title || 'Architecture'}`,
      'view-comparison': 'Cross-Model Architecture Comparison',
      'view-simulator': 'Live Computational Complexity Simulator',
      'view-dataset': 'Research Papers & Citations'
    };
    breadcrumb.textContent = titles[pageId] || 'Atlas Home';
  }

  // Auto trigger KaTeX math render
  triggerKaTeX();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function setupGlobalEvents() {
  // Global Paper Selector in Top Header
  const headerSelect = document.getElementById('headerPaperSelector');
  if (headerSelect) {
    headerSelect.innerHTML = PAPERS_DATA.map((p, idx) => `<option value="${idx}">${p.title}</option>`).join('');
    headerSelect.addEventListener('change', (e) => {
      selectPaper(parseInt(e.target.value, 10));
      switchNavPage('view-paper-explorer');
    });
  }

  // Theme Toggle Button
  const themeToggle = document.getElementById('themeToggleBtn');
  if (themeToggle) {
    themeToggle.addEventListener('click', () => {
      const isDark = document.body.classList.contains('theme-dark');
      document.body.classList.toggle('theme-dark', !isDark);
      document.body.classList.toggle('theme-light', isDark);
      document.documentElement.classList.toggle('dark', !isDark);
    });
  }

  // Global Search Input
  const searchInput = document.getElementById('globalSearchInput');
  if (searchInput) {
    searchInput.addEventListener('input', (e) => {
      handleGlobalSearch(e.target.value.toLowerCase().trim());
    });
  }

  // Mastery Button on Paper Explorer
  const studyBtn = document.getElementById('paperStudyStatusBtn');
  if (studyBtn) {
    studyBtn.addEventListener('click', () => {
      const activePaper = PAPERS_DATA[ACTIVE_PAPER_INDEX];
      if (activePaper) toggleMastery(activePaper.paper_id);
    });
  }

  // Explorer Sub-View Switcher Tabs
  document.querySelectorAll('.view-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      const subview = tab.dataset.subview;
      if (!subview) return;
      ACTIVE_EXPLORER_SUBVIEW = subview;
      document.querySelectorAll('.view-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      document.querySelectorAll('.explorer-subview').forEach(s => s.classList.remove('active'));
      const target = document.getElementById(`subview-${subview}`);
      if (target) target.classList.add('active');
      triggerKaTeX();
    });
  });

  // Flow Simulation Button
  const simBtn = document.getElementById('simulateFlowBtn');
  if (simBtn) {
    simBtn.addEventListener('click', runTokenFlowSimulation);
  }

  // Expand All Cards Button
  const expandBtn = document.getElementById('expandAllCardsBtn');
  if (expandBtn) {
    expandBtn.addEventListener('click', toggleAllMacroCards);
  }

  // Modal Close Button
  const modalClose = document.getElementById('modalCloseBtn');
  const modal = document.getElementById('circuitModal');
  if (modalClose && modal) {
    modalClose.addEventListener('click', closeModal);
    modal.addEventListener('click', (e) => {
      if (e.target === modal) closeModal();
    });
  }

  // Simulator Range Sliders
  ['simSeqLen', 'simHiddenDim', 'simLayers', 'simHeads', 'simBatchSize', 'simStateDim'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('input', updateSimulator);
  });

  // Comparison Model Selectors
  const selA = document.getElementById('compareModelASelect');
  const selB = document.getElementById('compareModelBSelect');
  if (selA && selB) {
    selA.addEventListener('change', updateComparisonView);
    selB.addEventListener('change', updateComparisonView);
  }

  // Atlas Home Filter Pills
  document.querySelectorAll('.filter-pill').forEach(pill => {
    pill.addEventListener('click', () => {
      document.querySelectorAll('.filter-pill').forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      filterBentoCards(pill.dataset.filter);
    });
  });
}

// Global Search Filter Handler
function handleGlobalSearch(query) {
  if (ACTIVE_PAGE !== 'view-atlas-home') {
    switchNavPage('view-atlas-home');
  }
  const cards = document.querySelectorAll('.paper-bento-card');
  cards.forEach(card => {
    const text = card.textContent.toLowerCase();
    card.style.display = text.includes(query) ? 'flex' : 'none';
  });
}

// Filter Bento Cards by Paradigm
function filterBentoCards(category) {
  const cards = document.querySelectorAll('.paper-bento-card');
  cards.forEach(card => {
    if (category === 'all') {
      card.style.display = 'flex';
    } else {
      const paradigm = card.dataset.paradigm || '';
      card.style.display = paradigm.toLowerCase().includes(category) ? 'flex' : 'none';
    }
  });
}

// ==========================================================================
// PAGE 1: ATLAS HOME RENDERER
// ==========================================================================

function renderAtlasHome() {
  const grid = document.getElementById('homePapersGrid');
  if (!grid || !PAPERS_DATA.length) return;

  const paradigmMap = {
    'transformer': { name: 'Attention', badge: 'badge-attention', filter: 'attention', formula: '\\text{Attn}(Q,K,V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V', complexity: 'O(L² · D)' },
    'mamba': { name: 'State Space', badge: 'badge-ssm', filter: 'ssm', formula: 'h_t = \\bar{A}_t h_{t-1} + \\bar{B}_t x_t', complexity: 'O(L · D · N)' },
    'mod': { name: 'Dynamic Routing', badge: 'badge-mod', filter: 'mod', formula: 'R(X) = \\text{TopK}(w_r^T X, \\lfloor c \\cdot L \\rfloor)', complexity: 'O(c · L² · D)' },
    'moe_survey': { name: 'Mixture of Experts', badge: 'badge-moe', filter: 'moe', formula: 'y = \\sum_{i=1}^k G(x)_i E_i(x)', complexity: 'O(k · L · D)' },
    'swat': { name: 'Long Context', badge: 'badge-swat', filter: 'long-context', formula: 'A_{ij} = \\sigma\\left(\\frac{q_i k_j^T}{\\sqrt{d_k}} - \\gamma |i-j|\\right)', complexity: 'O(L · W · D)' },
    'titans': { name: 'Neural Memory', badge: 'badge-titans', filter: 'memory', formula: 'M_t = (1 - \\alpha_t) M_{t-1} + S_t', complexity: 'O(L · D)' },
    'transmamba': { name: 'SSM Hybrid', badge: 'badge-transmamba', filter: 'ssm', formula: 'y = \\text{MHA}(\\text{SSM}(X)) + \\text{FFN}(X)', complexity: 'O(L · D)' }
  };

  grid.innerHTML = PAPERS_DATA.map((paper, idx) => {
    const pInfo = paradigmMap[paper.paper_id] || { name: 'Deep Learning', badge: 'badge-attention', filter: 'all', formula: 'y = f(x)', complexity: 'O(L · D)' };
    const isMastered = MASTERED_PAPERS.has(paper.paper_id);

    return `
      <div class="paper-bento-card glass-panel" data-paradigm="${pInfo.filter}">
        <div class="bento-card-top">
          <div class="bento-card-header">
            <span class="bento-paradigm-badge ${pInfo.badge}">${pInfo.name}</span>
            <span class="bento-complexity-tag">${pInfo.complexity}</span>
          </div>
          <h3 class="bento-card-title">${paper.title}</h3>
          <p class="bento-card-authors">${paper.authors_venue_year}</p>
          <p class="bento-card-breakthrough">${paper.core_breakthrough}</p>
          
          <div class="bento-formula-preview">
            <span class="katex-render">$$${pInfo.formula}$$</span>
          </div>
        </div>

        <div class="bento-card-footer">
          <button class="bento-btn-primary" onclick="openPaperFromHome(${idx})">
            <span class="material-symbols-outlined text-[16px]">account_tree</span>
            <span>Explore Architecture</span>
          </button>
          <button class="bento-btn-secondary" onclick="openCompareFromHome('${paper.paper_id}')">
            <span class="material-symbols-outlined text-[16px]">compare_arrows</span>
          </button>
        </div>
      </div>
    `;
  }).join('');

  triggerKaTeX();
}

function openPaperFromHome(idx) {
  selectPaper(idx);
  switchNavPage('view-paper-explorer');
}

function openCompareFromHome(paperId) {
  const compareA = document.getElementById('compareModelASelect');
  if (compareA) compareA.value = paperId;
  updateComparisonView();
  switchNavPage('view-comparison');
}

// ==========================================================================
// PAGE 2: PAPER EXPLORER RENDERER
// ==========================================================================

function selectPaper(index) {
  ACTIVE_PAPER_INDEX = index;
  const paper = PAPERS_DATA[index];
  if (!paper) return;

  // Sync header select
  const headerSelect = document.getElementById('headerPaperSelector');
  if (headerSelect) headerSelect.value = index;

  renderPaperSelectorPills();
  renderPaperHero(paper);
  renderGraphicalCanvas(paper);
  renderMicroLayers(paper);
  renderProsCons(paper);
  renderReportedBenchmarks(paper);
  updateMasteryUI();
  triggerKaTeX();
}

function renderExplorer() {
  if (!PAPERS_DATA.length) return;
  selectPaper(ACTIVE_PAPER_INDEX);
}

function renderPaperSelectorPills() {
  const container = document.getElementById('explorerPaperPills');
  if (!container) return;

  container.innerHTML = PAPERS_DATA.map((p, idx) => `
    <button class="paper-pill-btn ${idx === ACTIVE_PAPER_INDEX ? 'active' : ''}" onclick="selectPaper(${idx})">
      <span>${p.title.split(':')[0]}</span>
    </button>
  `).join('');
}

function renderPaperHero(paper) {
  const title = document.getElementById('paperTitle');
  const authors = document.getElementById('paperAuthors');
  const breakthrough = document.getElementById('paperBreakthrough');
  const tags = document.getElementById('heroTags');

  if (title) title.textContent = paper.title;
  if (authors) authors.textContent = paper.authors_venue_year;
  if (breakthrough) breakthrough.textContent = paper.core_breakthrough;

  if (tags) {
    tags.innerHTML = `
      <span class="tag-badge">ID: ${paper.paper_id}</span>
      <span class="tag-badge">Layers: ${paper.layers.length} Macro Blocks</span>
      <span class="tag-badge">Asymptotic: ${paper.layers[0]?.flops_weight || 'O(L · D)'}</span>
    `;
  }
}

// Render Graphical Canvas Block Diagram
function renderGraphicalCanvas(paper) {
  const viewport = document.getElementById('diagramCanvasViewport');
  const diagramTitle = document.getElementById('diagramTitle');
  if (!viewport || !paper) return;

  if (diagramTitle) diagramTitle.textContent = `${paper.title} — Architectural Block Flow`;

  let blocksHtml = '';

  paper.layers.forEach((layer, idx) => {
    let blockClass = 'bg-blue';
    let drillDownKey = 'multihead_attention';

    const lName = layer.layer_name.toLowerCase();
    if (lName.includes('embed') || lName.includes('input') || lName.includes('position')) {
      blockClass = 'bg-pink';
      drillDownKey = 'multihead_attention';
    } else if (lName.includes('ssm') || lName.includes('state')) {
      blockClass = 'bg-orange';
      drillDownKey = 'selective_ssm';
    } else if (lName.includes('top-k') || lName.includes('rout') || lName.includes('depth')) {
      blockClass = 'bg-yellow';
      drillDownKey = 'topk_routing';
    } else if (lName.includes('sigmoid') || lName.includes('swat') || lName.includes('window')) {
      blockClass = 'bg-purple';
      drillDownKey = 'sigmoid_attention';
    } else if (lName.includes('norm') || lName.includes('gate')) {
      blockClass = 'bg-yellow';
    } else if (lName.includes('linear') || lName.includes('head') || lName.includes('output')) {
      blockClass = 'bg-purple';
    }

    const isStack = lName.includes('stack') || lName.includes('block') || lName.includes('nx');

    const blockMarkup = `
      <div class="diagram-block ${blockClass}" data-layer-idx="${idx}" onclick="openCircuitModal('${drillDownKey}', '${layer.layer_name}')">
        <div class="block-title-subgroup">
          <span class="block-title-text">${layer.layer_name}</span>
          <span class="block-desc-text">${layer.sublayers ? layer.sublayers.length + ' Sub-Operations' : 'Macro Layer'}</span>
        </div>
        <span class="block-action-chip">Inspect Mechanism ➔</span>
      </div>
    `;

    if (isStack) {
      blocksHtml += `
        <div class="diagram-stack-box">
          <span class="stack-multiplier-tag">N = 6 to 32 Identical Layers</span>
          ${blockMarkup}
        </div>
      `;
    } else {
      blocksHtml += blockMarkup;
    }

    if (idx < paper.layers.length - 1) {
      blocksHtml += `
        <svg class="flow-arrow-svg" width="20" height="24" viewBox="0 0 20 24" fill="none">
          <path d="M10 0V20M10 20L4 14M10 20L16 14" stroke="#919096" stroke-width="2" stroke-linecap="round"/>
        </svg>
      `;
    }
  });

  viewport.innerHTML = `<div class="flow-diagram-container" id="flowDiagramContainer">${blocksHtml}</div>`;
}

// Simulate Forward Token Pulse Animation
function runTokenFlowSimulation() {
  if (IS_SIMULATING_FLOW) return;
  IS_SIMULATING_FLOW = true;

  const blocks = document.querySelectorAll('.diagram-block');
  if (!blocks.length) {
    IS_SIMULATING_FLOW = false;
    return;
  }

  let index = 0;
  const interval = setInterval(() => {
    blocks.forEach(b => b.classList.remove('sim-active'));
    if (index < blocks.length) {
      blocks[index].classList.add('sim-active');
      index++;
    } else {
      clearInterval(interval);
      setTimeout(() => {
        blocks.forEach(b => b.classList.remove('sim-active'));
        IS_SIMULATING_FLOW = false;
      }, 600);
    }
  }, 400);
}

// Toggle Macro-Layer Cards Expand / Collapse All
function toggleAllMacroCards() {
  const cards = document.querySelectorAll('.layer-card');
  const anyCollapsed = Array.from(cards).some(c => c.classList.contains('collapsed'));
  cards.forEach(card => card.classList.toggle('collapsed', !anyCollapsed));
}

// Dynamic Tensor Dimension Calculator
function setTensorShapePreset(b, l) {
  CURRENT_TENSOR_PRESET = { b, l };
  
  document.querySelectorAll('.tensor-preset-pill').forEach(btn => {
    btn.classList.toggle('active', btn.textContent.includes(`B=${b}, L=${l.toLocaleString()}`));
  });

  const indicator = document.getElementById('tensorLiveDimIndicator');
  if (indicator) {
    indicator.textContent = `Active Dimensions: [Batch: ${b}, SeqLen: ${l.toLocaleString()}, Dim: 512]`;
  }

  // Re-render micro-layers with concrete calculated shapes
  const paper = PAPERS_DATA[ACTIVE_PAPER_INDEX];
  if (paper) renderMicroLayers(paper);
}

function formatConcreteTensorShape(symbolicStr, b, l) {
  if (!symbolicStr) return `[${b}, ${l}, 512]`;
  let formatted = symbolicStr
    .replace(/\bB\b/g, b)
    .replace(/\bL\b/g, l)
    .replace(/\bT\b/g, l)
    .replace(/\bd_model\b/g, '512')
    .replace(/\bd_ff\b/g, '2048')
    .replace(/\bd_k\b/g, '64')
    .replace(/\bh\b/g, '8')
    .replace(/\bN\b/g, '16');
  return formatted;
}

// Render Micro-Layers with Horizontal Flowchart & 5-Tab Inspection Cards
function renderMicroLayers(paper) {
  const grid = document.getElementById('layerCardsGrid');
  const sectionTitle = document.getElementById('microSectionTitle');
  if (!grid || !paper) return;

  if (sectionTitle) sectionTitle.textContent = `${paper.title} — Micro-Layer Breakdown & Math`;

  const { b, l } = CURRENT_TENSOR_PRESET;

  grid.innerHTML = paper.layers.map((macro, mIdx) => {
    const macroInShape = formatConcreteTensorShape(macro.tensor_in || '(B, L)', b, l);
    const macroOutShape = formatConcreteTensorShape(macro.tensor_out || '(B, L, 512)', b, l);

    // Horizontal pipeline steps
    const pipelineSteps = (macro.sublayers || []).map((sub, sIdx) => `
      <span class="pipeline-step">Step ${sIdx + 1}: ${sub.name}</span>
      ${sIdx < macro.sublayers.length - 1 ? '<span class="pipeline-arrow">➔</span>' : ''}
    `).join('');

    // Detailed Sublayer Cards
    const sublayersHtml = (macro.sublayers || []).map((sub, sIdx) => {
      const inShapeConcrete = formatConcreteTensorShape(sub.in_shape || '(B, L, 512)', b, l);
      const outShapeConcrete = formatConcreteTensorShape(sub.out_shape || '(B, L, 512)', b, l);

      const hyperparamsHtml = (sub.hyperparameters || []).map(hp => `
        <div class="hyperparam-chip">
          <div class="hyperparam-name">${hp.param} = ${hp.val}</div>
          <div class="hyperparam-desc">${hp.desc}</div>
        </div>
      `).join('') || '<p class="text-muted">Standard layer hyperparameters</p>';

      const codeEscaped = (sub.pytorch_code || 'x = layer(x)').replace(/"/g, '&quot;');

      return `
        <div class="sublayer-item-card">
          <div class="sublayer-header-row">
            <h4 class="sublayer-name">${mIdx + 1}.${sIdx + 1} — ${sub.name}</h4>
            <span class="sublayer-section-badge">${sub.source_section || 'Primary Paper'}</span>
          </div>

          <!-- Formula Hero Box -->
          <div class="sublayer-formula-hero">
            <div class="formula-math-render">
              <span class="katex-render">$$${sub.formula_katex || 'y = f(x)'}$$</span>
            </div>
            <button class="copy-math-btn" onclick="copyToClipboard('${sub.formula_katex ? sub.formula_katex.replace(/\\/g, '\\\\').replace(/'/g, "\\'") : ''}', this)">Copy KaTeX</button>
          </div>

          <p class="sublayer-narrative">${sub.explanation || ''} ${sub.math_breakdown || ''}</p>

          <!-- 5-Tab Multi-Inspection Panel -->
          <div class="sublayer-tabs-container" id="tabs-${mIdx}-${sIdx}">
            <div class="sublayer-tabs-nav">
              <button class="sublayer-tab-btn active" onclick="switchSublayerTab('${mIdx}-${sIdx}', 'shapes', this)">📐 Shapes</button>
              <button class="sublayer-tab-btn" onclick="switchSublayerTab('${mIdx}-${sIdx}', 'params', this)">⚙️ Hyperparams</button>
              <button class="sublayer-tab-btn" onclick="switchSublayerTab('${mIdx}-${sIdx}', 'hardware', this)">⚡ Hardware</button>
              <button class="sublayer-tab-btn" onclick="switchSublayerTab('${mIdx}-${sIdx}', 'stability', this)">⚠️ Stability</button>
              <button class="sublayer-tab-btn" onclick="switchSublayerTab('${mIdx}-${sIdx}', 'pytorch', this)">💻 PyTorch</button>
            </div>

            <!-- TAB 1: Shapes -->
            <div class="sublayer-tab-content active" id="tab-${mIdx}-${sIdx}-shapes">
              <div class="shape-badges-grid">
                <div class="shape-chip">
                  <div class="shape-chip-label">Input Tensor Shape:</div>
                  <div class="shape-chip-val">${inShapeConcrete}</div>
                </div>
                <div class="shape-chip">
                  <div class="shape-chip-label">Output Tensor Shape:</div>
                  <div class="shape-chip-val">${outShapeConcrete}</div>
                </div>
              </div>
            </div>

            <!-- TAB 2: Hyperparams -->
            <div class="sublayer-tab-content" id="tab-${mIdx}-${sIdx}-params">
              <div class="hyperparam-cards-grid">${hyperparamsHtml}</div>
            </div>

            <!-- TAB 3: Hardware -->
            <div class="sublayer-tab-content" id="tab-${mIdx}-${sIdx}-hardware">
              <div class="hardware-callout-box">
                <strong>GPU Kernel & Memory Flow:</strong> ${sub.hardware_notes || 'Optimized for high GPU tensor core utilization.'}
              </div>
            </div>

            <!-- TAB 4: Stability -->
            <div class="sublayer-tab-content" id="tab-${mIdx}-${sIdx}-stability">
              <div class="stability-callout-box">
                <strong>Failure Modes & Mitigations:</strong> ${sub.failure_modes || 'Standard gradient clipping and residual scaling prevent instability.'}
              </div>
            </div>

            <!-- TAB 5: PyTorch Code -->
            <div class="sublayer-tab-content" id="tab-${mIdx}-${sIdx}-pytorch">
              <div class="pytorch-code-container">
                <pre><code>${sub.pytorch_code || 'x = layer(x)'}</code></pre>
                <button class="copy-code-btn" onclick="copyToClipboard('${codeEscaped}', this)">Copy Code</button>
              </div>
            </div>
          </div>
        </div>
      `;
    }).join('');

    return `
      <div class="layer-card" id="macroCard-${mIdx}">
        <div class="layer-card-header" onclick="toggleMacroCard(${mIdx})">
          <div class="layer-header-left">
            <span class="layer-idx-badge">${mIdx + 1}</span>
            <h3 class="layer-title-text">${macro.layer_name}</h3>
          </div>
          <div class="layer-header-right">
            <span class="layer-tag tag-math">${macroInShape} ➔ ${macroOutShape}</span>
            <span class="layer-tag">${macro.flops_weight || 'O(L · D)'}</span>
            <span class="material-symbols-outlined layer-collapse-icon">expand_more</span>
          </div>
        </div>

        <div class="layer-card-body">
          <div class="sublayer-pipeline-flowchart">
            ${pipelineSteps}
          </div>
          ${sublayersHtml}
        </div>
      </div>
    `;
  }).join('');

  triggerKaTeX();
}

function toggleMacroCard(idx) {
  const card = document.getElementById(`macroCard-${idx}`);
  if (card) card.classList.toggle('collapsed');
}

function switchSublayerTab(idPrefix, tabName, btnEl) {
  const parent = document.getElementById(`tabs-${idPrefix}`);
  if (!parent) return;
  parent.querySelectorAll('.sublayer-tab-btn').forEach(b => b.classList.remove('active'));
  parent.querySelectorAll('.sublayer-tab-content').forEach(c => c.classList.remove('active'));
  btnEl.classList.add('active');
  const target = document.getElementById(`tab-${idPrefix}-${tabName}`);
  if (target) target.classList.add('active');
}

// Copy Code / KaTeX Helper
function copyToClipboard(text, btnEl) {
  navigator.clipboard.writeText(text).then(() => {
    const orig = btnEl.textContent;
    btnEl.textContent = 'Copied! ✓';
    setTimeout(() => { btnEl.textContent = orig; }, 1500);
  });
}

// Render Pros & Cons View
function renderProsCons(paper) {
  const prosList = document.getElementById('prosList');
  const consList = document.getElementById('consList');
  if (!prosList || !consList || !paper) return;

  prosList.innerHTML = (paper.pros || []).map(p => `
    <div class="pro-con-item pro-item">
      <div class="pro-con-claim">${p.claim}</div>
      <div class="pro-con-detail">${p.mechanistic_reasoning || ''}</div>
      <span class="citation-chip">${p.source_section || 'Primary Paper'}</span>
    </div>
  `).join('');

  consList.innerHTML = (paper.cons || []).map(c => `
    <div class="pro-con-item con-item">
      <div class="pro-con-claim">${c.claim}</div>
      <div class="pro-con-detail">${c.mechanistic_reasoning || ''}</div>
      <span class="citation-chip">${c.source_section || 'Primary Paper'}</span>
    </div>
  `).join('');
}

// Render Reported Empirical Benchmarks View
function renderReportedBenchmarks(paper) {
  const tbody = document.getElementById('reportedMetricsTableBody');
  if (!tbody || !paper) return;

  tbody.innerHTML = (paper.reported_metrics || []).map(m => `
    <tr>
      <td><strong>${m.task}</strong></td>
      <td>${m.metric}</td>
      <td><span class="badge-success">${m.value}</span></td>
      <td><span class="citation-chip">${m.source_citation}</span></td>
      <td>${m.context}</td>
    </tr>
  `).join('');
}

// ==========================================================================
// PAGE 3: MODEL COMPARISON RENDERER
// ==========================================================================

function renderModelComparison() {
  const selA = document.getElementById('compareModelASelect');
  const selB = document.getElementById('compareModelBSelect');
  if (!selA || !selB || !PAPERS_DATA.length) return;

  selA.innerHTML = PAPERS_DATA.map((p, i) => `<option value="${p.paper_id}" ${i === 0 ? 'selected' : ''}>${p.title}</option>`).join('');
  selB.innerHTML = PAPERS_DATA.map((p, i) => `<option value="${p.paper_id}" ${i === 1 ? 'selected' : ''}>${p.title}</option>`).join('');

  updateComparisonView();
}

function setComparisonPreset(idA, idB) {
  const selA = document.getElementById('compareModelASelect');
  const selB = document.getElementById('compareModelBSelect');
  if (selA) selA.value = idA;
  if (selB) selB.value = idB;
  updateComparisonView();
}

function updateComparisonView() {
  const selA = document.getElementById('compareModelASelect');
  const selB = document.getElementById('compareModelBSelect');
  if (!selA || !selB) return;

  const paperA = PAPERS_DATA.find(p => p.paper_id === selA.value) || PAPERS_DATA[0];
  const paperB = PAPERS_DATA.find(p => p.paper_id === selB.value) || PAPERS_DATA[1];

  // Executive summaries
  const execGrid = document.getElementById('comparisonExecutiveGrid');
  if (execGrid) {
    execGrid.innerHTML = `
      <div class="exec-card">
        <div class="exec-card-header">
          <h4 class="exec-model-name">${paperA.title}</h4>
          <span class="exec-badge">${paperA.authors_venue_year}</span>
        </div>
        <p class="exec-breakthrough">${paperA.core_breakthrough}</p>
      </div>
      <div class="exec-card">
        <div class="exec-card-header">
          <h4 class="exec-model-name">${paperB.title}</h4>
          <span class="exec-badge">${paperB.authors_venue_year}</span>
        </div>
        <p class="exec-breakthrough">${paperB.core_breakthrough}</p>
      </div>
    `;
  }

  // Column headers
  const colA = document.getElementById('colHeaderModelA');
  const colB = document.getElementById('colHeaderModelB');
  if (colA) colA.textContent = `${paperA.title.split(':')[0]} Architecture`;
  if (colB) colB.textContent = `${paperB.title.split(':')[0]} Architecture`;

  // Layer Alignment Table
  const alignTbody = document.getElementById('layerAlignmentTableBody');
  if (alignTbody) {
    const stages = [
      { stage: '1. Input Representation', a: paperA.layers[0]?.layer_name || 'Standard Embedding', b: paperB.layers[0]?.layer_name || 'Standard Embedding' },
      { stage: '2. Core Sequence Mixer', a: paperA.layers[1]?.layer_name || 'Dense Multi-Head Attention', b: paperB.layers[1]?.layer_name || 'Selective SSM / Routing' },
      { stage: '3. Channel / Feedforward', a: 'Position-Wise Feedforward (d_ff=2048)', b: 'GLU / Gated Linear Expansion' },
      { stage: '4. Normalization & Residual', a: 'Post-LN / Pre-LN + Skip Add', b: 'RMSNorm / Fused Residual Norm' },
      { stage: '5. Output Projection Head', a: 'Linear Projection to Vocab + Softmax', b: 'Linear Projection to Vocab + Softmax' }
    ];

    alignTbody.innerHTML = stages.map(s => `
      <tr>
        <td><strong>${s.stage}</strong></td>
        <td>${s.a}</td>
        <td>${s.b}</td>
      </tr>
    `).join('');
  }

  // Head-to-Head Pros vs Cons
  const h2h = document.getElementById('headToHeadProsCons');
  if (h2h) {
    h2h.innerHTML = `
      <div class="pros-card glass-panel">
        <div class="pro-con-header pro-header">
          <div class="pro-con-icon">✓</div>
          <h3>${paperA.title.split(':')[0]} Advantages vs ${paperB.title.split(':')[0]}</h3>
        </div>
        <div class="pro-con-list">
          ${(paperA.pros || []).map(p => `<div class="pro-con-item pro-item"><div class="pro-con-claim">${p.claim}</div><div class="pro-con-detail">${p.mechanistic_reasoning || ''}</div></div>`).join('')}
        </div>
      </div>
      <div class="pros-card glass-panel">
        <div class="pro-con-header pro-header">
          <div class="pro-con-icon">✓</div>
          <h3>${paperB.title.split(':')[0]} Advantages vs ${paperA.title.split(':')[0]}</h3>
        </div>
        <div class="pro-con-list">
          ${(paperB.pros || []).map(p => `<div class="pro-con-item pro-item"><div class="pro-con-claim">${p.claim}</div><div class="pro-con-detail">${p.mechanistic_reasoning || ''}</div></div>`).join('')}
        </div>
      </div>
    `;
  }

  // Complexity Matrix Table
  const compTbody = document.getElementById('complexityMatrixTableBody');
  const compA = document.getElementById('colCompModelA');
  const compB = document.getElementById('colCompModelB');
  if (compA) compA.textContent = paperA.title.split(':')[0];
  if (compB) compB.textContent = paperB.title.split(':')[0];

  if (compTbody) {
    compTbody.innerHTML = `
      <tr>
        <td><strong>Training Time Scaling</strong></td>
        <td>${paperA.paper_id === 'mamba' ? 'O(L) Parallel Scan' : 'O(L²) Parallel MatMul'}</td>
        <td>${paperB.paper_id === 'mamba' ? 'O(L) Parallel Scan' : 'O(L²) Parallel MatMul'}</td>
        <td>Linear models scale to 1M+ tokens during training without OOM memory wall.</td>
      </tr>
      <tr>
        <td><strong>Inference Step FLOPs</strong></td>
        <td>${paperA.paper_id === 'mamba' ? 'O(1) Per-Token' : 'O(L) Attending to Cache'}</td>
        <td>${paperB.paper_id === 'mamba' ? 'O(1) Per-Token' : 'O(L) Attending to Cache'}</td>
        <td>Constant time inference eliminates throughput drop at long context lengths.</td>
      </tr>
      <tr>
        <td><strong>KV Cache / State Memory</strong></td>
        <td>${paperA.paper_id === 'mamba' ? 'O(1) Fixed State (16 MB)' : 'O(L) Growing Cache (GBs)'}</td>
        <td>${paperB.paper_id === 'mamba' ? 'O(1) Fixed State (16 MB)' : 'O(L) Growing Cache (GBs)'}</td>
        <td>Transformers suffer KV cache DRAM memory bottleneck at long sequence serving.</td>
      </tr>
      <tr>
        <td><strong>In-Context Induction Capability</strong></td>
        <td>99.8% Perfect Recall</td>
        <td>${paperB.paper_id === 'mamba' ? '99.4% Multi-Query Recall' : '99.8% Perfect Recall'}</td>
        <td>Full self-attention provides exact token-to-token lookup across full horizon.</td>
      </tr>
    `;
  }
}

// ==========================================================================
// PAGE 4: COMPLEXITY SIMULATOR (Live Math Formulas)
// ==========================================================================

function renderSimulator() {
  updateSimulator();
}

function updateSimulator() {
  const L = parseInt(document.getElementById('simSeqLen')?.value || '2048', 10);
  const D = parseInt(document.getElementById('simHiddenDim')?.value || '512', 10);
  const N_layers = parseInt(document.getElementById('simLayers')?.value || '6', 10);
  const h = parseInt(document.getElementById('simHeads')?.value || '8', 10);
  const B = parseInt(document.getElementById('simBatchSize')?.value || '2', 10);
  const N_state = parseInt(document.getElementById('simStateDim')?.value || '16', 10);

  // Update slider labels
  const seqLabel = document.getElementById('simSeqLenVal');
  const dimLabel = document.getElementById('simHiddenDimVal');
  const layersLabel = document.getElementById('simLayersVal');
  const headsLabel = document.getElementById('simHeadsVal');
  const batchLabel = document.getElementById('simBatchSizeVal');
  const stateLabel = document.getElementById('simStateDimVal');

  if (seqLabel) seqLabel.textContent = `${L.toLocaleString()} tokens`;
  if (dimLabel) dimLabel.textContent = `${D}`;
  if (layersLabel) layersLabel.textContent = `${N_layers}`;
  if (headsLabel) headsLabel.textContent = `${h}`;
  if (batchLabel) batchLabel.textContent = `${B}`;
  if (stateLabel) stateLabel.textContent = `${N_state}`;

  // Computations
  // 1. Transformer Attention FLOPs: 2 * B * (4 * L * D^2 + 2 * L^2 * D) * N_layers
  const transFlops = (2 * B * (4 * L * D * D + 2 * L * L * D) * N_layers) / 1e9;
  
  // 2. Mamba SSM Compute FLOPs: 2 * B * (3 * L * D * N_state + 4 * L * D^2) * N_layers
  const mambaFlops = (2 * B * (3 * L * D * N_state + 4 * L * D * D) * N_layers) / 1e9;

  // 3. Transformer KV Cache Memory (fp16 = 2 bytes): 2 * B * L * D * N_layers * 2 bytes
  const kvCacheBytes = 2 * B * L * D * N_layers * 2;
  const kvCacheMB = kvCacheBytes / (1024 * 1024);

  // 4. Mamba State Memory: B * D * N_state * N_layers * 2 bytes
  const mambaStateBytes = B * D * N_state * N_layers * 2;
  const mambaStateMB = mambaStateBytes / (1024 * 1024);

  // Update UI Elements
  const elTransFlops = document.getElementById('compTransFlops');
  const elMambaFlops = document.getElementById('compMambaFlops');
  const elMambaSavings = document.getElementById('compMambaSavings');
  const elKvMem = document.getElementById('compKvCacheMem');
  const elMambaMem = document.getElementById('compMambaMem');

  if (elTransFlops) elTransFlops.textContent = `${transFlops.toFixed(2)} GFLOPs`;
  if (elMambaFlops) elMambaFlops.textContent = `${mambaFlops.toFixed(2)} GFLOPs`;
  if (elMambaSavings) {
    const savings = ((1 - mambaFlops / transFlops) * 100).toFixed(1);
    elMambaSavings.textContent = `${savings}% compute vs Transformer`;
  }
  if (elKvMem) elKvMem.textContent = `${kvCacheMB.toFixed(2)} MB`;
  if (elMambaMem) elMambaMem.textContent = `${mambaStateMB.toFixed(2)} MB`;

  // Update Scaling Divergence Chart
  const maxMem = Math.max(kvCacheMB, mambaStateMB * 2, 1);
  const transPct = Math.min(100, Math.max(4, (kvCacheMB / maxMem) * 100));
  const mambaPct = Math.min(100, Math.max(2, (mambaStateMB / maxMem) * 100));

  const chartTransVal = document.getElementById('chartTransVal');
  const chartTransBar = document.getElementById('chartTransBar');
  const chartMambaVal = document.getElementById('chartMambaVal');
  const chartMambaBar = document.getElementById('chartMambaBar');

  if (chartTransVal) chartTransVal.textContent = `${kvCacheMB.toFixed(2)} MB`;
  if (chartTransBar) chartTransBar.style.width = `${transPct}%`;
  if (chartMambaVal) chartMambaVal.textContent = `${mambaStateMB.toFixed(2)} MB`;
  if (chartMambaBar) chartMambaBar.style.width = `${mambaPct}%`;
}

// ==========================================================================
// PAGE 5: PAPERS DATASET & BIBLIOGRAPHY RENDERER
// ==========================================================================

function renderDatasetPage() {
  const grid = document.getElementById('datasetCardsGrid');
  if (!grid || !PAPERS_DATA.length) return;

  grid.innerHTML = PAPERS_DATA.map((paper) => {
    const bibtex = `@article{${paper.paper_id}_deep_paper,\n  title={${paper.title}},\n  author={${paper.authors_venue_year.split(',')[0]}},\n  journal={${paper.authors_venue_year.split(',')[1] || 'arXiv'}},\n  year={2017--2026}\n}`;
    const escapedBib = bibtex.replace(/"/g, '&quot;');

    return `
      <div class="dataset-paper-card">
        <div>
          <h3 class="dataset-paper-title">${paper.title}</h3>
          <p class="dataset-paper-meta">${paper.authors_venue_year}</p>
          <div class="bibtex-box"><pre>${bibtex}</pre></div>
        </div>
        <button class="bento-btn-primary" onclick="copyToClipboard('${escapedBib}', this)">
          <span class="material-symbols-outlined text-[16px]">content_copy</span>
          <span>Copy BibTeX</span>
        </button>
      </div>
    `;
  }).join('');
}

// ==========================================================================
// DRILL-DOWN CIRCUIT MODAL
// ==========================================================================

function openCircuitModal(circuitKey, layerName) {
  const modal = document.getElementById('circuitModal');
  const catBadge = document.getElementById('modalCategoryBadge');
  const title = document.getElementById('modalBlockTitle');
  const subtitle = document.getElementById('modalBlockSubtitle');
  const body = document.getElementById('modalBody');

  if (!modal || !body) return;

  const data = INNER_CIRCUITS[circuitKey] || {
    title: layerName || 'Internal Mechanism',
    subtitle: 'Step-by-step tensor transformations and hardware notes',
    category: 'Layer Circuit',
    formula: 'y = f(x)',
    steps: [
      { step: '1. Input Ingestion', desc: 'Accepts input activations from previous stage.' },
      { step: '2. Transformation', desc: 'Performs parameter projection and non-linear activation.' }
    ],
    undertones: 'Executed using optimized CUDA kernels with tensor cores.',
    pytorch: 'x = layer(x)'
  };

  if (catBadge) catBadge.textContent = data.category;
  if (title) title.textContent = data.title;
  if (subtitle) subtitle.textContent = data.subtitle;

  const stepsHtml = (data.steps || []).map(s => `
    <div style="margin-bottom:0.75rem;padding:0.75rem;background:var(--bg-surface-low);border-radius:var(--radius-sm);border:1px solid var(--border-outline-variant)">
      <strong style="color:var(--primary);font-size:0.85rem">${s.step}</strong>
      <p style="color:var(--text-secondary);font-size:0.82rem;margin-top:0.25rem">${s.desc}</p>
    </div>
  `).join('');

  body.innerHTML = `
    <div style="background:rgba(15,23,42,0.9);border-left:3px solid var(--math-blue);padding:1rem;border-radius:0 6px 6px 0;margin-bottom:1.25rem">
      <span class="katex-render">$$${data.formula}$$</span>
    </div>

    <h4 style="font-size:0.95rem;font-weight:700;color:var(--text-primary);margin-bottom:0.75rem">Step-by-Step Computational Flow:</h4>
    ${stepsHtml}

    <div class="hardware-callout-box" style="margin-top:1rem">
      <strong>Hardware & Memory Undertones:</strong> ${data.undertones}
    </div>

    <div class="pytorch-code-container" style="margin-top:1rem">
      <pre><code>${data.pytorch}</code></pre>
      <button class="copy-code-btn" onclick="copyToClipboard('${data.pytorch.replace(/"/g, '&quot;')}', this)">Copy PyTorch</button>
    </div>
  `;

  modal.classList.add('open');
  modal.setAttribute('aria-hidden', 'false');
  triggerKaTeX();
}

function closeModal() {
  const modal = document.getElementById('circuitModal');
  if (modal) {
    modal.classList.remove('open');
    modal.setAttribute('aria-hidden', 'true');
  }
}

// Trigger KaTeX Auto-Render
function triggerKaTeX() {
  setTimeout(() => {
    if (window.renderMathInElement) {
      window.renderMathInElement(document.body, {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '$', right: '$', display: false }
        ],
        throwOnError: false
      });
    }
  }, 50);
}
