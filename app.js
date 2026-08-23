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
  },
  'mbconv_block': {
    title: 'Mobile Inverted Bottleneck (MBConvSD) Block',
    subtitle: '1x1 expansion, 3x3 depthwise conv, Squeeze-and-Excitation, and DropPath residual',
    category: 'Efficient CNN Architecture',
    formula: 'Y = X_{in} + \\text{DropPath}\\left(\\text{Conv}_{1\\times 1}\\left(\\text{SE}\\left(\\text{DWConv}_{3\\times 3}\\left(\\text{Conv}_{1\\times 1}(X_{in})\\right)\\right)\\right)\\right)',
    steps: [
      { step: '1. Inverted Channel Expansion', desc: 'Expand input channels from $C_{in}$ to $t \\cdot C_{in}$ ($t=4$ to $6$) using $1\\times 1$ pointwise convolution + BatchNorm + Swish activation.' },
      { step: '2. Depthwise Separable Spatial Filtering', desc: 'Apply independent $3\\times 3$ depthwise convolution per channel, decoupling spatial filtering from channel mixing with 8x-9x FLOPs reduction.' },
      { step: '3. Squeeze-and-Excitation Recalibration', desc: 'Compute channel-wise global average pooling $\\mathbf{z} = \\text{GAP}(\\mathbf{U})$ and dynamic channel gating $\\mathbf{s} = \\sigma(W_2 \\cdot \\text{Swish}(W_1 \\mathbf{z}))$ to scale features.' },
      { step: '4. Linear Bottleneck Projection', desc: 'Project expanded channels back to target output dimension $C_{out}$ without non-linearity to prevent manifold collapse.' },
      { step: '5. Stochastic Depth & Inverted Residual', desc: 'Add input $X_{in}$ to projected output using DropPath ($p=0.2$), creating gradient highways across all 7 stages.' }
    ],
    undertones: 'Concentrates 86.6% of the total 4.03M parameters in depthwise feature extraction, eliminating parameter waste in dense FC layers.',
    pytorch: `def forward(self, x):\n    identity = x\n    # 1. Expansion\n    if self.expand_conv:\n        x = self.expand_conv(x)\n    # 2. Depthwise\n    x = self.dw_conv(x)\n    # 3. Squeeze-and-Excitation\n    x = self.se_layer(x)\n    # 4. Pointwise Linear Projection\n    x = self.proj_conv(x)\n    # 5. Stochastic Depth & Residual\n    if self.use_res_connect:\n        x = identity + self.drop_path(x)\n    return x`
  },
  'se_attention': {
    title: 'Squeeze-and-Excitation (SE) Channel Attention',
    subtitle: 'Adaptive channel-wise feature recalibration with global context',
    category: 'Channel Attention Mechanism',
    formula: '\\mathbf{z}_c = \\frac{1}{H \\times W}\\sum_{i=1}^H \\sum_{j=1}^W u_c(i,j), \\quad \\mathbf{s} = \\sigma\\left(W_2 \\cdot \\text{Swish}(W_1 \\mathbf{z})\\right), \\quad \\tilde{\\mathbf{X}} = \\mathbf{s} \\odot \\mathbf{U}',
    steps: [
      { step: '1. Squeeze Operation (Global Information Embedding)', desc: 'Generate channel-wise statistics $\\mathbf{z} \\in \\mathbb{R}^C$ using Global Average Pooling (GAP) across spatial dimensions $H \\times W$.' },
      { step: '2. Excitation Operation (Adaptive Recalibration)', desc: 'Capture non-linear channel dependencies using bottleneck MLP with reduction ratio $r=4$: $\\mathbf{s} = \\sigma(W_2 \\cdot \\text{Swish}(W_1 \\mathbf{z}))$.' },
      { step: '3. Scale Operation (Channel Weighting)', desc: 'Rescale feature map $\\mathbf{U}$ by channel activation weights: $\\tilde{\\mathbf{x}}_c = s_c \\cdot \\mathbf{u}_c$, amplifying informative feature channels and suppressing noise.' }
    ],
    undertones: 'Adds less than 1.5% parameter overhead while consistently providing +1.5% to +2.0% top-1 accuracy gain on fine-grained classes like CIFAR-100.',
    pytorch: `def forward(self, x):\n    b, c, _, _ = x.shape\n    # Squeeze\n    z = x.mean((2, 3), keepdim=True)\n    # Excitation\n    s = torch.sigmoid(self.fc2(F.silu(self.fc1(z))))\n    # Scale\n    return x * s`
  },
  'the_head_funnel': {
    title: 'TheHead: Parameter-Free GAP Classification Funnel',
    subtitle: '1x1 expansion, spatial collapse, and linear class projection',
    category: 'CNN Classification Head',
    formula: '\\mathbf{v} = \\text{GAP}\\left(\\text{Conv}_{1\\times 1}(X_{body})\\right) \\in \\mathbb{R}^{1280}, \\quad \\mathbf{y}_{logits} = W_{fc} \\cdot \\text{Dropout}(\\mathbf{v}, P=0.2) + \\mathbf{b}',
    steps: [
      { step: '1. Layer 18: 1x1 Expansion Conv', desc: 'Expand final Stage 7 features from $4\\times 4\\times 320$ to $4\\times 4\\times 1280$ channels before pooling.' },
      { step: '2. Layer 19: Global Average Pooling', desc: 'Average all 16 spatial pixels ($4\\times 4$) into a single 1,280-dim feature vector, completely eliminating dense spatial parameters.' },
      { step: '3. Layer 20: Dropout Regularization', desc: 'Apply $P_{drop}=0.2$ dropout to feature activations to prevent classifier co-adaptation and memorization.' },
      { step: '4. Layer 21: Linear Classifier', desc: 'Project 1,280 features to 100 CIFAR-100 class output logits via single dense matrix multiplication ($1280 \\times 100$).' }
    ],
    undertones: 'Replaces multi-million parameter dense flattening layers with parameter-free GAP, preserving an ultra-compact 15.4 MB total model footprint.',
    pytorch: `def forward(self, x):\n    x = self.head_conv(x)       # (B, 1280, 4, 4)\n    v = self.gap(x).flatten(1)  # (B, 1280)\n    v = self.dropout(v)        # Dropout p=0.2\n    logits = self.classifier(v) # (B, 100)\n    return logits`
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
    'transmamba': { name: 'SSM Hybrid', badge: 'badge-transmamba', filter: 'ssm', formula: 'y = \\text{MHA}(\\text{SSM}(X)) + \\text{FFN}(X)', complexity: 'O(L · D)' },
    'p100supercnn': { name: 'Edge Vision CNN', badge: 'badge-attention', filter: 'cnn', formula: '\\text{Eff} = \\frac{\\text{Acc}}{\\text{MParams}} = 18.32, \\quad 4.03\\text{M Params} \\; (15.4\\text{MB})', complexity: 'O(H \\cdot W \\cdot C \\cdot K^2)' }
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
  renderPosterAnalytics(paper);
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

// ==========================================================================
// AUTHENTIC RESEARCH PAPER SVG ARCHITECTURE DIAGRAM RENDERERS
// ==========================================================================

function renderGraphicalCanvas(paper) {
  const viewport = document.getElementById('diagramCanvasViewport');
  const diagramTitle = document.getElementById('diagramTitle');
  if (!viewport || !paper) return;

  if (diagramTitle) diagramTitle.textContent = `${paper.title} — Canonical Architecture Diagram`;

  let svgHtml = '';
  switch (paper.paper_id) {
    case 'transformer':
      svgHtml = renderTransformerSVG();
      break;
    case 'mamba':
      svgHtml = renderMambaSVG();
      break;
    case 'mod':
      svgHtml = renderMoDSVG();
      break;
    case 'moe_survey':
      svgHtml = renderMoESVG();
      break;
    case 'swat':
      svgHtml = renderSWATSVG();
      break;
    case 'titans':
      svgHtml = renderTitansSVG();
      break;
    case 'transmamba':
      svgHtml = renderTransMambaSVG();
      break;
    case 'p100supercnn':
      svgHtml = renderP100SuperCNNSVG();
      break;
    default:
      svgHtml = renderTransformerSVG();
      break;
  }

  viewport.innerHTML = `
    <div class="paper-svg-diagram-wrapper" id="paperSvgWrapper">
      ${svgHtml}
    </div>
  `;
}

// 1. TRANSFORMER (Vaswani et al., 2017 - Figure 1)
function renderTransformerSVG() {
  return `
    <svg class="architecture-svg-root" viewBox="0 0 760 880" fill="none" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#919096" class="svg-arrow-head"/>
        </marker>
        <marker id="arrow-active" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#c5ff22"/>
        </marker>
      </defs>

      <!-- ENCODER SECTION (LEFT) -->
      <!-- Inputs & Embedding -->
      <text x="200" y="850" class="svg-label-title" style="font-size:15px; fill:#e5e2e2;">Inputs</text>
      <path d="M 200,835 L 200,805" class="svg-flow-path" marker-end="url(#arrow)"/>
      
      <g class="svg-node-interactive" onclick="openCircuitModal('multihead_attention', 'Input & Positional Embedding')">
        <rect x="110" y="760" width="180" height="44" rx="8" fill="rgba(244, 114, 182, 0.22)" stroke="#f472b6" stroke-width="1.8"/>
        <text x="200" y="782" class="svg-label-title">Input Embedding</text>
      </g>

      <path d="M 200,760 L 200,720" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Positional Encoding Left -->
      <circle cx="200" cy="700" r="14" fill="#201f20" stroke="#c5c5d2" stroke-width="1.8"/>
      <path d="M 194,700 L 206,700 M 200,694 L 200,706" stroke="#c5c5d2" stroke-width="2"/>
      
      <g class="svg-node-interactive" onclick="openCircuitModal('multihead_attention', 'Sinusoidal Positional Encoding')">
        <circle cx="70" cy="700" r="18" fill="rgba(79, 155, 255, 0.2)" stroke="#4f9bff" stroke-width="1.8"/>
        <path d="M 60,700 Q 65,692 70,700 T 80,700" stroke="#4f9bff" stroke-width="2.5" fill="none"/>
        <text x="70" y="734" class="svg-label-sub">Positional Encoding</text>
      </g>
      <path d="M 88,700 L 186,700" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Encoder Nx Outer Box -->
      <rect x="85" y="315" width="230" height="365" rx="16" fill="rgba(30, 41, 59, 0.25)" stroke="#64748b" stroke-width="1.5" stroke-dasharray="6 6"/>
      <text x="58" y="495" class="svg-label-title" style="font-size:18px; fill:#c5ff22; font-family:'JetBrains Mono';">N×</text>

      <path d="M 200,686 L 200,640" class="svg-flow-path"/>

      <!-- 3-way arrow branch for Q, K, V into Multi-Head Attention -->
      <path d="M 200,640 L 140,640 L 140,620" class="svg-flow-path" marker-end="url(#arrow)"/>
      <path d="M 200,640 L 200,620" class="svg-flow-path" marker-end="url(#arrow)"/>
      <path d="M 200,640 L 260,640 L 260,620" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Residual bypass around MHA -->
      <path d="M 200,650 L 98,650 L 98,510 L 110,510" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Multi-Head Attention Box -->
      <g class="svg-node-interactive" onclick="openCircuitModal('multihead_attention', 'Multi-Head Self-Attention')">
        <rect x="110" y="570" width="180" height="50" rx="8" fill="rgba(251, 146, 60, 0.25)" stroke="#fb923c" stroke-width="1.8"/>
        <text x="200" y="590" class="svg-label-title">Multi-Head</text>
        <text x="200" y="606" class="svg-label-title">Attention</text>
      </g>

      <path d="M 200,570 L 200,530" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Add & Norm 1 Box -->
      <g class="svg-node-interactive" onclick="openCircuitModal('multihead_attention', 'Add & LayerNorm 1')">
        <rect x="110" y="490" width="180" height="40" rx="8" fill="rgba(253, 224, 71, 0.22)" stroke="#fde047" stroke-width="1.8"/>
        <text x="200" y="510" class="svg-label-title">Add & Norm</text>
      </g>

      <path d="M 200,490 L 200,455" class="svg-flow-path"/>

      <!-- Residual bypass around FFN -->
      <path d="M 200,470 L 98,470 L 98,345 L 110,345" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Feed Forward Box -->
      <g class="svg-node-interactive" onclick="openCircuitModal('multihead_attention', 'Position-wise Feed-Forward Network')">
        <rect x="110" y="405" width="180" height="50" rx="8" fill="rgba(79, 155, 255, 0.25)" stroke="#4f9bff" stroke-width="1.8"/>
        <text x="200" y="425" class="svg-label-title">Feed</text>
        <text x="200" y="441" class="svg-label-title">Forward</text>
      </g>

      <path d="M 200,405 L 200,365" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Add & Norm 2 Box -->
      <g class="svg-node-interactive" onclick="openCircuitModal('multihead_attention', 'Add & LayerNorm 2')">
        <rect x="110" y="325" width="180" height="40" rx="8" fill="rgba(253, 224, 71, 0.22)" stroke="#fde047" stroke-width="1.8"/>
        <text x="200" y="345" class="svg-label-title">Add & Norm</text>
      </g>

      <!-- Cross-Attention Connection from Encoder to Decoder -->
      <path d="M 200,325 L 200,285 L 350,285 L 350,445 L 430,445" class="svg-flow-path" stroke="#c5ff22" stroke-width="2.2" marker-end="url(#arrow)"/>
      <path d="M 350,445 L 350,460 L 460,460 L 460,475" class="svg-flow-path" stroke="#c5ff22" stroke-width="2.2" marker-end="url(#arrow)"/>
      <path d="M 350,445 L 350,475 L 490,475" class="svg-flow-path" stroke="#c5ff22" stroke-width="2.2" marker-end="url(#arrow)"/>

      <!-- DECODER SECTION (RIGHT) -->
      <!-- Outputs shifted right -->
      <text x="520" y="850" class="svg-label-title" style="font-size:15px; fill:#e5e2e2;">Outputs (shifted right)</text>
      <path d="M 520,835 L 520,805" class="svg-flow-path" marker-end="url(#arrow)"/>

      <g class="svg-node-interactive" onclick="openCircuitModal('multihead_attention', 'Output Embedding')">
        <rect x="430" y="760" width="180" height="44" rx="8" fill="rgba(244, 114, 182, 0.22)" stroke="#f472b6" stroke-width="1.8"/>
        <text x="520" y="782" class="svg-label-title">Output Embedding</text>
      </g>

      <path d="M 520,760 L 520,720" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Positional Encoding Right -->
      <circle cx="520" cy="700" r="14" fill="#201f20" stroke="#c5c5d2" stroke-width="1.8"/>
      <path d="M 514,700 L 526,700 M 520,694 L 520,706" stroke="#c5c5d2" stroke-width="2"/>
      
      <g class="svg-node-interactive" onclick="openCircuitModal('multihead_attention', 'Sinusoidal Positional Encoding')">
        <circle cx="660" cy="700" r="18" fill="rgba(79, 155, 255, 0.2)" stroke="#4f9bff" stroke-width="1.8"/>
        <path d="M 650,700 Q 655,692 660,700 T 670,700" stroke="#4f9bff" stroke-width="2.5" fill="none"/>
        <text x="660" y="734" class="svg-label-sub">Positional Encoding</text>
      </g>
      <path d="M 642,700 L 534,700" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Decoder Nx Outer Box -->
      <rect x="405" y="195" width="230" height="485" rx="16" fill="rgba(30, 41, 59, 0.25)" stroke="#64748b" stroke-width="1.5" stroke-dasharray="6 6"/>
      <text x="650" y="440" class="svg-label-title" style="font-size:18px; fill:#c5ff22; font-family:'JetBrains Mono';">N×</text>

      <path d="M 520,686 L 520,650" class="svg-flow-path"/>

      <!-- 3-way arrow into Masked Multi-Head Attention -->
      <path d="M 520,650 L 460,650 L 460,630" class="svg-flow-path" marker-end="url(#arrow)"/>
      <path d="M 520,650 L 520,630" class="svg-flow-path" marker-end="url(#arrow)"/>
      <path d="M 520,650 L 580,650 L 580,630" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Residual bypass around Masked MHA -->
      <path d="M 520,660 L 622,660 L 622,530 L 610,530" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Masked Multi-Head Attention Box -->
      <g class="svg-node-interactive" onclick="openCircuitModal('multihead_attention', 'Masked Causal Multi-Head Attention')">
        <rect x="430" y="580" width="180" height="50" rx="8" fill="rgba(251, 146, 60, 0.25)" stroke="#fb923c" stroke-width="1.8"/>
        <text x="520" y="600" class="svg-label-title">Masked Multi-Head</text>
        <text x="520" y="616" class="svg-label-title">Attention</text>
      </g>

      <path d="M 520,580 L 520,550" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Decoder Add & Norm 1 -->
      <g class="svg-node-interactive" onclick="openCircuitModal('multihead_attention', 'Decoder Add & LayerNorm 1')">
        <rect x="430" y="510" width="180" height="40" rx="8" fill="rgba(253, 224, 71, 0.22)" stroke="#fde047" stroke-width="1.8"/>
        <text x="520" y="530" class="svg-label-title">Add & Norm</text>
      </g>

      <!-- Query input arrow into Cross Attention -->
      <path d="M 520,510 L 520,465" class="svg-flow-path"/>
      <path d="M 520,490 L 622,490 L 622,355 L 610,355" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Cross Multi-Head Attention Box -->
      <g class="svg-node-interactive" onclick="openCircuitModal('multihead_attention', 'Encoder-Decoder Cross Attention')">
        <rect x="430" y="415" width="180" height="50" rx="8" fill="rgba(251, 146, 60, 0.25)" stroke="#fb923c" stroke-width="1.8"/>
        <text x="520" y="435" class="svg-label-title">Multi-Head</text>
        <text x="520" y="451" class="svg-label-title">Attention</text>
      </g>

      <path d="M 520,415 L 520,375" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Decoder Add & Norm 2 -->
      <g class="svg-node-interactive" onclick="openCircuitModal('multihead_attention', 'Decoder Add & LayerNorm 2')">
        <rect x="430" y="335" width="180" height="40" rx="8" fill="rgba(253, 224, 71, 0.22)" stroke="#fde047" stroke-width="1.8"/>
        <text x="520" y="355" class="svg-label-title">Add & Norm</text>
      </g>

      <path d="M 520,335 L 520,300" class="svg-flow-path"/>
      <path d="M 520,315 L 622,315 L 622,190 L 610,190" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Decoder Feed Forward Box -->
      <g class="svg-node-interactive" onclick="openCircuitModal('multihead_attention', 'Decoder Feed-Forward Network')">
        <rect x="430" y="250" width="180" height="50" rx="8" fill="rgba(79, 155, 255, 0.25)" stroke="#4f9bff" stroke-width="1.8"/>
        <text x="520" y="270" class="svg-label-title">Feed</text>
        <text x="520" y="286" class="svg-label-title">Forward</text>
      </g>

      <path d="M 520,250 L 520,210" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Decoder Add & Norm 3 -->
      <g class="svg-node-interactive" onclick="openCircuitModal('multihead_attention', 'Decoder Add & LayerNorm 3')">
        <rect x="430" y="170" width="180" height="40" rx="8" fill="rgba(253, 224, 71, 0.22)" stroke="#fde047" stroke-width="1.8"/>
        <text x="520" y="190" class="svg-label-title">Add & Norm</text>
      </g>

      <path d="M 520,170 L 520,142" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Linear Projection -->
      <g class="svg-node-interactive" onclick="openCircuitModal('multihead_attention', 'Final Linear Projection')">
        <rect x="430" y="100" width="180" height="42" rx="8" fill="rgba(192, 132, 252, 0.25)" stroke="#c084fc" stroke-width="1.8"/>
        <text x="520" y="121" class="svg-label-title">Linear</text>
      </g>

      <path d="M 520,100 L 520,83" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Softmax -->
      <g class="svg-node-interactive" onclick="openCircuitModal('multihead_attention', 'Softmax Probability Layer')">
        <rect x="430" y="45" width="180" height="38" rx="8" fill="rgba(74, 222, 128, 0.25)" stroke="#4ade80" stroke-width="1.8"/>
        <text x="520" y="64" class="svg-label-title">Softmax</text>
      </g>

      <path d="M 520,45 L 520,25" class="svg-flow-path" marker-end="url(#arrow)"/>
      <text x="520" y="15" class="svg-label-title" style="font-size:15px; fill:#c5ff22;">Output Probabilities</text>
    </svg>
    <div class="paper-figure-caption">Figure 1: The Transformer — model architecture (Vaswani et al., Attention Is All You Need, 2017)</div>
  `;
}

// 2. MAMBA ARCHITECTURE (Gu & Dao, 2023 - Figure 3)
function renderMambaSVG() {
  return `
    <svg class="architecture-svg-root" viewBox="0 0 760 760" fill="none" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#919096" class="svg-arrow-head"/>
        </marker>
      </defs>

      <!-- Input Token -->
      <text x="360" y="730" class="svg-label-title" style="font-size:16px; fill:#e5e2e2;">Input Token x ∈ ℝ^{B × L × D}</text>
      <path d="M 360,710 L 360,670" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Mamba Outer Block Container -->
      <rect x="90" y="110" width="560" height="540" rx="16" fill="rgba(30, 41, 59, 0.25)" stroke="#64748b" stroke-width="1.5" stroke-dasharray="6 6"/>
      <text x="115" y="138" class="svg-label-title" style="font-size:15px; fill:#c5ff22; font-family:'JetBrains Mono';">Mamba Block (S6 Selective SSM)</text>

      <!-- Residual Bypass Line -->
      <path d="M 360,670 L 610,670 L 610,80 L 380,80" class="svg-flow-path" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)"/>

      <!-- Input Split to Left & Right Branches -->
      <path d="M 360,670 L 235,670 L 235,600" class="svg-flow-path" marker-end="url(#arrow)"/>
      <path d="M 360,670 L 485,670 L 485,600" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- LEFT BRANCH (SSM Path) -->
      <g class="svg-node-interactive" onclick="openCircuitModal('selective_ssm', 'Linear Expansion (D → 2D)')">
        <rect x="140" y="555" width="190" height="45" rx="8" fill="rgba(192, 132, 252, 0.22)" stroke="#c084fc" stroke-width="1.8"/>
        <text x="235" y="577" class="svg-label-title">Linear (D → 2D)</text>
      </g>

      <path d="M 235,555 L 235,515" class="svg-flow-path" marker-end="url(#arrow)"/>

      <g class="svg-node-interactive" onclick="openCircuitModal('selective_ssm', '1D Causal Convolution (Kernel Size=4)')">
        <rect x="140" y="470" width="190" height="45" rx="8" fill="rgba(79, 155, 255, 0.25)" stroke="#4f9bff" stroke-width="1.8"/>
        <text x="235" y="492" class="svg-label-title">1D Conv (Kernel=4)</text>
      </g>

      <path d="M 235,470 L 235,435" class="svg-flow-path" marker-end="url(#arrow)"/>

      <g class="svg-node-interactive" onclick="openCircuitModal('selective_ssm', 'SiLU Non-linear Activation')">
        <rect x="140" y="395" width="190" height="40" rx="8" fill="rgba(74, 222, 128, 0.22)" stroke="#4ade80" stroke-width="1.8"/>
        <text x="235" y="415" class="svg-label-title">SiLU Activation</text>
      </g>

      <path d="M 235,395 L 235,360" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- S6 Core -->
      <g class="svg-node-interactive" onclick="openCircuitModal('selective_ssm', 'S6 Selective Discretization & SRAM Parallel Scan')">
        <rect x="125" y="270" width="220" height="90" rx="10" fill="rgba(251, 146, 60, 0.3)" stroke="#fb923c" stroke-width="2"/>
        <text x="235" y="295" class="svg-label-title" style="font-size:15px; fill:#fb923c;">SSM (S6 Core)</text>
        <text x="235" y="318" class="svg-label-sub">Δ = softplus(Parameter + s_Δ(x))</text>
        <text x="235" y="338" class="svg-label-sub">B(x), C(x) Dynamic Projections</text>
      </g>

      <!-- RIGHT BRANCH (Gating Path) -->
      <g class="svg-node-interactive" onclick="openCircuitModal('selective_ssm', 'Linear Gating Projection (D → 2D)')">
        <rect x="390" y="555" width="190" height="45" rx="8" fill="rgba(192, 132, 252, 0.22)" stroke="#c084fc" stroke-width="1.8"/>
        <text x="485" y="577" class="svg-label-title">Linear Gating (D → 2D)</text>
      </g>

      <path d="M 485,555 L 485,435" class="svg-flow-path" marker-end="url(#arrow)"/>

      <g class="svg-node-interactive" onclick="openCircuitModal('selective_ssm', 'Gating SiLU Activation')">
        <rect x="390" y="395" width="190" height="40" rx="8" fill="rgba(74, 222, 128, 0.22)" stroke="#4ade80" stroke-width="1.8"/>
        <text x="485" y="415" class="svg-label-title">SiLU Activation</text>
      </g>

      <!-- Convergence to Hadamard Product ⊗ -->
      <path d="M 235,270 L 235,220 L 342,220" class="svg-flow-path" marker-end="url(#arrow)"/>
      <path d="M 485,395 L 485,220 L 378,220" class="svg-flow-path" marker-end="url(#arrow)"/>

      <circle cx="360" cy="220" r="18" fill="#201f20" stroke="#fde047" stroke-width="2"/>
      <text x="360" y="221" class="svg-label-title" style="font-size:18px; fill:#fde047;">⊗</text>

      <path d="M 360,202 L 360,165" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Output Linear Projection -->
      <g class="svg-node-interactive" onclick="openCircuitModal('selective_ssm', 'Linear Output Projection (2D → D)')">
        <rect x="265" y="125" width="190" height="40" rx="8" fill="rgba(192, 132, 252, 0.22)" stroke="#c084fc" stroke-width="1.8"/>
        <text x="360" y="145" class="svg-label-title">Linear (2D → D)</text>
      </g>

      <path d="M 360,125 L 360,98" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Residual Add ⊕ -->
      <circle cx="360" cy="80" r="16" fill="#201f20" stroke="#c5c5d2" stroke-width="2"/>
      <path d="M 353,80 L 367,80 M 360,73 L 360,87" stroke="#c5c5d2" stroke-width="2"/>

      <path d="M 360,64 L 360,35" class="svg-flow-path" marker-end="url(#arrow)"/>
      <text x="360" y="20" class="svg-label-title" style="font-size:16px; fill:#c5ff22;">Output y ∈ ℝ^{B × L × D}</text>
    </svg>
    <div class="paper-figure-caption">Figure 3: The Mamba Block — Selective State Space Architecture (Gu & Dao, 2023)</div>
  `;
}

// 3. MIXTURE-OF-DEPTHS (Raposo et al., DeepMind 2024 - Figure 1 & 2)
function renderMoDSVG() {
  return `
    <svg class="architecture-svg-root" viewBox="0 0 760 760" fill="none" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#919096" class="svg-arrow-head"/>
        </marker>
      </defs>

      <text x="360" y="730" class="svg-label-title" style="font-size:16px; fill:#e5e2e2;">Input Sequence X ∈ ℝ^{B × L × D}</text>
      <path d="M 360,710 L 360,660" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Router Scoring -->
      <g class="svg-node-interactive" onclick="openCircuitModal('topk_routing', 'Token Router Scoring r_i = w_r^T x_i')">
        <rect x="250" y="610" width="220" height="50" rx="8" fill="rgba(253, 224, 71, 0.25)" stroke="#fde047" stroke-width="1.8"/>
        <text x="360" y="628" class="svg-label-title">Router Scoring</text>
        <text x="360" y="646" class="svg-label-sub">r_i = w_r^T x_i ∈ ℝ</text>
      </g>

      <path d="M 360,610 L 360,565" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Top-k Capacity Selector -->
      <g class="svg-node-interactive" onclick="openCircuitModal('topk_routing', 'Top-k Capacity Selector (k = floor(c * L))')">
        <rect x="230" y="505" width="260" height="60" rx="10" fill="rgba(251, 146, 60, 0.3)" stroke="#fb923c" stroke-width="2"/>
        <text x="360" y="527" class="svg-label-title">Top-K Capacity Selector</text>
        <text x="360" y="548" class="svg-label-sub">k = ⌊c · L⌋ Tokens (e.g. 50% Capacity)</text>
      </g>

      <!-- Dynamic Branching -->
      <!-- Routed Path (Left) -->
      <path d="M 270,505 L 200,450" class="svg-flow-path" stroke="#4ade80" stroke-width="2.5" marker-end="url(#arrow)"/>
      <text x="140" y="480" class="svg-label-sub" style="fill:#4ade80; font-weight:700;">Top-k Tokens</text>

      <!-- Skipped Path (Right) -->
      <path d="M 450,505 L 560,505 L 560,180 L 465,180" class="svg-flow-path" stroke="#fb923c" stroke-width="2" stroke-dasharray="4 4" marker-end="url(#arrow)"/>
      <text x="590" y="340" class="svg-label-sub" style="fill:#fb923c;">Skipped Tokens (Residual Identity)</text>

      <!-- Transformer Computation on Selected Tokens -->
      <g class="svg-node-interactive" onclick="openCircuitModal('topk_routing', 'Self-Attention Block on Top-K Tokens')">
        <rect x="100" y="390" width="200" height="50" rx="8" fill="rgba(79, 155, 255, 0.25)" stroke="#4f9bff" stroke-width="1.8"/>
        <text x="200" y="410" class="svg-label-title">Multi-Head Attention</text>
        <text x="200" y="426" class="svg-label-sub">Top-k Tokens Only</text>
      </g>

      <path d="M 200,390 L 200,350" class="svg-flow-path" marker-end="url(#arrow)"/>

      <g class="svg-node-interactive" onclick="openCircuitModal('topk_routing', 'Feed-Forward Block on Top-K Tokens')">
        <rect x="100" y="295" width="200" height="50" rx="8" fill="rgba(79, 155, 255, 0.25)" stroke="#4f9bff" stroke-width="1.8"/>
        <text x="200" y="315" class="svg-label-title">MLP / Feed Forward</text>
        <text x="200" y="331" class="svg-label-sub">Top-k Tokens Only</text>
      </g>

      <path d="M 200,295 L 200,255" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Router Weight Scaling -->
      <g class="svg-node-interactive" onclick="openCircuitModal('topk_routing', 'Router Probability Weighting P(r_i) * f(x_i)')">
        <rect x="100" y="210" width="200" height="45" rx="8" fill="rgba(253, 224, 71, 0.22)" stroke="#fde047" stroke-width="1.8"/>
        <text x="200" y="230" class="svg-label-title">Router Weight Scaling</text>
        <text x="200" y="244" class="svg-label-sub">P(r_i) · f(x_i)</text>
      </g>

      <path d="M 200,210 L 200,180 L 255,180" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Recombination: Weighted Scatter-Add -->
      <g class="svg-node-interactive" onclick="openCircuitModal('topk_routing', 'Weighted Scatter-Add Recombination')">
        <rect x="235" y="150" width="250" height="60" rx="10" fill="rgba(74, 222, 128, 0.28)" stroke="#4ade80" stroke-width="2"/>
        <text x="360" y="173" class="svg-label-title" style="font-size:15px; fill:#4ade80;">Scatter-Add Recombination</text>
        <text x="360" y="193" class="svg-label-sub">Preserves Original Causal Sequence Order</text>
      </g>

      <path d="M 360,150 L 360,100" class="svg-flow-path" marker-end="url(#arrow)"/>
      <text x="360" y="70" class="svg-label-title" style="font-size:16px; fill:#c5ff22;">Output Sequence Y ∈ ℝ^{B × L × D}</text>
    </svg>
    <div class="paper-figure-caption">Figure 1 & 2: Mixture-of-Depths Dynamic Allocation (Raposo et al., DeepMind 2024)</div>
  `;
}

// 4. MIXTURE OF EXPERTS (Mu & Lin, 2024 / Shazeer et al.)
function renderMoESVG() {
  return `
    <svg class="architecture-svg-root" viewBox="0 0 760 760" fill="none" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#919096" class="svg-arrow-head"/>
        </marker>
      </defs>

      <text x="360" y="730" class="svg-label-title" style="font-size:16px; fill:#e5e2e2;">Token Input x</text>
      <path d="M 360,710 L 360,655" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Residual Bypass line -->
      <path d="M 360,680 L 680,680 L 680,150 L 380,150" class="svg-flow-path" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)"/>

      <!-- Sparse Gating Router -->
      <g class="svg-node-interactive" onclick="openCircuitModal('topk_routing', 'Sparse Softmax Gating Router G(x) = Softmax(TopK(H(x), k))')">
        <rect x="230" y="590" width="260" height="65" rx="10" fill="rgba(251, 146, 60, 0.3)" stroke="#fb923c" stroke-width="2"/>
        <text x="360" y="613" class="svg-label-title">Sparse Gating Router</text>
        <text x="360" y="634" class="svg-label-sub">G(x) = Softmax(TopK(H(x), k))</text>
      </g>

      <!-- Router branching into 4 Experts -->
      <path d="M 280,590 L 130,490" class="svg-flow-path" marker-end="url(#arrow)"/>
      <path d="M 330,590 L 280,490" class="svg-flow-path" marker-end="url(#arrow)"/>
      <path d="M 390,590 L 440,490" class="svg-flow-path" stroke-dasharray="4 4" marker-end="url(#arrow)"/>
      <path d="M 440,590 L 590,490" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Expert 1 -->
      <g class="svg-node-interactive" onclick="openCircuitModal('topk_routing', 'Expert 1 FFN')">
        <rect x="70" y="410" width="120" height="80" rx="8" fill="rgba(79, 155, 255, 0.25)" stroke="#4f9bff" stroke-width="1.8"/>
        <text x="130" y="440" class="svg-label-title">Expert 1</text>
        <text x="130" y="460" class="svg-label-sub">FFN (Active)</text>
      </g>

      <!-- Expert 2 -->
      <g class="svg-node-interactive" onclick="openCircuitModal('topk_routing', 'Expert 2 FFN')">
        <rect x="220" y="410" width="120" height="80" rx="8" fill="rgba(79, 155, 255, 0.25)" stroke="#4f9bff" stroke-width="1.8"/>
        <text x="280" y="440" class="svg-label-title">Expert 2</text>
        <text x="280" y="460" class="svg-label-sub">FFN (Active)</text>
      </g>

      <!-- Expert 3 (Inactive) -->
      <g class="svg-node-interactive" onclick="openCircuitModal('topk_routing', 'Expert 3 FFN (Skipped in Top-2)')">
        <rect x="380" y="410" width="120" height="80" rx="8" fill="rgba(30, 41, 59, 0.2)" stroke="#64748b" stroke-width="1.5" stroke-dasharray="4 4"/>
        <text x="440" y="440" class="svg-label-title" style="fill:#919096;">Expert 3</text>
        <text x="440" y="460" class="svg-label-sub">Skipped (Zero)</text>
      </g>

      <!-- Expert N -->
      <g class="svg-node-interactive" onclick="openCircuitModal('topk_routing', 'Expert N FFN')">
        <rect x="530" y="410" width="120" height="80" rx="8" fill="rgba(79, 155, 255, 0.25)" stroke="#4f9bff" stroke-width="1.8"/>
        <text x="590" y="440" class="svg-label-title">Expert N</text>
        <text x="590" y="460" class="svg-label-sub">FFN (Inactive)</text>
      </g>

      <!-- Convergence to Weighted Summation Σ -->
      <path d="M 130,410 L 130,340 L 330,300" class="svg-flow-path" marker-end="url(#arrow)"/>
      <path d="M 280,410 L 280,340 L 340,300" class="svg-flow-path" marker-end="url(#arrow)"/>
      <path d="M 590,410 L 590,340 L 380,300" class="svg-flow-path" stroke-dasharray="4 4" marker-end="url(#arrow)"/>

      <g class="svg-node-interactive" onclick="openCircuitModal('topk_routing', 'Weighted Combination: y = sum(G(x)_i * E_i(x))')">
        <rect x="240" y="240" width="240" height="60" rx="10" fill="rgba(74, 222, 128, 0.28)" stroke="#4ade80" stroke-width="2"/>
        <text x="360" y="263" class="svg-label-title">Weighted Summation Σ</text>
        <text x="360" y="283" class="svg-label-sub">y = ∑ G(x)_i · E_i(x)</text>
      </g>

      <path d="M 360,240 L 360,170" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Residual Add ⊕ -->
      <circle cx="360" cy="150" r="16" fill="#201f20" stroke="#c5c5d2" stroke-width="2"/>
      <path d="M 353,150 L 367,150 M 360,143 L 360,157" stroke="#c5c5d2" stroke-width="2"/>

      <path d="M 360,134 L 360,90" class="svg-flow-path" marker-end="url(#arrow)"/>
      <text x="360" y="60" class="svg-label-title" style="font-size:16px; fill:#c5ff22;">Output Representation</text>
    </svg>
    <div class="paper-figure-caption">Figure 2: Sparse Mixture of Experts Architecture (Mu & Lin, 2024)</div>
  `;
}

// 5. SWAT ARCHITECTURE (Fu et al., 2025/2026 - Figure 1)
function renderSWATSVG() {
  return `
    <svg class="architecture-svg-root" viewBox="0 0 760 760" fill="none" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#919096" class="svg-arrow-head"/>
        </marker>
      </defs>

      <text x="360" y="730" class="svg-label-title" style="font-size:16px; fill:#e5e2e2;">Input Sequence (Long Context L ≥ 32k)</text>
      <path d="M 360,710 L 360,660" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Sliding Window Partitioning -->
      <g class="svg-node-interactive" onclick="openCircuitModal('sigmoid_attention', 'Sliding Window Token Partitioning (W=2048)')">
        <rect x="235" y="605" width="250" height="55" rx="8" fill="rgba(244, 114, 182, 0.25)" stroke="#f472b6" stroke-width="1.8"/>
        <text x="360" y="626" class="svg-label-title">Sliding Window Partitioning</text>
        <text x="360" y="644" class="svg-label-sub">Fixed Window Size W = 2,048</text>
      </g>

      <path d="M 360,605 L 360,555" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Q, K, V Projections -->
      <g class="svg-node-interactive" onclick="openCircuitModal('sigmoid_attention', 'Q, K, V Projections')">
        <rect x="235" y="500" width="250" height="55" rx="8" fill="rgba(192, 132, 252, 0.25)" stroke="#c084fc" stroke-width="1.8"/>
        <text x="360" y="521" class="svg-label-title">Q, K, V Projections</text>
        <text x="360" y="539" class="svg-label-sub">Q = X W_Q,  K = X W_K,  V = X W_V</text>
      </g>

      <path d="M 360,500 L 360,450" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Sigmoid Attention Core with ALiBi Bias & KV Eviction -->
      <!-- Left: Frequency Decay Bias -->
      <g class="svg-node-interactive" onclick="openCircuitModal('sigmoid_attention', 'ALiBi / RoPE Frequency Bias Decay')">
        <rect x="70" y="350" width="140" height="95" rx="8" fill="rgba(253, 224, 71, 0.22)" stroke="#fde047" stroke-width="1.8"/>
        <text x="140" y="380" class="svg-label-title">Decay Bias</text>
        <text x="140" y="405" class="svg-label-sub">-γ |i - j|</text>
      </g>
      <path d="M 210,397 L 235,397" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Center: Sigmoid Core -->
      <g class="svg-node-interactive" onclick="openCircuitModal('sigmoid_attention', 'Sigmoid Attention Matrix Core')">
        <rect x="235" y="340" width="250" height="110" rx="10" fill="rgba(251, 146, 60, 0.3)" stroke="#fb923c" stroke-width="2"/>
        <text x="360" y="370" class="svg-label-title" style="font-size:15px; fill:#fb923c;">Sigmoid Attention Core</text>
        <text x="360" y="395" class="svg-label-sub">A_ij = σ( (q_i k_j^T)/√d_k - γ|i-j| )</text>
        <text x="360" y="420" class="svg-label-sub">No Global Softmax Reduction!</text>
      </g>

      <!-- Right: Bounded KV Buffer -->
      <g class="svg-node-interactive" onclick="openCircuitModal('sigmoid_attention', 'Bounded KV Buffer Eviction')">
        <rect x="510" y="350" width="150" height="95" rx="8" fill="rgba(79, 155, 255, 0.25)" stroke="#4f9bff" stroke-width="1.8"/>
        <text x="585" y="380" class="svg-label-title">KV Buffer</text>
        <text x="585" y="405" class="svg-label-sub">O(W) Fixed RAM</text>
      </g>
      <path d="M 485,397 L 510,397" class="svg-flow-path" marker-end="url(#arrow)"/>

      <path d="M 360,340 L 360,285" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Feed-Forward Block -->
      <g class="svg-node-interactive" onclick="openCircuitModal('sigmoid_attention', 'Feed-Forward Network')">
        <rect x="235" y="230" width="250" height="55" rx="8" fill="rgba(79, 155, 255, 0.25)" stroke="#4f9bff" stroke-width="1.8"/>
        <text x="360" y="251" class="svg-label-title">SwiGLU Feed-Forward</text>
        <text x="360" y="269" class="svg-label-sub">Linear & LayerNorm</text>
      </g>

      <path d="M 360,230 L 360,165" class="svg-flow-path" marker-end="url(#arrow)"/>

      <text x="360" y="130" class="svg-label-title" style="font-size:16px; fill:#c5ff22;">Output Representations (O(L · W) Complexity)</text>
    </svg>
    <div class="paper-figure-caption">Figure 1: SWAT — Sliding Window Sigmoid Attention (Fu et al., 2025/2026)</div>
  `;
}

// 6. TITANS ARCHITECTURE (Behrouz et al., Google Research 2024 - Figure 2)
function renderTitansSVG() {
  return `
    <svg class="architecture-svg-root" viewBox="0 0 760 760" fill="none" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#919096" class="svg-arrow-head"/>
        </marker>
      </defs>

      <text x="360" y="730" class="svg-label-title" style="font-size:16px; fill:#e5e2e2;">Input Sequence x_t</text>
      <path d="M 360,710 L 360,650" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Split to Short-Term and Long-Term -->
      <path d="M 360,650 L 220,650 L 220,580" class="svg-flow-path" marker-end="url(#arrow)"/>
      <path d="M 360,650 L 500,650 L 500,580" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- LEFT: Short-Term Memory Core (Sliding Attention) -->
      <g class="svg-node-interactive" onclick="openCircuitModal('multihead_attention', 'Short-Term Attention Core (Sliding Window Attention)')">
        <rect x="110" y="470" width="220" height="110" rx="10" fill="rgba(79, 155, 255, 0.25)" stroke="#4f9bff" stroke-width="2"/>
        <text x="220" y="500" class="svg-label-title" style="fill:#4f9bff;">Short-Term Memory</text>
        <text x="220" y="525" class="svg-label-title">Attention Core</text>
        <text x="220" y="550" class="svg-label-sub">High-precision local context</text>
      </g>

      <!-- RIGHT: Long-Term Neural Memory Core -->
      <g class="svg-node-interactive" onclick="openCircuitModal('selective_ssm', 'Long-Term Neural Memory Core (Surprise Metric + Associative Update)')">
        <rect x="390" y="470" width="240" height="110" rx="10" fill="rgba(74, 222, 128, 0.28)" stroke="#4ade80" stroke-width="2"/>
        <text x="510" y="500" class="svg-label-title" style="fill:#4ade80;">Long-Term Neural Memory</text>
        <text x="510" y="525" class="svg-label-sub">M_t = (1 - α_t) M_{t-1} + S_t</text>
        <text x="510" y="550" class="svg-label-sub">Surprise Metric S_t Memorization</text>
      </g>

      <!-- Convergence to Memory-Context Fusion Gate -->
      <path d="M 220,470 L 220,380 L 310,330" class="svg-flow-path" marker-end="url(#arrow)"/>
      <path d="M 510,470 L 510,380 L 410,330" class="svg-flow-path" marker-end="url(#arrow)"/>

      <g class="svg-node-interactive" onclick="openCircuitModal('topk_routing', 'Memory-Context Gated Fusion Unit G_t')">
        <rect x="235" y="270" width="250" height="70" rx="10" fill="rgba(253, 224, 71, 0.25)" stroke="#fde047" stroke-width="2"/>
        <text x="360" y="295" class="svg-label-title">Memory-Context Fusion Gate</text>
        <text x="360" y="318" class="svg-label-sub">y_t = G_t · y_{attn} + (1 - G_t) · y_{mem}</text>
      </g>

      <path d="M 360,270 L 360,210" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Output Feedforward & Projection -->
      <g class="svg-node-interactive" onclick="openCircuitModal('multihead_attention', 'Final Feedforward & Output Projection')">
        <rect x="235" y="150" width="250" height="60" rx="8" fill="rgba(192, 132, 252, 0.25)" stroke="#c084fc" stroke-width="1.8"/>
        <text x="360" y="173" class="svg-label-title">Feedforward & LayerNorm</text>
        <text x="360" y="193" class="svg-label-sub">Residual Add & Projection</text>
      </g>

      <path d="M 360,150 L 360,90" class="svg-flow-path" marker-end="url(#arrow)"/>

      <text x="360" y="60" class="svg-label-title" style="font-size:16px; fill:#c5ff22;">Output Representation y_t</text>
    </svg>
    <div class="paper-figure-caption">Figure 2: Titans — Learning to Memorize at Test Time (Behrouz et al., Google Research 2024)</div>
  `;
}

// 7. TRANSMAMBA ARCHITECTURE (2024 - Figure 1)
function renderTransMambaSVG() {
  return `
    <svg class="architecture-svg-root" viewBox="0 0 760 760" fill="none" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#919096" class="svg-arrow-head"/>
        </marker>
      </defs>

      <text x="360" y="730" class="svg-label-title" style="font-size:16px; fill:#e5e2e2;">Input Sequence X ∈ ℝ^{B × L × D}</text>
      <path d="M 360,710 L 360,650" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Stage 1: Mamba SSM Block -->
      <g class="svg-node-interactive" onclick="openCircuitModal('selective_ssm', 'Stage 1: Mamba S6 Selective State Space Block')">
        <rect x="220" y="580" width="280" height="70" rx="10" fill="rgba(251, 146, 60, 0.3)" stroke="#fb923c" stroke-width="2"/>
        <text x="360" y="605" class="svg-label-title" style="fill:#fb923c;">Stage 1: Mamba S6 SSM</text>
        <text x="360" y="628" class="svg-label-sub">Linear-Time Global Sequence Mixing</text>
      </g>

      <path d="M 360,580 L 360,515" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Stage 2: Cross-Gated RMSNorm -->
      <g class="svg-node-interactive" onclick="openCircuitModal('selective_ssm', 'Stage 2: Cross-Gated RMSNorm & Residual Connection')">
        <rect x="240" y="470" width="240" height="45" rx="8" fill="rgba(253, 224, 71, 0.22)" stroke="#fde047" stroke-width="1.8"/>
        <text x="360" y="493" class="svg-label-title">Cross-Gated RMSNorm</text>
      </g>

      <path d="M 360,470 L 360,405" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Stage 3: Window Multi-Head Attention -->
      <g class="svg-node-interactive" onclick="openCircuitModal('multihead_attention', 'Stage 3: Windowed Multi-Head Attention')">
        <rect x="220" y="335" width="280" height="70" rx="10" fill="rgba(79, 155, 255, 0.25)" stroke="#4f9bff" stroke-width="2"/>
        <text x="360" y="360" class="svg-label-title" style="fill:#4f9bff;">Stage 3: Multi-Head Attention</text>
        <text x="360" y="383" class="svg-label-sub">High-Precision In-Context Retrieval</text>
      </g>

      <path d="M 360,335 L 360,270" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Stage 4: SwiGLU Gated Feedforward -->
      <g class="svg-node-interactive" onclick="openCircuitModal('multihead_attention', 'Stage 4: SwiGLU Gated Feedforward Network')">
        <rect x="220" y="200" width="280" height="70" rx="10" fill="rgba(192, 132, 252, 0.25)" stroke="#c084fc" stroke-width="2"/>
        <text x="360" y="225" class="svg-label-title">Stage 4: SwiGLU Feedforward</text>
        <text x="360" y="248" class="svg-label-sub">Non-linear Dimension Expansion</text>
      </g>

      <path d="M 360,200 L 360,135" class="svg-flow-path" marker-end="url(#arrow)"/>

      <text x="360" y="100" class="svg-label-title" style="font-size:16px; fill:#c5ff22;">Hybrid Output Sequence Y</text>
    </svg>
    <div class="paper-figure-caption">Figure 1: TransMamba Hybrid SSM-Transformer Architecture (2024)</div>
  `;
}

// 8. P100SUPERCNN (Debi Prasad Panda & Pratik Kiran Rout, GITA 2025/2026 - Poster Architecture)
function renderP100SuperCNNSVG() {
  return `
    <svg class="architecture-svg-root" viewBox="0 0 780 920" fill="none" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#919096" class="svg-arrow-head"/>
        </marker>
        <marker id="arrow-active" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#c5ff22"/>
        </marker>
      </defs>

      <!-- Input RGB Image -->
      <text x="390" y="890" class="svg-label-title" style="font-size:15px; fill:#e5e2e2;">Input Image X ∈ ℝ^{32 × 32 × 3} (CIFAR-100)</text>
      <path d="M 390,875 L 390,845" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Stem Layer -->
      <g class="svg-node-interactive" onclick="openCircuitModal('mbconv_block', 'Lightweight 3x3 Conv Stem (32x32x3 → 32x32x32)')">
        <rect x="250" y="800" width="280" height="45" rx="8" fill="rgba(244, 114, 182, 0.25)" stroke="#f472b6" stroke-width="1.8"/>
        <text x="390" y="822" class="svg-label-title">Stem: 3×3 Conv + BatchNorm + Swish (32ch)</text>
      </g>

      <path d="M 390,800 L 390,765" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- 7-Stage Body Outer Container -->
      <rect x="50" y="340" width="680" height="420" rx="16" fill="rgba(30, 41, 59, 0.25)" stroke="#64748b" stroke-width="1.5" stroke-dasharray="6 6"/>
      <text x="75" y="365" class="svg-label-title" style="font-size:14px; fill:#c5ff22; font-family:'JetBrains Mono';">7-Stage Hierarchical Body (86.6% of 4.03M Parameters)</text>

      <!-- Stage Timeline Pipeline (Horizontal sequence of circles) -->
      <!-- Stage 1 -->
      <g class="svg-node-interactive" onclick="openCircuitModal('mbconv_block', 'Stage 1: First MBConvSD Block (32x32x32)')">
        <circle cx="100" cy="715" r="18" fill="rgba(79, 155, 255, 0.3)" stroke="#4f9bff" stroke-width="2"/>
        <text x="100" y="716" class="svg-label-title" style="font-size:11px;">S1</text>
        <text x="100" y="745" class="svg-label-sub">32×32 (32c)</text>
      </g>
      <path d="M 118,715 L 172,715" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Stage 2 -->
      <g class="svg-node-interactive" onclick="openCircuitModal('mbconv_block', 'Stage 2: Two MBConvSD Blocks (32x32x48)')">
        <circle cx="190" cy="715" r="18" fill="rgba(74, 222, 128, 0.3)" stroke="#4ade80" stroke-width="2"/>
        <text x="190" y="716" class="svg-label-title" style="font-size:11px;">S2</text>
        <text x="190" y="745" class="svg-label-sub">32×32 (48c)</text>
      </g>
      <path d="M 208,715 L 262,715" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Stage 3 (Downsampling 1) -->
      <g class="svg-node-interactive" onclick="openCircuitModal('mbconv_block', 'Stage 3: Downsampling Stage 1 (16x16x80)')">
        <circle cx="280" cy="715" r="18" fill="rgba(251, 146, 60, 0.3)" stroke="#fb923c" stroke-width="2"/>
        <text x="280" y="716" class="svg-label-title" style="font-size:11px;">S3↓</text>
        <text x="280" y="745" class="svg-label-sub">16×16 (80c)</text>
      </g>
      <path d="M 298,715 L 352,715" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Stage 4 (Downsampling 2) -->
      <g class="svg-node-interactive" onclick="openCircuitModal('mbconv_block', 'Stage 4: Downsampling Stage 2 (8x8x112)')">
        <circle cx="370" cy="715" r="18" fill="rgba(253, 224, 71, 0.3)" stroke="#fde047" stroke-width="2"/>
        <text x="370" y="716" class="svg-label-title" style="font-size:11px;">S4↓</text>
        <text x="370" y="745" class="svg-label-sub">8×8 (112c)</text>
      </g>
      <path d="M 388,715 L 442,715" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Stage 5 -->
      <g class="svg-node-interactive" onclick="openCircuitModal('mbconv_block', 'Stage 5: Three MBConvSD Blocks (8x8x192)')">
        <circle cx="460" cy="715" r="18" fill="rgba(192, 132, 252, 0.3)" stroke="#c084fc" stroke-width="2"/>
        <text x="460" y="716" class="svg-label-title" style="font-size:11px;">S5</text>
        <text x="460" y="745" class="svg-label-sub">8×8 (192c)</text>
      </g>
      <path d="M 478,715 L 532,715" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Stage 6 (Downsampling 3) -->
      <g class="svg-node-interactive" onclick="openCircuitModal('mbconv_block', 'Stage 6: Final Downsampling Stage 3 (4x4x320)')">
        <circle cx="550" cy="715" r="18" fill="rgba(244, 114, 182, 0.3)" stroke="#f472b6" stroke-width="2"/>
        <text x="550" y="716" class="svg-label-title" style="font-size:11px;">S6↓</text>
        <text x="550" y="745" class="svg-label-sub">4×4 (320c)</text>
      </g>
      <path d="M 568,715 L 622,715" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Stage 7 -->
      <g class="svg-node-interactive" onclick="openCircuitModal('mbconv_block', 'Stage 7: Final MBConvSD Block (4x4x320)')">
        <circle cx="640" cy="715" r="18" fill="rgba(74, 222, 128, 0.3)" stroke="#4ade80" stroke-width="2"/>
        <text x="640" y="716" class="svg-label-title" style="font-size:11px;">S7</text>
        <text x="640" y="745" class="svg-label-sub">4×4 (320c)</text>
      </g>

      <!-- INSET: Inside the MBConvSD Core Block -->
      <rect x="75" y="385" width="630" height="280" rx="12" fill="rgba(15, 23, 42, 0.85)" stroke="#38bdf8" stroke-width="1.8"/>
      <text x="95" y="408" class="svg-label-title" style="font-size:13.5px; fill:#38bdf8;">Core Component: Inside The MBConvSD Block</text>

      <!-- Residual Bypass in MBConvSD -->
      <path d="M 120,530 L 120,440 L 660,440 L 660,530" class="svg-flow-path" stroke="#64748b" stroke-width="2" stroke-dasharray="4 4"/>
      <text x="390" y="430" class="svg-label-sub" style="fill:#64748b;">Inverted Residual Skip Connection (Identity)</text>

      <!-- 1. 1x1 Expansion -->
      <g class="svg-node-interactive" onclick="openCircuitModal('mbconv_block', '1x1 Pointwise Expansion Conv (t=4 to 6)')">
        <rect x="100" y="490" width="105" height="85" rx="8" fill="rgba(192, 132, 252, 0.25)" stroke="#c084fc" stroke-width="1.8"/>
        <text x="152" y="520" class="svg-label-title" style="font-size:12px;">1×1 Conv</text>
        <text x="152" y="540" class="svg-label-sub">Expansion</text>
        <text x="152" y="558" class="svg-label-sub" style="fill:#c084fc;">t=4..6</text>
      </g>
      <path d="M 205,532 L 225,532" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- 2. 3x3 Depthwise Conv -->
      <g class="svg-node-interactive" onclick="openCircuitModal('mbconv_block', '3x3 Depthwise Separable Convolution')">
        <rect x="225" y="490" width="115" height="85" rx="8" fill="rgba(79, 155, 255, 0.25)" stroke="#4f9bff" stroke-width="1.8"/>
        <text x="282" y="520" class="svg-label-title" style="font-size:12px;">3×3 Depthwise</text>
        <text x="282" y="540" class="svg-label-sub">Separable</text>
        <text x="282" y="558" class="svg-label-sub" style="fill:#4f9bff;">8x FLOPs ↓</text>
      </g>
      <path d="M 340,532 L 360,532" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- 3. Squeeze-and-Excitation (SE) Attention -->
      <g class="svg-node-interactive" onclick="openCircuitModal('se_attention', 'Squeeze-and-Excitation (SE) Channel-Wise Attention')">
        <rect x="360" y="480" width="140" height="105" rx="10" fill="rgba(251, 146, 60, 0.3)" stroke="#fb923c" stroke-width="2"/>
        <text x="430" y="505" class="svg-label-title" style="font-size:13px; fill:#fb923c;">SE Attention</text>
        <text x="430" y="528" class="svg-label-sub">z = GAP(U)</text>
        <text x="430" y="548" class="svg-label-sub">s = σ(W2·Swish(W1·z))</text>
        <text x="430" y="568" class="svg-label-sub" style="fill:#fde047;">Recalibrate</text>
      </g>
      <path d="M 500,532 L 520,532" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- 4. 1x1 Linear Projection -->
      <g class="svg-node-interactive" onclick="openCircuitModal('mbconv_block', '1x1 Linear Bottleneck Projection (No Activation)')">
        <rect x="520" y="490" width="105" height="85" rx="8" fill="rgba(192, 132, 252, 0.25)" stroke="#c084fc" stroke-width="1.8"/>
        <text x="572" y="520" class="svg-label-title" style="font-size:12px;">1×1 Conv</text>
        <text x="572" y="540" class="svg-label-sub">Projection</text>
        <text x="572" y="558" class="svg-label-sub" style="fill:#4ade80;">Linear</text>
      </g>
      <path d="M 625,532 L 646,532" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- DropPath & Residual Add ⊕ -->
      <g class="svg-node-interactive" onclick="openCircuitModal('mbconv_block', 'DropPath (Stochastic Depth) & Inverted Residual Add')">
        <circle cx="660" cy="532" r="16" fill="#201f20" stroke="#c5ff22" stroke-width="2"/>
        <path d="M 653,532 L 667,532 M 660,525 L 660,539" stroke="#c5ff22" stroke-width="2"/>
        <text x="660" y="565" class="svg-label-sub" style="fill:#c5ff22;">DropPath</text>
      </g>

      <!-- Arrow from Body to TheHead -->
      <path d="M 390,340 L 390,295" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- THE HEAD (Final Classification 3D Funnel) -->
      <rect x="120" y="90" width="540" height="205" rx="14" fill="rgba(15, 23, 42, 0.9)" stroke="#4ade80" stroke-width="2"/>
      <text x="145" y="115" class="svg-label-title" style="font-size:14px; fill:#4ade80;">TheHead (Final Classification Subsystem)</text>

      <!-- Layer 18: 1x1 Conv Expansion -->
      <g class="svg-node-interactive" onclick="openCircuitModal('the_head_funnel', 'Layer 18: 1x1 Conv Expansion (320 → 1280 channels)')">
        <rect x="145" y="140" width="115" height="75" rx="8" fill="rgba(192, 132, 252, 0.25)" stroke="#c084fc" stroke-width="1.8"/>
        <text x="202" y="165" class="svg-label-title" style="font-size:12px;">Layer 18</text>
        <text x="202" y="183" class="svg-label-sub">1×1 Conv</text>
        <text x="202" y="201" class="svg-label-sub" style="fill:#c084fc;">320→1280ch</text>
      </g>
      <path d="M 260,177 L 275,177" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Layer 19: Global Average Pooling (GAP) -->
      <g class="svg-node-interactive" onclick="openCircuitModal('the_head_funnel', 'Layer 19: Global Average Pooling (GAP 4x4 → 1x1)')">
        <rect x="275" y="140" width="110" height="75" rx="8" fill="rgba(74, 222, 128, 0.25)" stroke="#4ade80" stroke-width="1.8"/>
        <text x="330" y="165" class="svg-label-title" style="font-size:12px;">Layer 19</text>
        <text x="330" y="183" class="svg-label-sub">GAP (AvgPool)</text>
        <text x="330" y="201" class="svg-label-sub" style="fill:#4ade80;">0 Params</text>
      </g>
      <path d="M 385,177 L 400,177" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Layer 20: Dropout -->
      <g class="svg-node-interactive" onclick="openCircuitModal('the_head_funnel', 'Layer 20: Dropout Regularization (p=0.2)')">
        <rect x="400" y="140" width="105" height="75" rx="8" fill="rgba(253, 224, 71, 0.22)" stroke="#fde047" stroke-width="1.8"/>
        <text x="452" y="165" class="svg-label-title" style="font-size:12px;">Layer 20</text>
        <text x="452" y="183" class="svg-label-sub">Dropout</text>
        <text x="452" y="201" class="svg-label-sub" style="fill:#fde047;">p = 0.2</text>
      </g>
      <path d="M 505,177 L 520,177" class="svg-flow-path" marker-end="url(#arrow)"/>

      <!-- Layer 21: Linear Classifier -->
      <g class="svg-node-interactive" onclick="openCircuitModal('the_head_funnel', 'Layer 21: Linear Classifier (1280 → 100 CIFAR-100 Classes)')">
        <rect x="520" y="140" width="125" height="75" rx="8" fill="rgba(79, 155, 255, 0.25)" stroke="#4f9bff" stroke-width="1.8"/>
        <text x="582" y="165" class="svg-label-title" style="font-size:12px;">Layer 21</text>
        <text x="582" y="183" class="svg-label-sub">Linear Layer</text>
        <text x="582" y="201" class="svg-label-sub" style="fill:#4f9bff;">100 Classes</text>
      </g>

      <!-- Output Class Logits -->
      <path d="M 390,90 L 390,45" class="svg-flow-path" marker-end="url(#arrow)"/>
      <text x="390" y="25" class="svg-label-title" style="font-size:16px; fill:#c5ff22;">Output: 100 CIFAR-100 Class Logits (4.03M Params, 15.4MB)</text>
    </svg>
    <div class="paper-figure-caption">Figure 1: P100SuperCNN — Bespoke Parameter-Efficient CNN for Resource-Constrained Environments (Debi Prasad Panda & Pratik Kiran Rout, GITA 2025/2026)</div>
  `;
}

// Simulate Forward Token Pulse Animation through SVG Nodes & Paths
function runTokenFlowSimulation() {
  if (IS_SIMULATING_FLOW) return;
  IS_SIMULATING_FLOW = true;

  const nodes = document.querySelectorAll('.svg-node-interactive');
  const paths = document.querySelectorAll('.svg-flow-path');
  const arrowHeads = document.querySelectorAll('.svg-arrow-head');

  if (!nodes.length) {
    IS_SIMULATING_FLOW = false;
    return;
  }

  let index = 0;
  const interval = setInterval(() => {
    nodes.forEach(n => n.classList.remove('sim-active'));
    paths.forEach(p => p.classList.remove('sim-active'));
    arrowHeads.forEach(a => a.classList.remove('sim-active'));

    if (index < nodes.length) {
      nodes[index].classList.add('sim-active');
      if (paths[index]) paths[index].classList.add('sim-active');
      if (arrowHeads[index]) arrowHeads[index].classList.add('sim-active');
      index++;
    } else {
      clearInterval(interval);
      setTimeout(() => {
        nodes.forEach(n => n.classList.remove('sim-active'));
        paths.forEach(p => p.classList.remove('sim-active'));
        arrowHeads.forEach(a => a.classList.remove('sim-active'));
        IS_SIMULATING_FLOW = false;
      }, 600);
    }
  }, 350);
}

// Toggle Macro-Layer Cards Expand / Collapse All
function toggleAllMacroCards(forceState) {
  const cards = document.querySelectorAll('.layer-card');
  if (!cards.length) return;

  const anyCollapsed = Array.from(cards).some(c => c.classList.contains('collapsed'));
  const shouldExpand = (typeof forceState === 'boolean') ? forceState : anyCollapsed;

  cards.forEach(card => {
    card.classList.toggle('collapsed', !shouldExpand);
  });

  // Update button texts and icons
  document.querySelectorAll('.expand-all-btn').forEach(btn => {
    const span = btn.querySelector('span:last-child') || btn;
    if (span) span.textContent = shouldExpand ? 'Collapse All Layers' : 'Expand All Layers';
    const icon = btn.querySelector('.material-symbols-outlined');
    if (icon) icon.textContent = shouldExpand ? 'unfold_less' : 'unfold_more';
  });
}

// Handle Canvas Expand All: switch to Micro-Layers tab and expand all
function handleCanvasExpandAll() {
  const layerTab = document.querySelector('.view-tab[data-subview="layer-breakdown"]');
  if (layerTab) {
    layerTab.click();
  }
  setTimeout(() => {
    toggleAllMacroCards(true);
    const section = document.getElementById('microLayersSection');
    if (section) {
      section.scrollIntoView({ behavior: 'smooth' });
    }
  }, 50);
}

window.toggleAllMacroCards = toggleAllMacroCards;
window.handleCanvasExpandAll = handleCanvasExpandAll;

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
      <div class="pro-con-detail">${p.detail || p.mechanistic_reasoning || ''}</div>
      <span class="citation-chip">${p.section || p.source_section || 'Primary Paper'}</span>
    </div>
  `).join('');

  consList.innerHTML = (paper.cons || []).map(c => `
    <div class="pro-con-item con-item">
      <div class="pro-con-claim">${c.claim}</div>
      <div class="pro-con-detail">${c.detail || c.mechanistic_reasoning || ''}</div>
      <span class="citation-chip">${c.section || c.source_section || 'Primary Paper'}</span>
    </div>
  `).join('');
}

// Render Reported Empirical Benchmarks View
function renderReportedBenchmarks(paper) {
  const tbody = document.getElementById('reportedMetricsTableBody');
  if (!tbody || !paper) return;

  const benchmarks = paper.reported_benchmarks || paper.reported_metrics || [];
  tbody.innerHTML = benchmarks.map(m => `
    <tr>
      <td><strong>${m.task || ''}</strong></td>
      <td>${m.metric_name || m.metric || ''}</td>
      <td><span class="badge-success">${m.score_value || m.value || ''}</span></td>
      <td><span class="citation-chip">${m.source_section || m.source_citation || 'Primary Paper'}</span></td>
      <td>${m.comparison_context || m.context || ''}</td>
    </tr>
  `).join('');
}

// Render Poster Presentation & Deep Vulnerability Analytics
function renderPosterAnalytics(paper) {
  const container = document.getElementById('posterAnalyticsContainer');
  if (!container || !paper) return;

  const isP100 = paper.paper_id === 'p100supercnn';

  if (isP100) {
    container.innerHTML = `
      <div class="poster-interactive-dashboard">
        <!-- Poster Academic Header -->
        <div class="poster-academic-header glass-panel">
          <div class="poster-university-badge">
            <span class="material-symbols-outlined text-[18px]">school</span>
            <span>Gandhi Institute for Technological Advancement (GITA), Bhubaneswar, Odisha</span>
          </div>
          <h2 class="poster-paper-title">An Efficient Deep Learning Model for Image Classification on Resource Constrained Environment</h2>
          <p class="poster-authors">Debi Prasad Panda and Pratik Kiran Rout</p>
          <p class="poster-dept">Department of Computer Science and Information Technology</p>
          <div class="poster-email-chips">
            <span class="email-chip">✉ debiprasadpanda73@gmail.com</span>
            <span class="email-chip">✉ pratikkiranrout@gmail.com</span>
            <span class="citation-chip">CIFAR-100 Benchmark Target (100 Classes, 32×32)</span>
          </div>
        </div>

        <!-- 3 Key Metric Cards -->
        <div class="poster-metrics-summary-grid">
          <div class="metric-card glass-panel">
            <div class="metric-card-top">
              <span class="metric-label">Efficiency Score</span>
              <span class="metric-trend text-primary">+26% vs EfficientNet-B0</span>
            </div>
            <div class="metric-main-val" style="color:#c5ff22;">18.32</div>
            <p class="metric-subtext">Accuracy / MParams ratio (Pareto sweet spot)</p>
          </div>

          <div class="metric-card glass-panel">
            <div class="metric-card-top">
              <span class="metric-label">Deployment Footprint</span>
              <span class="metric-trend text-primary">-66% vs ResNet-18</span>
            </div>
            <div class="metric-main-val" style="color:#38bdf8;">15.4 MB</div>
            <p class="metric-subtext">4.03M total weights (Ideal for edge microcontrollers)</p>
          </div>

          <div class="metric-card glass-panel">
            <div class="metric-card-top">
              <span class="metric-label">Augmentation Gain</span>
              <span class="metric-trend text-primary">+2.91% Accuracy</span>
            </div>
            <div class="metric-main-val" style="color:#4ade80;">73.71%</div>
            <p class="metric-subtext">3-Stage Curriculum policy over 160 epochs</p>
          </div>
        </div>

        <!-- Comparative Charts Grid -->
        <div class="poster-charts-grid">
          <!-- Chart 1: Parameter Efficiency Score -->
          <div class="poster-chart-card glass-panel">
            <div class="chart-header">
              <h4 class="chart-title">Parameter Efficiency Score (Accuracy / MParams)</h4>
              <span class="chart-badge">Higher is Better</span>
            </div>
            <div class="chart-bars-list">
              <div class="chart-bar-row">
                <div class="chart-bar-meta">
                  <span class="chart-bar-model-name highlight-model">P100SuperCNN (Proposed)</span>
                  <span class="chart-bar-val" style="color:#c5ff22;">18.32</span>
                </div>
                <div class="chart-bar-track">
                  <div class="chart-bar-fill highlight-fill" style="width: 89%;"></div>
                </div>
              </div>
              <div class="chart-bar-row">
                <div class="chart-bar-meta">
                  <span class="chart-bar-model-name">P100 (Complex)</span>
                  <span class="chart-bar-val">18.10</span>
                </div>
                <div class="chart-bar-track">
                  <div class="chart-bar-fill" style="width: 88%;"></div>
                </div>
              </div>
              <div class="chart-bar-row">
                <div class="chart-bar-meta">
                  <span class="chart-bar-model-name">MobileNetV2</span>
                  <span class="chart-bar-val">20.50</span>
                </div>
                <div class="chart-bar-track">
                  <div class="chart-bar-fill" style="width: 100%;"></div>
                </div>
              </div>
              <div class="chart-bar-row">
                <div class="chart-bar-meta">
                  <span class="chart-bar-model-name">EfficientNet-B0</span>
                  <span class="chart-bar-val">14.50</span>
                </div>
                <div class="chart-bar-track">
                  <div class="chart-bar-fill" style="width: 71%;"></div>
                </div>
              </div>
              <div class="chart-bar-row">
                <div class="chart-bar-meta">
                  <span class="chart-bar-model-name">MobileNetV3-Large</span>
                  <span class="chart-bar-val">13.90</span>
                </div>
                <div class="chart-bar-track">
                  <div class="chart-bar-fill" style="width: 68%;"></div>
                </div>
              </div>
              <div class="chart-bar-row">
                <div class="chart-bar-meta">
                  <span class="chart-bar-model-name">ResNet-18</span>
                  <span class="chart-bar-val">5.80</span>
                </div>
                <div class="chart-bar-track">
                  <div class="chart-bar-fill warn-fill" style="width: 28%;"></div>
                </div>
              </div>
              <div class="chart-bar-row">
                <div class="chart-bar-meta">
                  <span class="chart-bar-model-name">ResNet-50</span>
                  <span class="chart-bar-val">2.80</span>
                </div>
                <div class="chart-bar-track">
                  <div class="chart-bar-fill warn-fill" style="width: 14%;"></div>
                </div>
              </div>
            </div>
          </div>

          <!-- Chart 2: Model Storage Size (MB) -->
          <div class="poster-chart-card glass-panel">
            <div class="chart-header">
              <h4 class="chart-title">Model Storage Size on Disk (MB)</h4>
              <span class="chart-badge">Lower is Better</span>
            </div>
            <div class="chart-bars-list">
              <div class="chart-bar-row">
                <div class="chart-bar-meta">
                  <span class="chart-bar-model-name highlight-model">P100SuperCNN</span>
                  <span class="chart-bar-val" style="color:#c5ff22;">15.4 MB</span>
                </div>
                <div class="chart-bar-track">
                  <div class="chart-bar-fill highlight-fill" style="width: 34%;"></div>
                </div>
              </div>
              <div class="chart-bar-row">
                <div class="chart-bar-meta">
                  <span class="chart-bar-model-name">EfficientNet-B0</span>
                  <span class="chart-bar-val">20.2 MB</span>
                </div>
                <div class="chart-bar-track">
                  <div class="chart-bar-fill" style="width: 45%;"></div>
                </div>
              </div>
              <div class="chart-bar-row">
                <div class="chart-bar-meta">
                  <span class="chart-bar-model-name">MobileNetV3</span>
                  <span class="chart-bar-val">20.6 MB</span>
                </div>
                <div class="chart-bar-track">
                  <div class="chart-bar-fill" style="width: 46%;"></div>
                </div>
              </div>
              <div class="chart-bar-row">
                <div class="chart-bar-meta">
                  <span class="chart-bar-model-name">ResNet-18</span>
                  <span class="chart-bar-val">44.7 MB</span>
                </div>
                <div class="chart-bar-track">
                  <div class="chart-bar-fill warn-fill" style="width: 100%;"></div>
                </div>
              </div>
            </div>
          </div>

          <!-- Chart 3: Model Parameters (Millions) -->
          <div class="poster-chart-card glass-panel">
            <div class="chart-header">
              <h4 class="chart-title">Total Model Parameters (Millions)</h4>
              <span class="chart-badge">Lower is Better</span>
            </div>
            <div class="chart-bars-list">
              <div class="chart-bar-row">
                <div class="chart-bar-meta">
                  <span class="chart-bar-model-name highlight-model">P100SuperCNN</span>
                  <span class="chart-bar-val" style="color:#c5ff22;">4.03 M</span>
                </div>
                <div class="chart-bar-track">
                  <div class="chart-bar-fill highlight-fill" style="width: 34.4%;"></div>
                </div>
              </div>
              <div class="chart-bar-row">
                <div class="chart-bar-meta">
                  <span class="chart-bar-model-name">MobileNetV3</span>
                  <span class="chart-bar-val">5.40 M</span>
                </div>
                <div class="chart-bar-track">
                  <div class="chart-bar-fill" style="width: 46.1%;"></div>
                </div>
              </div>
              <div class="chart-bar-row">
                <div class="chart-bar-meta">
                  <span class="chart-bar-model-name">EfficientNet-B0</span>
                  <span class="chart-bar-val">5.60 M</span>
                </div>
                <div class="chart-bar-track">
                  <div class="chart-bar-fill" style="width: 47.8%;"></div>
                </div>
              </div>
              <div class="chart-bar-row">
                <div class="chart-bar-meta">
                  <span class="chart-bar-model-name">ResNet-18</span>
                  <span class="chart-bar-val">11.70 M</span>
                </div>
                <div class="chart-bar-track">
                  <div class="chart-bar-fill warn-fill" style="width: 100%;"></div>
                </div>
              </div>
            </div>
          </div>

          <!-- Chart 4: Progressive Augmentation Convergence -->
          <div class="poster-chart-card glass-panel">
            <div class="chart-header">
              <h4 class="chart-title">Progressive Augmentation Convergence Curve</h4>
              <span class="chart-badge">+2.91% Validation Gain</span>
            </div>
            <div style="width: 100%; height: 160px; display: flex; align-items: center; justify-content: center;">
              <svg viewBox="0 0 360 140" style="width:100%; height:100%;">
                <!-- Axes -->
                <line x1="30" y1="120" x2="340" y2="120" stroke="#64748b" stroke-width="1.5"/>
                <line x1="30" y1="120" x2="30" y2="20" stroke="#64748b" stroke-width="1.5"/>
                <text x="340" y="135" font-size="9" fill="#94a3b8" text-anchor="end">Epoch (0..160)</text>
                <text x="25" y="20" font-size="9" fill="#94a3b8" text-anchor="end">Acc %</text>
                
                <!-- Stage dividing dotted lines -->
                <line x1="120" y1="20" x2="120" y2="120" stroke="#475569" stroke-width="1" stroke-dasharray="3 3"/>
                <line x1="220" y1="20" x2="220" y2="120" stroke="#475569" stroke-width="1" stroke-dasharray="3 3"/>
                <text x="75" y="32" font-size="8" fill="#94a3b8" text-anchor="middle">Stage 1</text>
                <text x="170" y="32" font-size="8" fill="#94a3b8" text-anchor="middle">Stage 2</text>
                <text x="280" y="32" font-size="8" fill="#94a3b8" text-anchor="middle">Stage 3</text>

                <!-- Training Curve (Blue) -->
                <path d="M 30,115 Q 80,60 140,40 T 260,28 T 340,24" fill="none" stroke="#38bdf8" stroke-width="2.5"/>
                <!-- Validation Curve (Lime) -->
                <path d="M 30,118 Q 80,72 140,55 T 260,42 T 340,36" fill="none" stroke="#c5ff22" stroke-width="2.5"/>

                <!-- Legend -->
                <circle cx="210" cy="112" r="3.5" fill="#38bdf8"/>
                <text x="218" y="115" font-size="8" fill="#e2e8f0">Train (92.4%)</text>
                <circle cx="280" cy="112" r="3.5" fill="#c5ff22"/>
                <text x="288" y="115" font-size="8" fill="#e2e8f0">Val (73.71%)</text>
              </svg>
            </div>
          </div>
        </div>

        <!-- Deep Vulnerability & Engineering Failure Modes Section -->
        <div class="poster-vulnerability-section glass-panel">
          <div class="section-header">
            <div>
              <span class="level-pill" style="background: rgba(239, 68, 68, 0.2); color: #f87171;">Failure Mode & Robustness Audit</span>
              <h3 class="section-title">Deep Architectural Vulnerability & Mitigation Breakdown</h3>
            </div>
            <p class="section-desc">Critical analysis of the most sensitive mechanisms, numerical failure modes, and hardware bottlenecks.</p>
          </div>

          <div class="vulnerability-cards-grid">
            <!-- Vulnerability 1 -->
            <div class="vulnerability-card">
              <div class="vuln-header">
                <span class="vuln-severity severe">High Sensitivity</span>
                <h4>1. Gradient Vanishing in Inverted Bottlenecks</h4>
              </div>
              <p class="vuln-desc">Expanding channels by $t=4..6$ and projecting back to low dimension can cause rank loss and gradient dissipation across 7 stages.</p>
              <div class="vuln-solution">
                <strong>Engineering Mitigation:</strong> Removing non-linear activation after the $1\\times1$ linear projection layer preserves representation manifolds; inverted residual identity shortcuts maintain unobstructed backpropagation highways.
              </div>
            </div>

            <!-- Vulnerability 2 -->
            <div class="vulnerability-card">
              <div class="vuln-header">
                <span class="vuln-severity moderate">Moderate Sensitivity</span>
                <h4>2. Channel Compression Bottleneck in SE Attention</h4>
              </div>
              <p class="vuln-desc">Low reduction ratio $r \\ge 16$ causes aggressive loss of subtle inter-channel dependencies, while $r \\le 2$ blows up parameter budgets.</p>
              <div class="vuln-solution">
                <strong>Engineering Mitigation:</strong> Rigorously calibrated to $r=4$ with Swish non-linearity, capturing high-frequency inter-channel correlations at $<1.5\\%$ parameter overhead.
              </div>
            </div>

            <!-- Vulnerability 3 -->
            <div class="vulnerability-card">
              <div class="vuln-header">
                <span class="vuln-severity moderate">Dataset Constraint</span>
                <h4>3. Overfitting on Fine-Grained 32x32 CIFAR-100</h4>
              </div>
              <p class="vuln-desc">With 100 fine-grained classes and only 500 training images per class, high-capacity CNNs easily overfit to background noise.</p>
              <div class="vuln-solution">
                <strong>Engineering Mitigation:</strong> 3-Stage Curriculum Augmentation (Random Crop $\\rightarrow$ RandAugment $\\rightarrow$ CutMix/MixUp) combined with linear DropPath ($p=0.2$) yields $+2.91\\%$ validation accuracy boost.
              </div>
            </div>
          </div>
        </div>
      </div>
    `;
  } else {
    // Dynamic fallback poster breakdown for the other 7 papers
    container.innerHTML = `
      <div class="poster-interactive-dashboard">
        <div class="poster-academic-header glass-panel">
          <div class="poster-university-badge">
            <span class="material-symbols-outlined text-[18px]">menu_book</span>
            <span>Primary Research Publication</span>
          </div>
          <h2 class="poster-paper-title">${paper.title}</h2>
          <p class="poster-authors">${paper.authors_venue_year}</p>
          <div class="poster-email-chips">
            <span class="citation-chip">ID: ${paper.paper_id}</span>
            <span class="badge-success">${paper.layers.length} Macro Blocks</span>
            <span class="citation-chip">${paper.layers[0]?.flops_weight || 'O(L · D)'}</span>
          </div>
        </div>

        <div class="poster-metrics-summary-grid">
          <div class="metric-card glass-panel">
            <div class="metric-card-top">
              <span class="metric-label">Macro Blocks</span>
              <span class="metric-trend text-primary">Hierarchy</span>
            </div>
            <div class="metric-main-val" style="color:#c5ff22;">${paper.layers.length}</div>
            <p class="metric-subtext">Total stacked stages in core backbone</p>
          </div>

          <div class="metric-card glass-panel">
            <div class="metric-card-top">
              <span class="metric-label">Sublayer Mechanisms</span>
              <span class="metric-trend text-primary">Circuits</span>
            </div>
            <div class="metric-main-val" style="color:#38bdf8;">${paper.layers.reduce((acc, l) => acc + (l.sublayers?.length || 0), 0)}</div>
            <p class="metric-subtext">Detailed mathematical sublayer transforms</p>
          </div>

          <div class="metric-card glass-panel">
            <div class="metric-card-top">
              <span class="metric-label">Asymptotic Complexity</span>
              <span class="metric-trend text-primary">FLOP Scaling</span>
            </div>
            <div class="metric-main-val" style="color:#4ade80; font-size:1.3rem;">${paper.layers[0]?.flops_weight || 'O(L · D)'}</div>
            <p class="metric-subtext">Sequence vs hidden dimension compute profile</p>
          </div>
        </div>

        <div class="poster-vulnerability-section glass-panel">
          <div class="section-header">
            <div>
              <span class="level-pill" style="background: rgba(239, 68, 68, 0.2); color: #f87171;">Failure Mode & Robustness Audit</span>
              <h3 class="section-title">Architectural Vulnerabilities & Failure Modes</h3>
            </div>
            <p class="section-desc">Key numerical and memory failure modes reported across sublayers.</p>
          </div>

          <div class="vulnerability-cards-grid">
            ${paper.layers.flatMap(l => l.sublayers || []).slice(0, 3).map((sub, i) => `
              <div class="vulnerability-card">
                <div class="vuln-header">
                  <span class="vuln-severity ${i === 0 ? 'severe' : 'moderate'}">${i === 0 ? 'High Sensitivity' : 'Mechanism Constraint'}</span>
                  <h4>${sub.name}</h4>
                </div>
                <p class="vuln-desc">${sub.failure_modes || 'Numerical instability under extreme inputs.'}</p>
                <div class="vuln-solution">
                  <strong>Hardware & Mitigation Note:</strong> ${sub.hardware_notes || 'Standard gradient clipping and fp16 scaling.'}
                </div>
              </div>
            `).join('')}
          </div>
        </div>
      </div>
    `;
  }
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
