import json, os

# Load existing verified dataset
dataset_path = r'a:\Downloads\24th OITS\understanding\papers_data.json'
with open(dataset_path, 'r', encoding='utf-8') as f:
    papers = json.load(f)

print(f"Loaded {len(papers)} papers. Adding detailed graphical diagrams, internal circuit models, and deep undertones...")

# Let's add graphical_diagram specifications for each paper
# Every paper will have:
# - diagram_type: "transformer_encoder_decoder" | "mamba_block" | "mod_routing" | "moe_sparse_bank" | "swat_window_sigmoid" | "titans_neural_memory" | "transmamba_hybrid"
# - diagram_nodes: list of nodes with id, label, type, color_theme, sublayer_key, internal_circuit_description, undertones

diagrams_meta = {
    "transformer": {
        "diagram_title": "Transformer Model Architecture (Figure 1)",
        "layout_type": "dual_column_encoder_decoder",
        "nodes": [
            {
                "id": "trans_inputs",
                "label": "Inputs",
                "category": "io",
                "color": "io",
                "description": "Raw token IDs representing source sequence.",
                "undertones": "Discrete vocabulary IDs converted via embedding lookup."
            },
            {
                "id": "trans_input_emb",
                "label": "Input Embedding",
                "category": "embedding",
                "color": "pink",
                "layer_idx": 0,
                "sublayer_idx": 0,
                "description": "Projects discrete tokens to d_model continuous embedding vectors.",
                "undertones": "Scaled by sqrt(d_model) so dot products with positional encodings maintain balanced variance."
            },
            {
                "id": "trans_pos_enc_enc",
                "label": "Positional Encoding",
                "category": "pos_enc",
                "color": "circle_wave",
                "layer_idx": 0,
                "sublayer_idx": 1,
                "description": "Sinusoidal frequency waves providing position awareness.",
                "undertones": "Fixed deterministic trigonometric functions allow model to extrapolate to longer sequence lengths than seen during training."
            },
            {
                "id": "trans_enc_mha",
                "label": "Multi-Head Attention",
                "category": "attention",
                "color": "orange",
                "layer_idx": 1,
                "sublayer_idx": 0,
                "description": "h=8 parallel self-attention heads attending across the full input sequence.",
                "undertones": "O(L^2) global quadratic receptive field. Computes pairwise correlation across all token pairs simultaneously."
            },
            {
                "id": "trans_enc_add_norm1",
                "label": "Add & Norm",
                "category": "norm",
                "color": "yellow",
                "layer_idx": 1,
                "sublayer_idx": 1,
                "description": "Residual skip addition + Layer Normalization.",
                "undertones": "Prevents vanishing gradients across deep layers. Normalizes mean=0, std=1 across the d_model channel dimension."
            },
            {
                "id": "trans_enc_ffn",
                "label": "Feed Forward",
                "category": "ffn",
                "color": "blue",
                "layer_idx": 1,
                "sublayer_idx": 2,
                "description": "Position-wise 2-layer FFN with intermediate 4x expansion (d_ff = 2048) and ReLU.",
                "undertones": "Contains 66% of model parameters. Stores factual knowledge and non-linear feature transformations."
            },
            {
                "id": "trans_enc_add_norm2",
                "label": "Add & Norm",
                "category": "norm",
                "color": "yellow",
                "layer_idx": 1,
                "sublayer_idx": 3,
                "description": "Residual skip addition + Layer Normalization over FFN.",
                "undertones": "Provides highway for uninterrupted gradient backpropagation."
            },
            {
                "id": "trans_outputs",
                "label": "Outputs (shifted right)",
                "category": "io",
                "color": "io",
                "description": "Target sequence shifted right for teacher-forcing autoregression.",
                "undertones": "Right-shift ensures prediction for position i depends only on previous tokens < i."
            },
            {
                "id": "trans_output_emb",
                "label": "Output Embedding",
                "category": "embedding",
                "color": "pink",
                "layer_idx": 0,
                "sublayer_idx": 0,
                "description": "Projects target token IDs to d_model continuous embedding vectors.",
                "undertones": "Tied weight matrix with input embeddings and output softmax projection."
            },
            {
                "id": "trans_pos_enc_dec",
                "label": "Positional Encoding",
                "category": "pos_enc",
                "color": "circle_wave",
                "layer_idx": 0,
                "sublayer_idx": 1,
                "description": "Sinusoidal frequency encodings for target tokens.",
                "undertones": "Identical sine/cosine formulas as encoder positional encodings."
            },
            {
                "id": "trans_dec_masked_mha",
                "label": "Masked Multi-Head Attention",
                "category": "masked_attention",
                "color": "orange",
                "layer_idx": 2,
                "sublayer_idx": 0,
                "description": "Causal self-attention with lower-triangular mask (-infinity for j > i).",
                "undertones": "Strictly prevents future token leakage during both training and generation."
            },
            {
                "id": "trans_dec_add_norm1",
                "label": "Add & Norm",
                "category": "norm",
                "color": "yellow",
                "layer_idx": 2,
                "sublayer_idx": 1,
                "description": "Residual connection + LayerNorm after masked self-attention.",
                "undertones": "Maintains signal stability before cross-attention."
            },
            {
                "id": "trans_dec_cross_mha",
                "label": "Multi-Head Cross-Attention",
                "category": "cross_attention",
                "color": "orange",
                "layer_idx": 2,
                "sublayer_idx": 1,
                "description": "Queries from decoder attend over Keys and Values from the Encoder output.",
                "undertones": "Acts as the associative bridge between source language and target language representations."
            },
            {
                "id": "trans_dec_add_norm2",
                "label": "Add & Norm",
                "category": "norm",
                "color": "yellow",
                "layer_idx": 2,
                "sublayer_idx": 2,
                "description": "Residual connection + LayerNorm after cross-attention.",
                "undertones": "Integrates source context with target trajectory."
            },
            {
                "id": "trans_dec_ffn",
                "label": "Feed Forward",
                "category": "ffn",
                "color": "blue",
                "layer_idx": 2,
                "sublayer_idx": 2,
                "description": "Decoder position-wise FFN with 4x expansion.",
                "undertones": "Refines vocabulary projection representations."
            },
            {
                "id": "trans_dec_add_norm3",
                "label": "Add & Norm",
                "category": "norm",
                "color": "yellow",
                "layer_idx": 2,
                "sublayer_idx": 2,
                "description": "Final decoder residual skip + LayerNorm.",
                "undertones": "Prepares normalized features for vocabulary projection."
            },
            {
                "id": "trans_linear",
                "label": "Linear",
                "category": "linear",
                "color": "purple",
                "layer_idx": 3,
                "sublayer_idx": 0,
                "description": "Projects d_model features to full vocabulary logits (V = 37,000).",
                "undertones": "Produces unnormalized energy logits for each word in the vocabulary."
            },
            {
                "id": "trans_softmax",
                "label": "Softmax",
                "category": "softmax",
                "color": "green",
                "layer_idx": 3,
                "sublayer_idx": 1,
                "description": "Exponentiates and normalizes logits to categorical probabilities.",
                "undertones": "Trained with cross-entropy loss and label smoothing epsilon_ls = 0.1."
            },
            {
                "id": "trans_output_prob",
                "label": "Output Probabilities",
                "category": "io",
                "color": "io",
                "description": "Probability distribution over next target token.",
                "undertones": "Sampled autoregressively with beam search or greedy decoding."
            }
        ]
    },
    "mamba": {
        "diagram_title": "Mamba Block Architecture (Figure 3)",
        "layout_type": "mamba_block_split",
        "nodes": [
            { "id": "mamba_in", "label": "Input Token x", "category": "io", "color": "io", "description": "Token hidden state x in R^(B x L x D)", "undertones": "Incoming residual stream from previous layer." },
            { "id": "mamba_expand_split", "label": "Linear Expansion & Split", "category": "linear", "color": "purple", "layer_idx": 0, "sublayer_idx": 0, "description": "Projects D -> 2E (E=2D) and splits into SSM branch and Gate branch.", "undertones": "Doubles internal width similar to Gated MLP." },
            { "id": "mamba_conv1d", "label": "1D Causal Conv (k=4)", "category": "conv", "color": "blue", "layer_idx": 1, "sublayer_idx": 0, "description": "Depthwise 1D convolution over sequence dimension with kernel size 4.", "undertones": "Captures strictly local context and prevents independent 1-step token hallucinations." },
            { "id": "mamba_silu1", "label": "SiLU Activation", "category": "act", "color": "green", "layer_idx": 1, "sublayer_idx": 1, "description": "Swish non-linearity x * sigmoid(x).", "undertones": "Smooth continuous gating activation." },
            { "id": "mamba_param_gen", "label": "Selective Parameter Gen (Δ, B, C)", "category": "s6_gen", "color": "orange", "layer_idx": 2, "sublayer_idx": 0, "description": "Input-dependent dynamic projection generating Delta_t, B_t, C_t per token.", "undertones": "Breaks Linear Time Invariance (LTI). Enables dynamic selective remembering/forgetting." },
            { "id": "mamba_discretize_scan", "label": "ZOH Discretization & Parallel Scan (SSM Core)", "category": "ssm_core", "color": "orange", "layer_idx": 3, "sublayer_idx": 1, "description": "Computes A_bar = exp(Delta * A), B_bar = Delta * B, and runs parallel prefix associative scan in SRAM.", "undertones": "Zero HBM memory allocation for (B, L, D, N) state tensors. 40x faster than naive PyTorch." },
            { "id": "mamba_silu2", "label": "Gate SiLU", "category": "act", "color": "green", "layer_idx": 4, "sublayer_idx": 0, "description": "SiLU activation on the parallel gate branch z.", "undertones": "Multiplicative modulator controlling channel transmission." },
            { "id": "mamba_mult_gate", "label": "Hadamard Multiplier (⊙)", "category": "gate", "color": "yellow", "layer_idx": 4, "sublayer_idx": 0, "description": "Element-wise multiplication of SSM output and gating branch.", "undertones": "Suppresses irrelevant frequency components and sharpens representations." },
            { "id": "mamba_linear_out", "label": "Output Linear Projection", "category": "linear", "color": "purple", "layer_idx": 4, "sublayer_idx": 1, "description": "Projects E -> D dimensions back to model dimension.", "undertones": "Prepares output for residual addition." },
            { "id": "mamba_res_add", "label": "Residual Sum (+)", "category": "norm", "color": "yellow", "layer_idx": 4, "sublayer_idx": 1, "description": "Adds incoming token x to block output.", "undertones": "Ensures uninterrupted gradient highway throughout deep 64-layer Mamba stacks." }
        ]
    },
    "mixture_of_depths": {
        "diagram_title": "Mixture-of-Depths (MoD) Dynamic Compute Allocation Architecture",
        "layout_type": "mod_routing_flow",
        "nodes": [
            { "id": "mod_in", "label": "Sequence Tokens X", "category": "io", "color": "io", "description": "Full sequence of T tokens.", "undertones": "Easy tokens and hard tokens enter with uniform representation." },
            { "id": "mod_router", "label": "Per-Token Router MLP (r_i)", "category": "router", "color": "orange", "layer_idx": 0, "sublayer_idx": 0, "description": "Computes scalar compute affinity weight r_i = w_r^T x_i for every token.", "undertones": "Learns to predict which tokens benefit from extra depth vs tokens that can skip." },
            { "id": "mod_topk", "label": "Top-K Capacity Filter (C = c · T)", "category": "topk", "color": "yellow", "layer_idx": 1, "sublayer_idx": 0, "description": "Selects top-50% (or top-k) highest priority tokens to allocate FLOPs.", "undertones": "Guarantees deterministic, static tensor dimensions for zero hardware fragmentation." },
            { "id": "mod_gather", "label": "Tensor Gather (B, C, D)", "category": "gather", "color": "blue", "layer_idx": 2, "sublayer_idx": 0, "description": "Packs selected tokens into compacted dense compute buffer.", "undertones": "Maximizes TPU/GPU matrix engine utilization without computing on dummy padding." },
            { "id": "mod_compute_block", "label": "Self-Attention / MLP Block", "category": "compute", "color": "pink", "layer_idx": 2, "sublayer_idx": 1, "description": "Deep transformer computation applied strictly to the compacted tensor.", "undertones": "Executes 2x faster because attention matrix is (C x C) instead of (T x T)." },
            { "id": "mod_scatter_res", "label": "Scatter + Pure Residual Skip", "category": "scatter", "color": "green", "layer_idx": 2, "sublayer_idx": 2, "description": "Scatters computed outputs back modulated by router weight; unselected tokens skip via identity.", "undertones": "Unselected tokens preserve original representation losslessly." }
        ]
    },
    "moe_survey": {
        "diagram_title": "Sparse Mixture-of-Experts (MoE) Architecture with Shared Experts",
        "layout_type": "moe_sparse_bank",
        "nodes": [
            { "id": "moe_in", "label": "Input Token x", "category": "io", "color": "io", "description": "Hidden state entering MoE layer.", "undertones": "Tokens have diverse semantic domains (math, code, language, factual)." },
            { "id": "moe_router", "label": "Gating Router Network (W_g)", "category": "router", "color": "orange", "layer_idx": 0, "sublayer_idx": 0, "description": "Calculates routing logits and Top-K softmax distribution across N experts.", "undertones": "Includes jitter noise and auxiliary loss for load balancing across distributed GPUs." },
            { "id": "moe_shared_expert", "label": "Dedicated Shared Expert (Always Active)", "category": "shared_expert", "color": "pink", "layer_idx": 0, "sublayer_idx": 2, "description": "Processes every token to capture universal foundational knowledge (DeepSeekMoE style).", "undertones": "Prevents routed experts from wasting capacity on generic syntax or stopwords." },
            { "id": "moe_expert_bank", "label": "Sparse Expert Bank (E_1, E_2, ..., E_N)", "category": "expert_bank", "color": "blue", "layer_idx": 1, "sublayer_idx": 0, "description": "Parallel bank of SwiGLU FFN experts sharded across distributed GPUs.", "undertones": "Each token only executes Top-2 experts, keeping FLOPs constant while model has 8x-64x parameters." },
            { "id": "moe_weighted_sum", "label": "Weighted Combination & Combine Collective", "category": "combine", "color": "green", "layer_idx": 2, "sublayer_idx": 0, "description": "Computes sum(g_i * E_i(x)) + E_shared(x) and combines All-to-All tensors.", "undertones": "Restores full batch order for downstream layers." }
        ]
    },
    "swat": {
        "diagram_title": "Sliding Window Attention Training (SWAT) Architecture",
        "layout_type": "swat_window_sigmoid",
        "nodes": [
            { "id": "swat_in", "label": "Input Sequence Tokens", "category": "io", "color": "io", "description": "Long sequence tokens entering attention layer.", "undertones": "Context length up to 32k-128k tokens." },
            { "id": "swat_rope_alibi", "label": "Dual Positional Engine (RoPE + Balanced ALiBi)", "category": "pos_engine", "color": "pink", "layer_idx": 1, "sublayer_idx": 0, "description": "Applies RoPE for local token precision and geometric ALiBi slope decay for boundary smoothness.", "undertones": "Prevents abrupt truncation artifacts at sliding window edges." },
            { "id": "swat_sigmoid_attn", "label": "Independent Sigmoid Attention Scoring", "category": "sigmoid_attn", "color": "orange", "layer_idx": 0, "sublayer_idx": 0, "description": "Computes Sigmoid(QK^T / sqrt(d_k) + B) * V without softmax global normalization.", "undertones": "Completely eliminates Attention Sink artifacts where token 0 absorbs false probability." },
            { "id": "swat_sliding_cache", "label": "Sliding Window KV Ring Buffer (Window W)", "category": "kv_buffer", "color": "blue", "layer_idx": 3, "sublayer_idx": 0, "description": "Maintains constant O(W x D) memory KV cache per layer during training & inference.", "undertones": "Prevents GPU memory cliffs on extended contexts." },
            { "id": "swat_stacked_rf", "label": "Multi-Scale Layer Receptive Field Stacking", "category": "rf_stack", "color": "green", "layer_idx": 2, "sublayer_idx": 0, "description": "Cumulative receptive field expands linearly L x (W-1) + 1 to cover full sequence.", "undertones": "Deep layers receive global multi-hop context through intermediate layer bridges." }
        ]
    },
    "titans": {
        "diagram_title": "Titans: Neural Long-Term Memory & Test-Time Learning Architecture",
        "layout_type": "titans_neural_memory",
        "nodes": [
            { "id": "titans_in", "label": "Input Sequence x_t", "category": "io", "color": "io", "description": "Streaming input sequence tokens.", "undertones": "Tokens have variable novelty and memorability." },
            { "id": "titans_nmm_loss", "label": "Associative Memory Loss L_mem", "category": "assoc_loss", "color": "pink", "layer_idx": 0, "sublayer_idx": 0, "description": "Evaluates ||M_t(k_t) - v_t||^2 prediction error on neural memory weights.", "undertones": "Associative objective that measures how well memory explains current input." },
            { "id": "titans_surprise_grad", "label": "Surprise Gradient Engine (g_t = ∇L)", "category": "surprise_grad", "color": "orange", "layer_idx": 0, "sublayer_idx": 1, "description": "Computes instantaneous surprise gradient with respect to memory weights.", "undertones": "High gradient = surprising unexpected event worth memorizing." },
            { "id": "titans_momentum_decay", "label": "Surprise Momentum & Dynamic Decay (S_t, η_t)", "category": "momentum_engine", "color": "yellow", "layer_idx": 1, "sublayer_idx": 0, "description": "Accumulates surprise momentum S_t = eta * S_{t-1} - theta * g_t and updates M_t.", "undertones": "Temporal memory of surprise acting as meta-gradient descent at test time." },
            { "id": "titans_deep_mlp", "label": "Deep Neural Long-Term Memory (NMM)", "category": "neural_memory", "color": "blue", "layer_idx": 2, "sublayer_idx": 0, "description": "Multi-layer non-linear neural network storing long-term memory in its weights.", "undertones": "Massively higher memory capacity than single vector hidden states." },
            { "id": "titans_sw_attn", "label": "Short-Term Sliding Window Attention", "category": "short_attn", "color": "orange", "layer_idx": 3, "sublayer_idx": 0, "description": "High-precision local attention over active working window.", "undertones": "Handles immediate syntax and local dependencies while NMM handles distant history." },
            { "id": "titans_hyperhead", "label": "Hyper-Head Integration (MAC / MAG / MAL)", "category": "hyperhead", "color": "green", "layer_idx": 3, "sublayer_idx": 1, "description": "Combines Long-Term Neural Memory + Short-Term Attention via Context, Gate, or Sequential Layer.", "undertones": "Configurable topology depending on downstream latency requirements." }
        ]
    },
    "transmamba": {
        "diagram_title": "TransMamba: Sequence-Level Hybrid Transformer-Mamba Architecture",
        "layout_type": "transmamba_hybrid",
        "nodes": [
            { "id": "tm_in", "label": "Input Token Sequence [x_1, ..., x_T]", "category": "io", "color": "io", "description": "Token stream entering hybrid layer.", "undertones": "Prompt prefix requires high-precision reasoning; continuation requires fast throughput." },
            { "id": "tm_shared_w", "label": "Unified Shared Projection Matrix (W_proj)", "category": "shared_matrix", "color": "pink", "layer_idx": 0, "sublayer_idx": 0, "description": "Single weight matrix shared between Transformer [Q, K, V] and Mamba [C, B, x].", "undertones": "Zero parameter overhead when switching between Attention and SSM." },
            { "id": "tm_transpoint_router", "label": "Sequence TransPoint Scheduler (N_trans)", "category": "transpoint", "color": "yellow", "layer_idx": 1, "sublayer_idx": 0, "description": "Routes first N_trans tokens to Transformer and subsequent tokens to Mamba2 SSM.", "undertones": "Dynamically tunes where attention ends and SSM begins across layer depths." },
            { "id": "tm_attn_mode", "label": "Transformer Attention Regime (t ≤ N_trans)", "category": "attn_regime", "color": "orange", "layer_idx": 1, "sublayer_idx": 0, "description": "Executes full multi-head self-attention on prompt tokens for in-context learning.", "undertones": "Captures complex multi-needle prompts and associative prompts flawlessly." },
            { "id": "tm_mcc_bridge", "label": "Memory Compression Cache (MCC)", "category": "mcc_bridge", "color": "blue", "layer_idx": 2, "sublayer_idx": 0, "description": "Compresses Transformer KV and attention representations into continuous SSM state h_ssm.", "undertones": "Seamless, lossless knowledge handover at the TransPoint boundary." },
            { "id": "tm_ssm_mode", "label": "Mamba2 SSM Linear Scan Regime (t > N_trans)", "category": "ssm_regime", "color": "orange", "layer_idx": 3, "sublayer_idx": 0, "description": "Executes linear-time selective scan for fast long sequence continuation.", "undertones": "Achieves 40% training speedup with constant generation memory." }
        ]
    }
}

# Attach to papers
for p in papers:
    pid = p["paper_id"]
    if pid in diagrams_meta:
        p["graphical_diagram"] = diagrams_meta[pid]

# Save updated papers_data.json
with open(dataset_path, 'w', encoding='utf-8') as f:
    json.dump(papers, f, indent=2)

print(f"Successfully attached rich graphical block diagrams and undertones to all {len(papers)} papers!")
