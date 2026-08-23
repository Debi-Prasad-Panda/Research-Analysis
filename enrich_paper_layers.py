import json

with open("papers_data.json", "r", encoding="utf-8") as f:
    papers = json.load(f)

# Comprehensive research metadata for all layers of all 7 papers
ENRICHMENTS = {
    "transformer": {
        "Input & Positional Encoding Layer": {
            "tensor_in": "(B, L)",
            "tensor_out": "(B, L, 512)",
            "flops_weight": "O(L · D)",
            "memory_weight": "O(B · L · D)",
            "sublayer_extras": [
                {
                    "name": "Token Embedding Lookup",
                    "in_shape": "(B, L) integer IDs",
                    "out_shape": "(B, L, 512) float embeddings",
                    "math_breakdown": "Maps integer token indices {0..V-1} into continuous vector space ℝ^512 and multiplies by sqrt(512)=22.62 to balance magnitude with positional encodings.",
                    "hyperparameters": [{"param": "V", "val": "37,000", "desc": "Vocabulary size"}, {"param": "d_model", "val": "512", "desc": "Embedding dimension"}],
                    "hardware_notes": "Memory-bound gather operation from GPU DRAM. High cache hit rate when batch sequences share subwords.",
                    "failure_modes": "Without sqrt(d_model) scaling, embedding values become negligible relative to PE sinusoidal amplitudes, degrading early training convergence.",
                    "pytorch_code": "x_emb = self.embedding(token_ids) * math.sqrt(self.d_model)"
                },
                {
                    "name": "Sinusoidal Positional Encoding",
                    "in_shape": "(B, L, 512)",
                    "out_shape": "(B, L, 512)",
                    "math_breakdown": "Deterministic harmonic waves. Even indices use sin(pos / 10000^(2i/d)), odd use cos. Dot product PE(pos+k) · PE(pos) is a function of offset k alone, providing translation invariance.",
                    "hyperparameters": [{"param": "max_len", "val": "5,000", "desc": "Precomputed horizon"}, {"param": "base", "val": "10,000", "desc": "Wavelength geometric base"}],
                    "hardware_notes": "Precomputed and pinned in GPU constant memory. Zero runtime FLOPs during inference via static lookup.",
                    "failure_modes": "Cannot extrapolate accurately beyond context horizons unseen during training without frequency rescaling (RoPE / YaRN).",
                    "pytorch_code": "pe[:, 0::2] = torch.sin(position * div_term)\npe[:, 1::2] = torch.cos(position * div_term)\nx = x + pe[:x.size(1)]"
                },
                {
                    "name": "Embedding Dropout & Summation",
                    "in_shape": "(B, L, 512)",
                    "out_shape": "(B, L, 512)",
                    "math_breakdown": "Randomly zeros out channels with probability P_drop=0.1, scaling surviving elements by 1/(1-0.1) = 1.11 to preserve expected activation energy.",
                    "hyperparameters": [{"param": "P_drop", "val": "0.1", "desc": "Dropout probability"}],
                    "hardware_notes": "Fused into the embedding addition kernel to eliminate redundant DRAM roundtrips.",
                    "failure_modes": "Overly high dropout rates (e.g. >0.2) cause representation collapse and disrupt positional signal alignment.",
                    "pytorch_code": "x = self.dropout(x_emb + pe)"
                }
            ]
        },
        "Encoder Block Stack (N = 6 Layers)": {
            "tensor_in": "(B, L, 512)",
            "tensor_out": "(B, L, 512)",
            "flops_weight": "O(L^2 · D + L · D^2)",
            "memory_weight": "O(B · h · L^2)",
            "sublayer_extras": [
                {
                    "name": "Multi-Head Self-Attention (MHSA)",
                    "in_shape": "(B, L, 512)",
                    "out_shape": "(B, L, 512)",
                    "math_breakdown": "Q=X W_Q, K=X W_K, V=X W_V -> Reshape to (B, h=8, L, d_k=64) -> Attention score S = Q K^T / 8 -> Softmax(S) -> Multiply by V -> Concat 8 heads -> Project through W_O.",
                    "hyperparameters": [{"param": "h", "val": "8", "desc": "Attention heads"}, {"param": "d_k", "val": "64", "desc": "Key/Query head dim"}, {"param": "d_v", "val": "64", "desc": "Value head dim"}],
                    "hardware_notes": "FlashAttention fuses QK^T, Softmax, and SV in on-chip SRAM, cutting memory bandwidth from O(L^2) to O(L).",
                    "failure_modes": "Without 1/sqrt(d_k) scaling, for large d_k dot products grow large, pushing softmax into regions with vanishing gradients.",
                    "pytorch_code": "scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)\nattn = torch.softmax(scores, dim=-1)\nout = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, D)\nout = self.w_o(out)"
                },
                {
                    "name": "Residual Connection & Post-LayerNorm 1",
                    "in_shape": "(B, L, 512)",
                    "out_shape": "(B, L, 512)",
                    "math_breakdown": "Identity highway: z = X + Dropout(MHSA(X)). Normalization: LN(z) = ((z - μ) / sqrt(σ^2 + ε)) · γ + β computed across channel dimension d=512.",
                    "hyperparameters": [{"param": "ε", "val": "1e-5", "desc": "Numerical stabilizer"}, {"param": "γ, β", "val": "ℝ^512", "desc": "Learnable gain and bias"}],
                    "hardware_notes": "Pre-LayerNorm variant is preferred in modern transformers to stabilize gradients in deep stacks without warm-up heuristics.",
                    "failure_modes": "Post-LN can suffer from vanishing gradients at initialization for N > 12 layers unless strict learning rate warmup is enforced.",
                    "pytorch_code": "x = self.norm1(x + self.dropout(self.mha(x)))"
                },
                {
                    "name": "Position-Wise Feed-Forward Network (FFN)",
                    "in_shape": "(B, L, 512)",
                    "out_shape": "(B, L, 512)",
                    "math_breakdown": "Two-stage linear expansion: W_1 expands 512 -> 2048 (4x factor). ReLU introduces point-wise non-linearity. W_2 projects 2048 -> 512 back to model dimension.",
                    "hyperparameters": [{"param": "d_ff", "val": "2048", "desc": "Inner expansion dimension"}, {"param": "activation", "val": "ReLU / GeLU", "desc": "Non-linear activation"}],
                    "hardware_notes": "Matrix multiplications GEMM 1 and GEMM 2 dominate parameter count (66% of total block parameters). High compute density on Tensor Cores.",
                    "failure_modes": "Dead ReLUs if learning rate is excessively high; modern models use SwiGLU / GeLU to mitigate dying neurons.",
                    "pytorch_code": "x = self.w2(torch.relu(self.w1(x)))"
                },
                {
                    "name": "Residual Connection & Post-LayerNorm 2",
                    "in_shape": "(B, L, 512)",
                    "out_shape": "(B, L, 512)",
                    "math_breakdown": "Second residual bypass and LayerNorm step: X_out = LN(X_norm1 + Dropout(FFN(X_norm1))). Prepares representations for subsequent stack layer.",
                    "hyperparameters": [{"param": "d_model", "val": "512", "desc": "Channel width"}],
                    "hardware_notes": "Fused Add+Norm CUDA kernel reduces global memory access latency.",
                    "failure_modes": "LayerNorm saturation if gamma scale parameters drift unchecked.",
                    "pytorch_code": "x = self.norm2(x + self.dropout(self.ffn(x)))"
                }
            ]
        },
        "Decoder Block Stack (N = 6 Layers)": {
            "tensor_in": "(B, L_target, 512) & (B, L_source, 512)",
            "tensor_out": "(B, L_target, 512)",
            "flops_weight": "O(L_tgt^2 · D + L_tgt · L_src · D)",
            "memory_weight": "O(B · N · (L_tgt + L_src) · D)",
            "sublayer_extras": [
                {
                    "name": "Masked Multi-Head Self-Attention",
                    "in_shape": "(B, L, 512)",
                    "out_shape": "(B, L, 512)",
                    "math_breakdown": "Applies additive causal mask M where M_ij = -inf for j > i before softmax. Prevents position i from attending to future tokens j > i.",
                    "hyperparameters": [{"param": "mask_val", "val": "-1e9", "desc": "Minus infinity approximation"}],
                    "hardware_notes": "During inference, cached past keys/values (KV cache) avoid recomputing historical token projections: O(L) generation step.",
                    "failure_modes": "Causal mask leakage leads to training-inference discrepancy and autoregressive degradation.",
                    "pytorch_code": "scores = scores.masked_fill(mask == 0, -1e9)\nattn = torch.softmax(scores, dim=-1)"
                },
                {
                    "name": "Encoder-Decoder Cross-Attention",
                    "in_shape": "Q: (B, L_tgt, 512), KV: (B, L_src, 512)",
                    "out_shape": "(B, L_tgt, 512)",
                    "math_breakdown": "Queries projected from decoder stream, Keys and Values projected from encoder final stack output: CrossAttn = softmax(Q_dec K_enc^T / sqrt(d_k)) V_enc.",
                    "hyperparameters": [{"param": "h", "val": "8", "desc": "Cross heads"}],
                    "hardware_notes": "Encoder KV states are static throughout the entire decoding phase and computed once during prompt prefill.",
                    "failure_modes": "Attention dispersion over long source sequences; alignment collapse if encoder representations lack contrast.",
                    "pytorch_code": "cross_attn = torch.softmax(q_dec @ k_enc.transpose(-2, -1) / math.sqrt(d_k), -1) @ v_enc"
                },
                {
                    "name": "Decoder Position-Wise FFN & LayerNorm",
                    "in_shape": "(B, L, 512)",
                    "out_shape": "(B, L, 512)",
                    "math_breakdown": "Identical structure to encoder FFN: expands channel dimension by 4x to synthesize cross-lingual or next-token contextual features.",
                    "hyperparameters": [{"param": "d_ff", "val": "2048", "desc": "FFN expansion"}],
                    "hardware_notes": "Standard high-throughput GEMM layer.",
                    "failure_modes": "Representation bottleneck if intermediate capacity d_ff is constrained below 4x.",
                    "pytorch_code": "x = self.norm3(x + self.dropout(self.ffn(x)))"
                }
            ]
        },
        "Final Linear & Softmax Output Head": {
            "tensor_in": "(B, L, 512)",
            "tensor_out": "(B, L, V)",
            "flops_weight": "O(B · L · D · V)",
            "memory_weight": "O(B · L · V)",
            "sublayer_extras": [
                {
                    "name": "Vocabulary Linear Projection",
                    "in_shape": "(B, L, 512)",
                    "out_shape": "(B, L, 37000)",
                    "math_breakdown": "Projects final hidden states through transpose of embedding matrix W_e^T (weight tying) to produce unnormalized logits across vocabulary.",
                    "hyperparameters": [{"param": "tied_weights", "val": "True", "desc": "Shared with embedding matrix"}],
                    "hardware_notes": "Large matrix multiplication across vocabulary. Can be a latency bottleneck for V > 100,000 without split-K kernels.",
                    "failure_modes": "Logit drift without proper layer normalization preceding the projection head.",
                    "pytorch_code": "logits = self.linear_head(decoder_out)"
                },
                {
                    "name": "Next-Token Softmax Classifier",
                    "in_shape": "(B, L, 37000)",
                    "out_shape": "(B, L, 37000)",
                    "math_breakdown": "P(y_t+1 = i | context) = exp(z_i) / sum_j exp(z_j). Computes calibrated conditional probability distribution over dictionary.",
                    "hyperparameters": [{"param": "temperature", "val": "1.0", "desc": "Logit scaling parameter"}],
                    "hardware_notes": "Fused CrossEntropyLoss in PyTorch computes log_softmax and NLL loss simultaneously in FP32 to avoid numerical underflow.",
                    "failure_modes": "Numerical overflow when max logit is large; fixed via subtraction of max(z) before exponentiation.",
                    "pytorch_code": "probs = torch.softmax(logits / temperature, dim=-1)"
                }
            ]
        }
    },
    "mamba": {
        "Selective State Space (SSM) Layer": {
            "tensor_in": "(B, L, D)",
            "tensor_out": "(B, L, D)",
            "flops_weight": "O(L · D · N_state)",
            "memory_weight": "O(B · D · N_state) [O(1) in L]",
            "sublayer_extras": [
                {
                    "name": "Input Linear Expansion & Dual-Branch Split",
                    "in_shape": "(B, L, D)",
                    "out_shape": "x: (B, L, E), z: (B, L, E) where E = 2D",
                    "math_breakdown": "Projects input x in ℝ^D to expanded 2E dimension via linear projection matrix W_in in ℝ^(D x 2E), then chunks into SSM branch x and Gate branch z.",
                    "hyperparameters": [{"param": "E", "val": "2 · D", "desc": "Expanded inner dimension"}, {"param": "D", "val": "2048", "desc": "Model dimension"}],
                    "hardware_notes": "Executed as a single fused GEMM across the 2E width to maximize memory bus saturation.",
                    "failure_modes": "Under-expansion (E < 2D) constrains state space expressivity below Transformer parity.",
                    "pytorch_code": "xz = self.in_proj(x)\nx, z = xz.chunk(2, dim=-1)"
                },
                {
                    "name": "1D Causal Convolution Sublayer",
                    "in_shape": "(B, L, E)",
                    "out_shape": "(B, L, E)",
                    "math_breakdown": "1D convolution with kernel size k=4 and causal left-padding of k-1 zeros. Prevents future token leakage and models local n-gram token interactions.",
                    "hyperparameters": [{"param": "d_conv", "val": "4", "desc": "Convolution kernel size"}],
                    "hardware_notes": "During inference, cached in a circular ring buffer of size (B, E, k-1), eliminating recomputation.",
                    "failure_modes": "Non-causal padding leaks future sequence context, causing autoregressive failure.",
                    "pytorch_code": "x = self.conv1d(x.transpose(1, 2))[:, :, :L].transpose(1, 2)"
                },
                {
                    "name": "Input-Dependent Parameter Generation (Selective Delta, B, C)",
                    "in_shape": "(B, L, E)",
                    "out_shape": "Δ: (B, L, E), B: (B, L, N), C: (B, L, N)",
                    "math_breakdown": "B(t) = Linear_B(x_t), C(t) = Linear_C(x_t), Δ(t) = Softplus(Parameter_Δ + Linear_Δ(x_t)). This selectivity allows the model to filter out irrelevant tokens dynamically.",
                    "hyperparameters": [{"param": "N_state", "val": "16", "desc": "SSM hidden state size"}, {"param": "dt_rank", "val": "D / 16", "desc": "Delta projection rank"}],
                    "hardware_notes": "Low-rank delta projection keeps parameter count modest while allowing high-dimensional state transitions.",
                    "failure_modes": "Without Softplus activation, negative Delta values cause exponential state explosion or numerical divergence.",
                    "pytorch_code": "delta = F.softplus(self.dt_proj(x))\nB = self.b_proj(x)\nC = self.c_proj(x)"
                },
                {
                    "name": "Hardware-Aware Zero-Order Hold (ZOH) & Parallel Associative Scan",
                    "in_shape": "x: (B, L, E), Δ: (B, L, E), A: (E, N)",
                    "out_shape": "(B, L, E)",
                    "math_breakdown": "Ā = exp(Δ A), B̄ = (Δ A)^(-1)(exp(Δ A) - I) · Δ B ≈ Δ B. Parallel scan computes h_t = Ā_t h_t-1 + B̄_t x_t across sequence in O(L log L) work in fast SRAM.",
                    "hyperparameters": [{"param": "A_init", "val": "HiPPO / S4D", "desc": "Continuous transition initialization"}],
                    "hardware_notes": "Fused CUDA kernel keeps the entire 4D recurrence state in on-chip SRAM, avoiding slow HBM DRAM read/write cycles.",
                    "failure_modes": "Evaluating naive sequential recurrence in PyTorch causes 20-40x slowdown due to GPU kernel launch and DRAM latency.",
                    "pytorch_code": "y = selective_scan_fn(x, delta, A, B, C, D)"
                },
                {
                    "name": "Multiplicative Gating (Hadamard Product) & Output Linear Projection",
                    "in_shape": "y: (B, L, E), z: (B, L, E)",
                    "out_shape": "(B, L, D)",
                    "math_breakdown": "Gating: y_gated = y ⊙ SiLU(z). Output projection: out = W_out · y_gated + x_residual. Re-compresses inner dimension E back to model dimension D.",
                    "hyperparameters": [{"param": "W_out", "val": "ℝ^(E x D)", "desc": "Output linear projection"}],
                    "hardware_notes": "SiLU element-wise product and residual sum are fused directly into output projection GEMM epilogue.",
                    "failure_modes": "Omitting the multiplicative gate impairs associative recall and selective in-context retrieval benchmarks.",
                    "pytorch_code": "out = self.out_proj(y * F.silu(z)) + residual"
                }
            ]
        }
    },
    "mixture_of_depths": {
        "Dynamic Routing & Capacity Allocation Layer": {
            "tensor_in": "(B, T, D)",
            "tensor_out": "(B, T, D)",
            "flops_weight": "O(c · T · D^2 + c^2 · T^2 · D)",
            "memory_weight": "O(c · B · T · D)",
            "sublayer_extras": [
                {
                    "name": "Per-Token Routing Prediction",
                    "in_shape": "(B, T, D)",
                    "out_shape": "(B, T) router scalar weights",
                    "math_breakdown": "Linear router vector w_r in ℝ^D computes scalar score r_i = w_r^T x_i for each token position. Represents importance of executing block computation for token i.",
                    "hyperparameters": [{"param": "c", "val": "0.125", "desc": "Capacity factor (12.5% tokens compute)"}],
                    "hardware_notes": "O(T · D) lightweight linear projection. Vectorized across entire batch in a single pass.",
                    "failure_modes": "Router collapse where router assigns identical scores to all tokens; mitigated by auxiliary router load loss.",
                    "pytorch_code": "router_logits = self.router(x).squeeze(-1)"
                },
                {
                    "name": "Top-K Capacity Filtering & Sorting",
                    "in_shape": "(B, T)",
                    "out_shape": "Selected indices I of length C = ⌊c · T⌋",
                    "math_breakdown": "Computes capacity C = ⌊c · T⌋. Identifies indices of top C routing scores along sequence dimension: I = TopK(router_logits, k=C).",
                    "hyperparameters": [{"param": "C", "val": "c · T", "desc": "Fixed token capacity budget"}],
                    "hardware_notes": "Sorting is performed on-device. Fixed tensor shape C ensures statically compiled GPU execution kernels.",
                    "failure_modes": "Token dropping during bursty informational sequences if capacity budget C is set too aggressively low (<0.10).",
                    "pytorch_code": "topk_weights, topk_indices = torch.topk(router_logits, k=C, dim=-1)"
                },
                {
                    "name": "Tensor Gather to Dense Sub-Batch",
                    "in_shape": "(B, T, D) full sequence",
                    "out_shape": "(B, C, D) dense compute sub-tensor",
                    "math_breakdown": "Packs non-contiguous selected tokens into contiguous dense tensor: X_selected = gather(X, dim=1, index=I).",
                    "hyperparameters": [{"param": "D", "val": "4096", "desc": "Hidden dimension"}],
                    "hardware_notes": "High bandwidth gather kernel. Once packed, subsequent attention and MLP operate at 100% compute efficiency.",
                    "failure_modes": "Memory fragmentation if gather is not performed into contiguous buffer.",
                    "pytorch_code": "x_selected = torch.gather(x, 1, topk_indices.unsqueeze(-1).expand(-1, -1, D))"
                },
                {
                    "name": "Heavy Transformer / MLP Block Execution",
                    "in_shape": "(B, C, D)",
                    "out_shape": "(B, C, D)",
                    "math_breakdown": "Executes full Multi-Head Attention and FFN strictly over the C selected tokens. Reduces attention compute by (1 - c^2) and MLP by (1 - c).",
                    "hyperparameters": [{"param": "c", "val": "0.125", "desc": "Compute reduction factor"}],
                    "hardware_notes": "Attention complexity drops from O(T^2) to O(C^2) = O(c^2 T^2), yielding ~64x attention speedup for c=0.125.",
                    "failure_modes": "Information isolation if critical syntactic tokens are bypassed across too many consecutive layers.",
                    "pytorch_code": "y_selected = self.transformer_block(x_selected)"
                },
                {
                    "name": "Scatter & Weighted Residual Highway Bypass",
                    "in_shape": "y_selected: (B, C, D), x_orig: (B, T, D)",
                    "out_shape": "(B, T, D) output sequence",
                    "math_breakdown": "Selected tokens updated as y_i = r_i · Y_selected_i + x_i. Bypassed tokens undergo pure identity residual: y_j = x_j.",
                    "hyperparameters": [{"param": "residual_mode", "val": "weighted", "desc": "Differentiable router weight"}],
                    "hardware_notes": "Scatter operation restores original sequence topology without token loss or order permutation.",
                    "failure_modes": "Gradient starvation for unselected tokens if router weights are not properly normalized.",
                    "pytorch_code": "out = x.clone()\nout.scatter_add_(1, topk_indices.unsqueeze(-1).expand(-1, -1, D), y_selected * topk_weights.unsqueeze(-1))"
                }
            ]
        }
    },
    "moe_survey": {
        "Sparse Mixture of Experts (MoE) Layer": {
            "tensor_in": "(B, L, D)",
            "tensor_out": "(B, L, D)",
            "flops_weight": "O(k · L · D^2) [Active FLOPs]",
            "memory_weight": "O(E · D^2) [Total Params]",
            "sublayer_extras": [
                {
                    "name": "Gating Router Network",
                    "in_shape": "(B, L, D)",
                    "out_shape": "(B, L, E) routing probabilities",
                    "math_breakdown": "H(x) = W_g x + Noise. Softmax is taken over Top-k expert logits: G(x)_i = Softmax(TopK(H(x), k)).",
                    "hyperparameters": [{"param": "E", "val": "8 / 16 / 64", "desc": "Total expert count"}, {"param": "k", "val": "2", "desc": "Active experts per token"}],
                    "hardware_notes": "Lightweight routing network. Crucial for token-to-device load balancing across distributed GPU clusters.",
                    "failure_modes": "Expert starvation or routing collapse (routing all tokens to 1-2 dominant experts); prevented by auxiliary balance loss.",
                    "pytorch_code": "gate_logits = self.gate(x)\nweights, indices = torch.topk(F.softmax(gate_logits, dim=-1), k=self.k)"
                },
                {
                    "name": "Dedicated Shared Expert Processing",
                    "in_shape": "(B, L, D)",
                    "out_shape": "(B, L, D)",
                    "math_breakdown": "An unconditional, always-active feed-forward expert E_shared(x) processes all tokens regardless of router decision.",
                    "hyperparameters": [{"param": "shared_expert_count", "val": "1 or 2", "desc": "Always-active experts"}],
                    "hardware_notes": "Captures common domain-invariant representations, freeing routed experts to specialize aggressively.",
                    "failure_modes": "Excessive shared expert capacity dilutes the compute savings benefits of sparse routing.",
                    "pytorch_code": "shared_out = self.shared_expert(x)"
                },
                {
                    "name": "Sparse Expert Bank Computation",
                    "in_shape": "Dispatched tokens per expert",
                    "out_shape": "Expert outputs E_i(x)",
                    "math_breakdown": "Tokens dispatched to corresponding expert FFNs in parallel: E_i(x) = W_2,i · Act(W_1,i x).",
                    "hyperparameters": [{"param": "expert_dim", "val": "4 · D", "desc": "FFN expansion per expert"}],
                    "hardware_notes": "Expert parallelism distributes distinct expert weights across separate GPU memory ranks with All-to-All communication.",
                    "failure_modes": "Communication bottlenecks during All-to-All token dispatch if token distribution is highly skewed.",
                    "pytorch_code": "expert_out = self.experts[expert_idx](dispatched_tokens)"
                },
                {
                    "name": "Weighted Combination & Combine Collective",
                    "in_shape": "Expert responses + Shared response",
                    "out_shape": "(B, L, D)",
                    "math_breakdown": "Final layer output fuses routed responses and shared response: y = sum_{i=1}^k G(x)_i · E_i(x) + E_shared(x).",
                    "hyperparameters": [{"param": "k", "val": "2", "desc": "Combine active count"}],
                    "hardware_notes": "Fused All-to-All reduce scatter kernel merges representations back to sequence layout.",
                    "failure_modes": "Inference memory overhead: while compute is O(k), all E expert weights must reside in VRAM.",
                    "pytorch_code": "out = (expert_outputs * weights.unsqueeze(-1)).sum(dim=1) + shared_out"
                }
            ]
        }
    },
    "swat": {
        "Sliding Window Attention (SWAT) Layer": {
            "tensor_in": "(B, L, D)",
            "tensor_out": "(B, L, D)",
            "flops_weight": "O(W · L · D)",
            "memory_weight": "O(B · W · D) [Bounded Cache]",
            "sublayer_extras": [
                {
                    "name": "Dual Positional Engine (RoPE + ALiBi)",
                    "in_shape": "(B, L, D)",
                    "out_shape": "(B, L, D)",
                    "math_breakdown": "Applies 2D rotation matrices to Q and K (RoPE) and adds distance-proportional linear slope penalty -m|i-j| (ALiBi) to attention logits.",
                    "hyperparameters": [{"param": "W", "val": "2048 / 4096", "desc": "Sliding window size"}, {"param": "base_rope", "val": "500,000", "desc": "RoPE base frequency"}],
                    "hardware_notes": "Allows model to extrapolate to 100k+ tokens during inference without fine-tuning.",
                    "failure_modes": "Loss of global context if ALiBi decay slopes are too steep on deep layers.",
                    "pytorch_code": "q_rot, k_rot = apply_rotary_emb(q, k, freqs)\nscores = (q_rot @ k_rot.transpose(-2, -1)) / math.sqrt(d_k) + alibi_bias"
                },
                {
                    "name": "Independent Sigmoid Attention Scoring",
                    "in_shape": "(B, h, L, W)",
                    "out_shape": "(B, h, L, W) attention weights",
                    "math_breakdown": "Computes element-wise independent sigmoid sigma(S_ij) instead of Softmax. Prevents probability dilution over long windows.",
                    "hyperparameters": [{"param": "sigmoid_scale", "val": "1 / sqrt(d_k)", "desc": "Logit scaling factor"}],
                    "hardware_notes": "Eliminates global reduction step across sequence dimension, enabling fully local streaming GPU execution.",
                    "failure_modes": "Attention mass uncalibrated if bias term is unlearned; requires learnable scaling multiplier.",
                    "pytorch_code": "attn_weights = torch.sigmoid(scores - bias)"
                },
                {
                    "name": "Sliding Window KV Ring Buffer",
                    "in_shape": "(B, h, 1, D) new token",
                    "out_shape": "(B, h, W, D) rolling cache",
                    "math_breakdown": "Maintains bounded rolling cache of size W. Automatically overwrites entries older than current_pos - W.",
                    "hyperparameters": [{"param": "W", "val": "2048", "desc": "KV buffer capacity"}],
                    "hardware_notes": "Inference memory footprint is strictly O(W) and never increases regardless of generated sequence length.",
                    "failure_modes": "Cannot directly recall tokens generated beyond horizon W without multi-layer receptive field stacking.",
                    "pytorch_code": "kv_cache[:, :, ptr % W] = new_kv\nptr += 1"
                }
            ]
        }
    },
    "titans": {
        "Neural Long-Term Memory (NMM) Layer": {
            "tensor_in": "(B, L, D)",
            "tensor_out": "(B, L, D)",
            "flops_weight": "O(L · D^2 + W · L · D)",
            "memory_weight": "O(M_weights + B · W · D)",
            "sublayer_extras": [
                {
                    "name": "Surprise Gradient Engine",
                    "in_shape": "x_t in ℝ^D",
                    "out_shape": "g_t = ∇_M ℒ_mem",
                    "math_breakdown": "Measures prediction error on incoming token: ℒ_mem = ‖M_t(k_t) - v_t‖^2. Gradient g_t = ∇_M ℒ_mem quantifies how 'surprising' the token is.",
                    "hyperparameters": [{"param": "NMM_depth", "val": "2 layers", "desc": "Neural memory network depth"}],
                    "hardware_notes": "Gradient computed via fast backward pass through memory network during forward inference rollout.",
                    "failure_modes": "Gradient explosion if surprise loss is unclipped on out-of-distribution input tokens.",
                    "pytorch_code": "loss = F.mse_loss(self.memory(k_t), v_t)\ngrad = torch.autograd.grad(loss, self.memory.parameters())"
                },
                {
                    "name": "Surprise Momentum & Dynamic Decay",
                    "in_shape": "g_t surprise gradient",
                    "out_shape": "S_t momentum update",
                    "math_breakdown": "Updates momentum state: S_t = η_t S_t-1 - θ_t g_t. Incorporates adaptive decay (1 - α_t) to forget transient noise while storing key facts.",
                    "hyperparameters": [{"param": "η", "val": "0.9", "desc": "Momentum factor"}, {"param": "α", "val": "0.01", "desc": "Decay rate"}],
                    "hardware_notes": "Maintained in persistent GPU registers for instantaneous state transition.",
                    "failure_modes": "Memory saturation if decay factor α is set to 0 over ultra-long sequences (1M+ tokens).",
                    "pytorch_code": "momentum = eta * momentum - theta * grad\nmemory_weights = (1 - alpha) * memory_weights + momentum"
                },
                {
                    "name": "Hyper-Head Integration (MAC / MAG / MAL)",
                    "in_shape": "Memory output y_mem + Attention output y_attn",
                    "out_shape": "(B, L, D)",
                    "math_breakdown": "Memory as Context (MAC), Memory as Gating (MAG), or Memory as Layer (MAL) dynamically balances long-term neural recall with local window attention.",
                    "hyperparameters": [{"param": "arch_mode", "val": "MAG", "desc": "Memory as Gating variant"}],
                    "hardware_notes": "Gated fusion prevents long-term memory hallucinations from overriding immediate local conversational context.",
                    "failure_modes": "Gating collapse if one branch dominates early in pretraining.",
                    "pytorch_code": "gate = torch.sigmoid(self.gate_proj(y_attn))\nout = gate * y_mem + (1 - gate) * y_attn"
                }
            ]
        }
    },
    "transmamba": {
        "Hybrid Transformer-Mamba Switching Layer": {
            "tensor_in": "(B, L, D)",
            "tensor_out": "(B, L, D)",
            "flops_weight": "O(N_tr^2 · D + (L - N_tr) · D · N_state)",
            "memory_weight": "O(B · D · N_state) [Post-TransPoint]",
            "sublayer_extras": [
                {
                    "name": "Unified Shared Projection Matrix",
                    "in_shape": "(B, L, D)",
                    "out_shape": "Shared projection features",
                    "math_breakdown": "Unified matrix W_proj computes Attention Q/K/V during prompt prefill and Mamba B/C/Δ during autoregression, eliminating dual-parameter bloat.",
                    "hyperparameters": [{"param": "shared_ratio", "val": "100%", "desc": "Zero redundant parameters"}],
                    "hardware_notes": "Single set of weights in GPU memory serves both attention prefill and linear SSM generation.",
                    "failure_modes": "Gradient interference between attention and SSM objectives if learning rate scaling is mismatched.",
                    "pytorch_code": "proj_features = self.unified_proj(x)"
                },
                {
                    "name": "Sequence TransPoint Scheduler & MCC Bridge",
                    "in_shape": "Dense KV Cache at t = N_trans",
                    "out_shape": "Initial Mamba State h_0",
                    "math_breakdown": "At token threshold N_trans (e.g. 512), Memory Compression Cache (MCC) compresses dense KV cache into initial SSM state h_0, then drops KV cache from VRAM.",
                    "hyperparameters": [{"param": "N_trans", "val": "512 / 1024", "desc": "Prompt transition boundary"}],
                    "hardware_notes": "Frees 90%+ of GPU VRAM right after prompt encoding, unlocking massive batch sizes during generation.",
                    "failure_modes": "Compression loss if MCC projection is under-parameterized; mitigated by non-linear bridge network.",
                    "pytorch_code": "h_0 = self.mcc_bridge(kv_cache)\ndel kv_cache\ntorch.cuda.empty_cache()"
                }
            ]
        }
    }
}

# Update papers data
for p in papers:
    pid = p["paper_id"]
    if pid in ENRICHMENTS:
        penr = ENRICHMENTS[pid]
        for layer in p["layers"]:
            lname = layer["layer_name"]
            if lname in penr:
                lenr = penr[lname]
                layer["tensor_in"] = lenr.get("tensor_in", "(B, L, D)")
                layer["tensor_out"] = lenr.get("tensor_out", "(B, L, D)")
                layer["flops_weight"] = lenr.get("flops_weight", "O(L · D)")
                layer["memory_weight"] = lenr.get("memory_weight", "O(B · L · D)")
                
                extras = lenr.get("sublayer_extras", [])
                for i, sub in enumerate(layer["sublayers"]):
                    if i < len(extras):
                        ex = extras[i]
                        sub["in_shape"] = ex.get("in_shape", "(B, L, D)")
                        sub["out_shape"] = ex.get("out_shape", "(B, L, D)")
                        sub["math_breakdown"] = ex.get("math_breakdown", "")
                        sub["hyperparameters"] = ex.get("hyperparameters", [])
                        sub["hardware_notes"] = ex.get("hardware_notes", "")
                        sub["failure_modes"] = ex.get("failure_modes", "")
                        sub["pytorch_code"] = ex.get("pytorch_code", "")

with open("papers_data.json", "w", encoding="utf-8") as f:
    json.dump(papers, f, indent=2)

print("Enriched ALL 7 papers in papers_data.json successfully!")
