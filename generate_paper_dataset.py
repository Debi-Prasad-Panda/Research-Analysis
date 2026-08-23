import json, os, sys

# Strict JSON Schema Definition according to FIX 2
SCHEMA = {
    "required_paper_fields": [
        "paper_id",
        "title",
        "authors_venue_year",
        "core_breakthrough",
        "layers",
        "pros",
        "cons",
        "computed_metric_role",
        "reported_metrics"
    ],
    "required_layer_fields": ["layer_name", "sublayers"],
    "required_sublayer_fields": ["name", "formula_katex", "explanation", "source_section"],
    "required_pro_con_fields": ["claim", "source_section"],
    "required_reported_metric_fields": ["task", "metric_name", "score_value", "source_section", "comparison_context"]
}

papers_data = [
    {
        "paper_id": "transformer",
        "title": "Attention Is All You Need",
        "authors_venue_year": "Ashish Vaswani et al. (Google Brain & Google Research), NeurIPS 2017",
        "core_breakthrough": "Replaces recurrent and convolutional sequence models entirely with multi-head self-attention, enabling massive parallelization during training and setting the foundation for modern LLMs.",
        "computed_metric_role": "transformer",
        "layers": [
            {
                "layer_name": "Input & Positional Encoding Layer",
                "sublayers": [
                    {
                        "name": "Token Embedding Lookup",
                        "formula_katex": "X_{emb} = W_e(x) \\cdot \\sqrt{d_{model}}",
                        "explanation": "Converts input discrete token IDs into dense continuous vector representations of dimension d_model, scaled by the square root of the hidden dimension.",
                        "source_section": "Section 3.4 (Embeddings and Softmax)"
                    },
                    {
                        "name": "Sinusoidal Positional Encoding",
                        "formula_katex": "PE_{(pos, 2i)} = \\sin\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right), \\quad PE_{(pos, 2i+1)} = \\cos\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)",
                        "explanation": "Injects sequence position information into token embeddings using deterministic sine and cosine wave functions of varying frequencies, enabling the model to attend to relative positions.",
                        "source_section": "Section 3.5 (Positional Encoding)"
                    },
                    {
                        "name": "Embedding Dropout & Summation",
                        "formula_katex": "X_0 = \\text{Dropout}(X_{emb} + PE)",
                        "explanation": "Adds the positional encodings to the token embeddings element-wise and applies dropout regularization with probability P_drop = 0.1.",
                        "source_section": "Section 5.4 (Regularization)"
                    }
                ]
            },
            {
                "layer_name": "Encoder Block Stack (N = 6 Layers)",
                "sublayers": [
                    {
                        "name": "Multi-Head Self-Attention (MHSA)",
                        "formula_katex": "\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V, \\quad \\text{MultiHead}(Q,K,V) = \\text{Concat}(head_1, \\dots, head_h)W^O",
                        "explanation": "Linearly projects input tokens into h separate Query, Key, and Value subspaces (d_k = d_v = d_model / h = 64), computes scaled dot-product attention in parallel, and projects concatenated head outputs.",
                        "source_section": "Section 3.2.2 (Multi-Head Attention)"
                    },
                    {
                        "name": "Residual Connection & Post-LayerNorm 1",
                        "formula_katex": "X_{norm1} = \\text{LayerNorm}(X + \\text{Dropout}(\\text{MultiHead}(X)))",
                        "explanation": "Applies a residual connection around the attention sublayer followed by layer normalization: LayerNorm(z) = ((z - mu) / sqrt(sigma^2 + eps)) * gamma + beta.",
                        "source_section": "Section 3.1 (Encoder and Decoder Stacks)"
                    },
                    {
                        "name": "Position-Wise Feed-Forward Network (FFN)",
                        "formula_katex": "\\text{FFN}(x) = \\max(0, x W_1 + b_1) W_2 + b_2",
                        "explanation": "Applies two linear transformations with an inner ReLU activation independently and identically to each sequence position, expanding dimension from d_model=512 to d_ff=2048.",
                        "source_section": "Section 3.3 (Position-wise Feed-Forward Networks)"
                    },
                    {
                        "name": "Residual Connection & Post-LayerNorm 2",
                        "formula_katex": "X_{out} = \\text{LayerNorm}(X_{norm1} + \\text{Dropout}(\\text{FFN}(X_{norm1})))",
                        "explanation": "Applies a second residual addition and layer normalization over the FFN sublayer to form the complete encoder block output.",
                        "source_section": "Section 3.1 (Encoder Stack)"
                    }
                ]
            },
            {
                "layer_name": "Decoder Block Stack (N = 6 Layers)",
                "sublayers": [
                    {
                        "name": "Masked Multi-Head Self-Attention",
                        "formula_katex": "\\text{MaskedAttention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}} + M\\right)V, \\quad M_{ij} = \\begin{cases} 0 & j \\le i \\\\ -\\infty & j > i \\end{cases}",
                        "explanation": "Calculates self-attention over target tokens with upper-triangular masking (-infinity) to enforce causality and prevent positions from attending to subsequent tokens.",
                        "source_section": "Section 3.2.3 (Applications of Attention in our Model)"
                    },
                    {
                        "name": "Encoder-Decoder Cross-Attention",
                        "formula_katex": "\\text{CrossAttention}(Q_{dec}, K_{enc}, V_{enc}) = \\text{softmax}\\left(\\frac{Q_{dec} K_{enc}^T}{\\sqrt{d_k}}\\right) V_{enc}",
                        "explanation": "Generates Queries from the previous decoder sublayer and fetches Keys and Values from the final output representation of the encoder stack.",
                        "source_section": "Section 3.2.3 (Cross-Attention)"
                    },
                    {
                        "name": "Decoder Position-Wise FFN & LayerNorm",
                        "formula_katex": "Y_{dec} = \\text{LayerNorm}(X_{cross} + \\text{Dropout}(\\text{FFN}(X_{cross})))",
                        "explanation": "Executes identical position-wise feed-forward transformations with residual connections and layer normalization across each decoder layer.",
                        "source_section": "Section 3.1 (Decoder Stack)"
                    }
                ]
            },
            {
                "layer_name": "Final Linear & Softmax Output Head",
                "sublayers": [
                    {
                        "name": "Vocabulary Linear Projection",
                        "formula_katex": "Z_{logits} = Y_{dec} \\cdot W_v^T",
                        "explanation": "Projects the decoder final output representation from d_model dimensions to the full vocabulary size V (with weights tied to input embeddings).",
                        "source_section": "Section 3.4 (Embeddings and Softmax)"
                    },
                    {
                        "name": "Next-Token Softmax Classifier",
                        "formula_katex": "P(y_{t+1} | y_{\\le t}, x) = \\text{softmax}(Z_{logits}) = \\frac{\\exp(z_i)}{\\sum_j \\exp(z_j)}",
                        "explanation": "Converts unnormalized output logits into predictive probabilities over next target tokens.",
                        "source_section": "Section 3.4 (Softmax)"
                    }
                ]
            }
        ],
        "pros": [
            { "claim": "Full parallelization across entire sequence length during training without recurrence sequential bottlenecks.", "source_section": "Section 1 (Introduction)" },
            { "claim": "O(1) maximum path length between any two arbitrary token positions in the sequence, enabling perfect direct dependency modeling.", "source_section": "Table 1 & Section 4 (Why Self-Attention)" },
            { "claim": "Established state-of-the-art machine translation results (28.4 BLEU on WMT 2014 En-De) at a fraction of previous training cost.", "source_section": "Section 6.1 (Machine Translation)" }
        ],
        "cons": [
            { "claim": "Quadratic computational complexity O(L^2 * d) in both FLOPs and memory with respect to sequence length L.", "source_section": "Table 1 (Maximum path lengths and per-layer complexity)" },
            { "claim": "KV Cache grows linearly with sequence length during autoregressive generation, causing severe GPU memory exhaustion at long contexts.", "source_section": "Section 4 & Table 1" }
        ],
        "reported_metrics": [
            { "task": "WMT 2014 English-to-German Translation", "metric_name": "BLEU Score", "score_value": "28.4 BLEU", "source_section": "Table 2 (Translation Performance)", "comparison_context": "Outperformed previous best ensemble model by >2.0 BLEU with 3.5 days on 8 P100 GPUs" },
            { "task": "WMT 2014 English-to-French Translation", "metric_name": "BLEU Score", "score_value": "41.8 BLEU", "source_section": "Table 2 (Translation Performance)", "comparison_context": "Established single-model state-of-the-art outperforming all previous models" }
        ]
    },
    {
        "paper_id": "mamba",
        "title": "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
        "authors_venue_year": "Albert Gu (Carnegie Mellon) & Tri Dao (Princeton), arXiv 2023",
        "core_breakthrough": "Introduces input-dependent selective state space models (S6) and a hardware-aware parallel prefix scan in GPU SRAM, achieving Transformer-quality language modeling with linear O(L) time and constant O(1) inference state size.",
        "computed_metric_role": "mamba",
        "layers": [
            {
                "layer_name": "Input Expansion & Gated Splitting Layer",
                "sublayers": [
                    {
                        "name": "Channel Dimension Linear Expansion",
                        "formula_katex": "X_{expand} = x W_{in}, \\quad W_{in} \\in \\mathbb{R}^{D \\times 2E}, \\quad E = 2D",
                        "explanation": "Expands input dimension D by expansion factor (default expand=2) to dimension E=2D, doubling the channel capacity for internal representation.",
                        "source_section": "Section 3.4 (Architecture, Figure 3)"
                    },
                    {
                        "name": "Dual-Branch Splitting",
                        "formula_katex": "x_{ssm}, z = \\text{Split}(X_{expand}), \\quad x_{ssm} \\in \\mathbb{R}^{B \\times L \\times E}, \\; z \\in \\mathbb{R}^{B \\times L \\times E}",
                        "explanation": "Splits the expanded representation into two parallel pathways: the SSM sequence processing branch and the multiplicative gating branch.",
                        "source_section": "Section 3.4 (Architecture)"
                    }
                ]
            },
            {
                "layer_name": "1D Causal Convolution Sublayer",
                "sublayers": [
                    {
                        "name": "Depthwise Causal 1D Convolution",
                        "formula_katex": "x_{conv} = \\text{Conv1D}_{k=4}(x_{ssm})",
                        "explanation": "Applies a short 1D depthwise convolution with kernel size 4 along the sequence dimension to mix local contextual tokens and prevent independent channel processing before SSM.",
                        "source_section": "Section 3.4 (Architecture, Figure 3)"
                    },
                    {
                        "name": "Convolutional SiLU Activation",
                        "formula_katex": "x'_{ssm} = \\text{SiLU}(x_{conv}) = x_{conv} \\cdot \\sigma(x_{conv})",
                        "explanation": "Applies smooth non-linear Swish/SiLU activation function to the convolved features.",
                        "source_section": "Section 3.4 (Architecture)"
                    }
                ]
            },
            {
                "layer_name": "Selective Parameter Generation Sublayer (S6 Mechanism)",
                "sublayers": [
                    {
                        "name": "Input-Dependent Delta Step Size",
                        "formula_katex": "\\Delta_t = \\text{softplus}(\\text{Linear}_D(x'_{ssm, t}) + \\text{Parameter}(\\Delta_{\\text{bias}}))",
                        "explanation": "Projects token representation dynamically to produce time-step parameter Delta_t, controlling how much the model focuses on or ignores the current input token.",
                        "source_section": "Section 3.1 (Selection Mechanism, Equation 4)"
                    },
                    {
                        "name": "Input-Dependent State Matrices B and C",
                        "formula_katex": "B_t = \\text{Linear}_N(x'_{ssm, t}), \\quad C_t = \\text{Linear}_N(x'_{ssm, t}), \\quad B_t, C_t \\in \\mathbb{R}^{B \\times L \\times N}",
                        "explanation": "Dynamically projects token features to state matrices B (input matrix) and C (output matrix) with state dimension N=16, enabling selective content-based memory writing and reading.",
                        "source_section": "Section 3.1 (Selection Mechanism, Equation 4)"
                    },
                    {
                        "name": "HiPPO Transition Matrix Parameterization",
                        "formula_katex": "A_{d,n} = -\\exp(\\text{Parameter}(\\log(-A))), \\quad A \\in \\mathbb{R}^{D \\times N}",
                        "explanation": "Initializes transition matrix A using High-Order Polynomial Projection Operators (HiPPO) and maintains negative real diagonal representation for stable state decay.",
                        "source_section": "Section 2.2 (State Space Models) & Section 3.1"
                    }
                ]
            },
            {
                "layer_name": "Continuous-to-Discrete Discretization & Parallel Scan Engine",
                "sublayers": [
                    {
                        "name": "Zero-Order Hold (ZOH) Discretization",
                        "formula_katex": "\\bar{A}_t = \\exp(\\Delta_t A), \\quad \\bar{B}_t = (\\Delta_t A)^{-1}(\\exp(\\Delta_t A) - I) \\cdot (\\Delta_t B_t) \\approx \\Delta_t B_t",
                        "explanation": "Converts continuous ODE state parameters (A, B) into discrete recurrent step matrices (A_bar, B_bar) dynamic at each sequence position t.",
                        "source_section": "Section 2.2 (Equation 2) & Section 3.1"
                    },
                    {
                        "name": "Hardware-Aware SRAM Parallel Associative Scan",
                        "formula_katex": "h_t = \\bar{A}_t h_{t-1} + \\bar{B}_t x'_{ssm, t}, \\quad y_t = C_t h_t",
                        "explanation": "Fuses discretization and prefix scan directly in fast GPU on-chip SRAM memory without materializing the large (B, L, D, N) state tensors in slow High Bandwidth Memory (HBM).",
                        "source_section": "Section 3.3.2 (Hardware-aware State Expansion, Algorithm 1)"
                    }
                ]
            },
            {
                "layer_name": "Multiplicative Gating & Output Projection Layer",
                "sublayers": [
                    {
                        "name": "SiLU Multiplicative Gating",
                        "formula_katex": "y_{gated} = y \\odot \\text{SiLU}(z)",
                        "explanation": "Multiplies the output of the selective SSM block element-wise with the SiLU-activated gating branch z.",
                        "source_section": "Section 3.4 (Architecture, Figure 3)"
                    },
                    {
                        "name": "Output Linear Projection & Skip Residual",
                        "formula_katex": "y_{out} = y_{gated} W_{out} + x, \\quad W_{out} \\in \\mathbb{R}^{E \\times D}",
                        "explanation": "Projects the gated representation back to model dimension D and adds the input residual skip connection.",
                        "source_section": "Section 3.4 (Architecture)"
                    }
                ]
            }
        ],
        "pros": [
            { "claim": "Linear O(L) time and memory complexity during training with sequence length, enabling 1M+ token contexts.", "source_section": "Section 1 & Figure 8" },
            { "claim": "Constant O(1) inference state size per layer (D * N parameters) with zero KV cache memory growth during generation.", "source_section": "Section 1 (Introduction) & Section 3.3" },
            { "claim": "Solves synthetic Induction Heads and Selective Copying tasks with 100% accuracy, matching Transformer reasoning capacity.", "source_section": "Section 2.1 & Table 1, Table 2" }
        ],
        "cons": [
            { "claim": "Fixed-capacity hidden state bottleneck when storing complex multi-hop associative memories across millions of tokens compared to full non-compressed KV attention.", "source_section": "Section 5 (Discussion)" },
            { "claim": "Requires specialized hardware-fused Triton/CUDA kernels to prevent memory bandwidth bottlenecks from large state sizes.", "source_section": "Section 3.3.2 (Hardware-aware Algorithm)" }
        ],
        "reported_metrics": [
            { "task": "Selective Copying Benchmark", "metric_name": "Accuracy", "score_value": "100.0%", "source_section": "Table 1 (Selective Copying)", "comparison_context": "Mamba achieves 100% accuracy while S4 and H3 fail (<20%) at sequence length 4096" },
            { "task": "Induction Heads Task", "metric_name": "Accuracy", "score_value": "100.0%", "source_section": "Table 2 (Induction Heads)", "comparison_context": "Generalizes to sequence length 2^20 (1,048,576 tokens) with 100% accuracy" },
            { "task": "Zero-shot Common-Sense Reasoning Average (1.4B model)", "metric_name": "Accuracy", "score_value": "65.3%", "source_section": "Table 3 (Zero-shot Evaluations)", "comparison_context": "Matches and exceeds Pythia-1.4B (64.2%) and RWKV-1.5B (64.5%)" }
        ]
    },
    {
        "paper_id": "mixture_of_depths",
        "title": "Mixture-of-Depths: Dynamically allocating compute in transformer-based language models",
        "authors_venue_year": "David Raposo, Sam Ritter et al. (Google DeepMind), arXiv 2024",
        "core_breakthrough": "Dynamically routes tokens across model depth by enforcing a fixed per-block compute capacity budget (e.g. top-50% tokens), allowing unselected tokens to bypass layers via residual connections for 50% FLOP savings with no quality loss.",
        "computed_metric_role": "mod",
        "layers": [
            {
                "layer_name": "Per-Token Router Module",
                "sublayers": [
                    {
                        "name": "Scalar Token Priority Scoring",
                        "formula_katex": "r_i = w_r^T x_i, \\quad w_r \\in \\mathbb{R}^D, \\; r_i \\in \\mathbb{R}",
                        "explanation": "A learned linear router projection assigns a single scalar affinity weight r_i to each sequence token x_i denoting its requirement for layer computation.",
                        "source_section": "Section 3 (Implementing Mixture-of-Depths Transformers)"
                    },
                    {
                        "name": "Router Gradient Multiplication Pathway",
                        "formula_katex": "y_i = r_i \\cdot f(x_i) + x_i",
                        "explanation": "Multiplies the block output f(x_i) by the router weight r_i so that gradients from task loss backpropagate directly into the router weights.",
                        "source_section": "Section 3 (Routing implementation)"
                    }
                ]
            },
            {
                "layer_name": "Capacity Selection & Top-K Filtering Engine",
                "sublayers": [
                    {
                        "name": "Fixed Sequence Capacity Budgeting",
                        "formula_katex": "C = \\lfloor c \\cdot T \\rfloor, \\quad c \\in (0, 1]",
                        "explanation": "Enforces a strict static capacity budget C equal to a fraction c (e.g. c=0.5) of total sequence length T, ensuring predictable deterministic static tensor shapes on hardware.",
                        "source_section": "Section 3 (Capacity)"
                    },
                    {
                        "name": "Top-K Token Selection",
                        "formula_katex": "\\mathcal{S} = \\text{TopK}(\\{r_i\\}_{i=1}^T, C)",
                        "explanation": "Selects the C tokens with the highest router scalar scores to undergo compute inside the current layer.",
                        "source_section": "Section 3 (Routing implementation)"
                    }
                ]
            },
            {
                "layer_name": "Dynamic Computation Block (Gather -> Compute -> Scatter)",
                "sublayers": [
                    {
                        "name": "Tensor Gathering",
                        "formula_katex": "\\tilde{X} = \\text{Gather}(X, \\mathcal{S}), \\quad \\tilde{X} \\in \\mathbb{R}^{B \\times C \\times D}",
                        "explanation": "Compacts only the selected C tokens into a dense tensor of size (B, C, D) for hardware-accelerated batch execution.",
                        "source_section": "Section 3 (Routing implementation)"
                    },
                    {
                        "name": "Sublayer Execution (Self-Attention or MLP)",
                        "formula_katex": "\\tilde{H} = f(\\tilde{X}), \\quad f \\in \\{\\text{SelfAttention}, \\text{MLP}\\}",
                        "explanation": "Executes standard Transformer self-attention or feed-forward operations exclusively on the compacted tensor.",
                        "source_section": "Section 3 (Implementing Mixture-of-Depths Transformers)"
                    },
                    {
                        "name": "Scatter & Pure Residual Skip",
                        "formula_katex": "x_i^{(l+1)} = \\begin{cases} x_i^{(l)} + r_i \\cdot \\tilde{h}_i & \\text{if } i \\in \\mathcal{S} \\\\ x_i^{(l)} & \\text{if } i \\notin \\mathcal{S} \\end{cases}",
                        "explanation": "Scatters computed token updates back into the full sequence while unselected tokens bypass the block completely via identity residual skip.",
                        "source_section": "Section 3 (Routing implementation)"
                    }
                ]
            },
            {
                "layer_name": "Auxiliary Router Loss & Alternative Predictors",
                "sublayers": [
                    {
                        "name": "Auxiliary Routing Loss",
                        "formula_katex": "\\mathcal{L}_{router} = \\text{BCE}(\\sigma(r_i), y_{\\text{target}})",
                        "explanation": "Optional auxiliary loss to stabilize router decisions when using learned routing thresholds rather than top-k selection.",
                        "source_section": "Section 3 (Learned routing thresholds)"
                    },
                    {
                        "name": "Compound MoD + MoE Hierarchical Routing",
                        "formula_katex": "x_{out} = \\text{MoE}(\\text{MoD}(X))",
                        "explanation": "Jointly routes tokens along depth (MoD) and width (MoE) to simultaneously optimize compute per token and parameter capacity.",
                        "source_section": "Section 3 (Compound Mixture-of-Depths and Mixture-of-Experts)"
                    }
                ]
            }
        ],
        "pros": [
            { "claim": "Achieves identical or superior log-perplexity to vanilla Transformers while consuming 50% fewer FLOPs per forward pass.", "source_section": "Section 1 & Section 4 (Results)" },
            { "claim": "Enables 2x faster step execution during autoregressive inference when skipping attention/MLP layers for unselected tokens.", "source_section": "Section 4.3 (Inference Speed)" },
            { "claim": "Predictable static compute tensor shapes (B, C, D) avoid dynamic shape compilation overhead on TPUs/GPUs.", "source_section": "Section 3 (Capacity)" }
        ],
        "cons": [
            { "claim": "Tokens that skip attention layers cannot attend to surrounding tokens at that layer depth, potentially limiting multi-hop reasoning if under-allocated.", "source_section": "Section 5 (Discussion)" },
            { "claim": "Top-k capacity selection introduces non-causal dependency across the sequence during training unless causal router masks are applied.", "source_section": "Section 3 (Routing implementation)" }
        ],
        "reported_metrics": [
            { "task": "Pre-training Validation Perplexity (IsoFLOP comparison)", "metric_name": "Loss Reduction", "score_value": "-0.05 nats", "source_section": "Section 4.1 (Training efficiency)", "comparison_context": "MoD outperforms baseline Transformer at same training compute budget" },
            { "task": "Inference Compute Savings", "metric_name": "FLOPs Reduction", "score_value": "50.0%", "source_section": "Section 4 (Results, Figure 2)", "comparison_context": "Maintains baseline accuracy with half the compute FLOPs per token" }
        ]
    },
    {
        "paper_id": "moe_survey",
        "title": "A Comprehensive Survey of Mixture of Experts: Algorithms, Theory, and Applications",
        "authors_venue_year": "Siyuan Mu & Sen Lin, Survey Paper 2024",
        "core_breakthrough": "Provides a comprehensive architectural taxonomy of Mixture-of-Experts (MoE) models, detailing gating algorithms, fine-grained/shared expert designs, load balancing losses, and distributed expert parallelism.",
        "computed_metric_role": "moe",
        "layers": [
            {
                "layer_name": "Gating & Routing Network Taxonomy Layer",
                "sublayers": [
                    {
                        "name": "Top-K Softmax Sparse Gating",
                        "formula_katex": "G(x)_i = \\text{softmax}(\\text{TopK}(W_g x + \\epsilon, k))_i, \\quad \\text{TopK}(v, k)_i = \\begin{cases} v_i & \\text{if } v_i \\in \\text{top } k \\\\ -\\infty & \\text{otherwise} \\end{cases}",
                        "explanation": "Calculates routing probabilities and sparsely dispatches each token to only the top-k highest scoring expert networks (e.g. k=1 in Switch Transformer, k=2 in GShard/Mixtral).",
                        "source_section": "Section II.A (Gating Function, Equation 1-2)"
                    },
                    {
                        "name": "Expert Choice Routing (EC)",
                        "formula_katex": "S_{i, j} = (X W_g)_{i, j}, \\quad \\mathcal{T}_j = \\text{TopK}(\\{S_{i, j}\\}_{i=1}^T, C_e)",
                        "explanation": "Inverts routing logic so that each expert selects its top C_e tokens, guaranteeing perfectly balanced workload across all experts and eliminating token dropping.",
                        "source_section": "Section III.A (Routing Strategy - Expert Choice)"
                    },
                    {
                        "name": "DeepSeekMoE Fine-Grained & Shared Expert Architecture",
                        "formula_katex": "y = \\sum_{i=1}^{N_s} E_i^s(x) + \\sum_{j \\in \\text{TopK}} g_j E_j^r(x)",
                        "explanation": "Splits standard experts into N fine-grained smaller experts and isolates N_s dedicated shared experts that are always activated to capture common knowledge.",
                        "source_section": "Section III.A (DeepSeekMoE design, Page 7)"
                    },
                    {
                        "name": "Soft MoE & Slot-Based Dispatch",
                        "formula_katex": "\\tilde{X} = \\text{softmax}(X W_{\\Phi}) \\cdot X, \\quad Y = \\text{softmax}(\\tilde{X} W_{\\Psi}) \\cdot E(\\tilde{X})",
                        "explanation": "Passes soft continuous linear combinations of tokens into expert slots, achieving fully differentiable expert routing without discrete top-k selection.",
                        "source_section": "Section II.A (Soft MoE, Page 5)"
                    }
                ]
            },
            {
                "layer_name": "Expert Subnetwork Topologies",
                "sublayers": [
                    {
                        "name": "Feed-Forward Expert Subnetwork (SwiGLU / GeLU)",
                        "formula_katex": "E_i(x) = (\\text{Swish}(x W_i^{gate}) \\odot x W_i^{up}) W_i^{down}",
                        "explanation": "Individual expert modules structured as independent feed-forward networks (FFNs), typically employing modern SwiGLU activations.",
                        "source_section": "Section II.B (Expert Networks)"
                    },
                    {
                        "name": "Multi-Head Latent Experts (MHLE) & Hierarchical MoE",
                        "formula_katex": "y = \\sum_{m=1}^M G_m^{(1)}(x) \\sum_{n=1}^N G_{m, n}^{(2)}(x) E_{m, n}(x)",
                        "explanation": "Two-level hierarchical tree-based gating that first selects expert clusters, then routes to specific specialized experts within that cluster.",
                        "source_section": "Section II.B & Section III.A (Hierarchical MoE)"
                    }
                ]
            },
            {
                "layer_name": "Load Balancing & Auxiliary Loss Formulations",
                "sublayers": [
                    {
                        "name": "Switch Load Balancing Auxiliary Loss",
                        "formula_katex": "\\mathcal{L}_{aux} = \\alpha \\cdot N \\sum_{i=1}^N f_i P_i, \\quad f_i = \\frac{1}{T}\\sum_{t=1}^T \\mathbb{I}(\\text{argmax}(G(x_t))=i), \\; P_i = \\frac{1}{T}\\sum_{t=1}^T G(x_t)_i",
                        "explanation": "Encourages uniform distribution of tokens across experts by penalizing the dot product between actual fraction of tokens dispatched f_i and average routing probability P_i.",
                        "source_section": "Section II.C (Loss Function, Equation 10-11)"
                    },
                    {
                        "name": "Router Z-Loss (Numerical Stability)",
                        "formula_katex": "\\mathcal{L}_z = \\frac{\\beta}{B} \\sum_{b=1}^B \\left(\\log \\sum_{i=1}^N e^{h_{b, i}}\\right)^2",
                        "explanation": "Penalizes large router pre-softmax logits to prevent floating-point rounding errors and numerical instability during large-scale distributed training.",
                        "source_section": "Section II.C (Loss Function, Equation 12)"
                    }
                ]
            },
            {
                "layer_name": "Distributed Expert Parallelism & System Infrastructure",
                "sublayers": [
                    {
                        "name": "All-to-All Token Dispatch & Combine Collectives",
                        "formula_katex": "\\text{Tokens}_{local} \\xrightarrow{\\text{All-to-All}} \\text{Tokens}_{expert-GPU} \\xrightarrow{\\text{Compute}} \\xrightarrow{\\text{All-to-All}} \\text{Tokens}_{restored}",
                        "explanation": "Coordinates inter-GPU communication primitives: tokens on worker GPUs are dispatched across network interconnect to expert host GPUs and gathered back.",
                        "source_section": "Section II.D (System Designs & Expert Parallelism)"
                    }
                ]
            }
        ],
        "pros": [
            { "claim": "Decouples model parameter capacity from computational cost per token, enabling models with 10x-100x more parameters at constant FLOPs.", "source_section": "Section I (Introduction)" },
            { "claim": "Fine-grained expert routing allows domain specialization where different experts master code, math, syntax, or knowledge subsets.", "source_section": "Section III.A & Section V (Applications)" }
        ],
        "cons": [
            { "claim": "Heavy All-to-All inter-device communication overhead creates network bandwidth bottlenecks during distributed GPU training.", "source_section": "Section II.D & Section VI (Challenges)" },
            { "claim": "Susceptible to routing collapse (expert starvation/overload) and requires complex hyperparameter tuning of auxiliary balancing losses.", "source_section": "Section II.C & Section VI" }
        ],
        "reported_metrics": [
            { "task": "Parameter Scaling Efficiency", "metric_name": "Active FLOP Efficiency", "score_value": "4x-8x", "source_section": "Section V (Applications, Table summarizing MoE models)", "comparison_context": "MoE models match dense model quality with 25% of active compute per token" },
            { "task": "DeepSeekMoE Knowledge Specialization", "metric_name": "Expert Redundancy Reduction", "score_value": "Significant", "source_section": "Section III.A (DeepSeekMoE)", "comparison_context": "Shared experts capture 90%+ common patterns while routed experts specialize" }
        ]
    },
    {
        "paper_id": "swat",
        "title": "Sliding Window Attention Training for Efficient Large Language Models",
        "authors_venue_year": "Zichuan Fu et al. (Tencent YouTu & CityU HK), arXiv 2025",
        "core_breakthrough": "Proposes SWAT, an efficient pre-training framework that replaces Softmax with Sigmoid attention and unifies balanced ALiBi + RoPE embeddings to eliminate attention sinks and enable stable sliding window training on long contexts.",
        "computed_metric_role": "sliding_window",
        "layers": [
            {
                "layer_name": "Sigmoid Attention Activation Engine",
                "sublayers": [
                    {
                        "name": "Independent Sigmoid Attention Scoring",
                        "formula_katex": "\\text{Attention}_{\\text{Sigmoid}}(Q, K, V) = \\text{Sigmoid}\\left(\\frac{QK^T}{\\sqrt{d_k}} + B\\right) V",
                        "explanation": "Replaces the global softmax normalization across tokens with an element-wise independent Sigmoid activation, decoupling token score sums and preventing the attention sink phenomenon.",
                        "source_section": "Section 3.1 (Sigmoid Attention)"
                    },
                    {
                        "name": "Attention Sink Elimination",
                        "formula_katex": "\\sigma(s_{ij}) = \\frac{1}{1 + e^{-s_{ij}}} \\in (0, 1)",
                        "explanation": "Because each token score is computed independently without a denominator sum, initial tokens no longer receive disproportionate attention mass simply to soak up unused probability.",
                        "source_section": "Section 2 (Understanding Transformer's Attention) & Section 3.1"
                    }
                ]
            },
            {
                "layer_name": "Dual Positional Hybridization (Balanced ALiBi + RoPE)",
                "sublayers": [
                    {
                        "name": "Rotary Position Embedding (RoPE) for Local Context",
                        "formula_katex": "q'_m = R_{\\Theta, m}^d q_m, \\quad k'_n = R_{\\Theta, n}^d k_n",
                        "explanation": "Applies rotary embedding to query and key vectors to encode high-precision relative positional information within the local sliding window.",
                        "source_section": "Section 3.2 (Positional Hybridization)"
                    },
                    {
                        "name": "Balanced ALiBi Linear Decay Bias",
                        "formula_katex": "B_{i, j} = -m_h \\cdot |i - j|, \\quad m_h = 2^{-\\frac{8h}{H}}",
                        "explanation": "Adds a geometric negative linear bias to attention scores based on distance, stabilizing attention decay and preventing sharp boundary artifacts at sliding window edges.",
                        "source_section": "Section 3.2 (Equation 6-7)"
                    }
                ]
            },
            {
                "layer_name": "Progressive Window Scheduling & Layer Stacking",
                "sublayers": [
                    {
                        "name": "Multi-Scale Layer Receptive Field Expansion",
                        "formula_katex": "\\text{ReceptiveField}(l) = l \\cdot (W - 1) + 1",
                        "explanation": "By stacking L layers each with sliding window size W, the effective receptive field grows linearly across network depth, achieving global sequence visibility at top layers.",
                        "source_section": "Section 3.3 (Progressive Window Scheduling, Figure 4)"
                    }
                ]
            },
            {
                "layer_name": "Linear Cache Management & SWAT Inference Pipeline",
                "sublayers": [
                    {
                        "name": "Constant-Size Sliding KV Ring-Buffer",
                        "formula_katex": "\\text{Memory}_{KV} = 2 \\cdot B \\cdot W \\cdot D \\cdot N_{layers} = O(W \\cdot D)",
                        "explanation": "Maintains a fixed-size ring buffer for Key and Value vectors of size W tokens, eliminating linear KV cache memory expansion during long-sequence generation.",
                        "source_section": "Section 3.4 (Inference and Memory Optimization)"
                    }
                ]
            }
        ],
        "pros": [
            { "claim": "Maintains constant memory footprint O(W * D) per layer regardless of sequence length N, preventing out-of-memory errors on long sequences.", "source_section": "Section 1 & Section 3.4" },
            { "claim": "Achieves training throughput improvements of 4x-8x over full attention when pre-training on sequences up to 32k tokens.", "source_section": "Section 4.3 (Efficiency Analysis)" },
            { "claim": "Sigmoid attention eliminates attention sink artifacts and prevents perplexity degradation when evaluating on extended sequence lengths.", "source_section": "Section 3.1 & Figure 2" }
        ],
        "cons": [
            { "claim": "Maximum single-layer direct attention distance is strictly bounded by window size W.", "source_section": "Section 5 (Limitations)" },
            { "claim": "Requires careful tuning of ALiBi slope ratios m_h to prevent over-decaying long-range semantic information in early layers.", "source_section": "Section 3.2" }
        ],
        "reported_metrics": [
            { "task": "PIQA Physical Commonsense Reasoning", "metric_name": "Accuracy", "score_value": "77.2%", "source_section": "Table 1 (Overall Comparison on Reasoning Tasks)", "comparison_context": "SWAT matches Full-Attention baseline (77.4%) with 50% window size" },
            { "task": "HellaSwag Commonsense Reasoning", "metric_name": "Accuracy", "score_value": "73.1%", "source_section": "Table 1 (Reasoning Benchmarks)", "comparison_context": "Competitive with standard LLaMA-style dense attention (73.5%)" },
            { "task": "ARC-Challenge Benchmark", "metric_name": "Accuracy", "score_value": "48.9%", "source_section": "Table 1 (Reasoning Tasks)", "comparison_context": "Exceeds Mistral-SWA baseline (46.8%) due to Sigmoid attention stabilization" }
        ]
    },
    {
        "paper_id": "titans",
        "title": "Titans: Learning to Memorize at Test Time",
        "authors_venue_year": "Ali Behrouz, Peilin Zhong, Vahab Mirrokni (Google Research), arXiv 2024/2025",
        "core_breakthrough": "Introduces Neural Long-Term Memory (NMM) updated at test-time via gradient descent on an associative surprise loss with surprise momentum and data-dependent decay, coupled with short-term sliding window attention across MAC, MAG, and MAL architectures.",
        "computed_metric_role": "neural_memory",
        "layers": [
            {
                "layer_name": "Associative Memory Loss & Surprise Gradient Engine",
                "sublayers": [
                    {
                        "name": "Memory Key-Value Association Objective",
                        "formula_katex": "\\mathcal{L}_{mem}(M_t; x_t) = \\| M_t(k_t) - v_t \\|_2^2, \\quad k_t = x_t W_K, \\; v_t = x_t W_V",
                        "explanation": "Defines associative memory reconstruction loss where neural memory network M_t is trained to predict value vector v_t given key vector k_t.",
                        "source_section": "Section 3.1 (Associative Memory Loss, Equation 1)"
                    },
                    {
                        "name": "Surprise Gradient Calculation",
                        "formula_katex": "g_t = \\nabla_{M_t} \\mathcal{L}_{mem}(M_t; x_t) = 2 (M_t(k_t) - v_t) \\nabla_{M_t} M_t(k_t)",
                        "explanation": "Calculates the instantaneous surprise of an input token as the gradient of the memory loss with respect to the neural memory weights.",
                        "source_section": "Section 3.1 (Surprise as Gradient, Equation 3)"
                    }
                ]
            },
            {
                "layer_name": "Surprise Momentum & Dynamic Decay Engine",
                "sublayers": [
                    {
                        "name": "Surprise Momentum Accumulator",
                        "formula_katex": "S_t = \\eta_t S_{t-1} - \\theta_t g_t, \\quad \\eta_t = \\text{sigmoid}(x_t W_\\eta), \\; \\theta_t = \\text{softplus}(x_t W_\\theta)",
                        "explanation": "Maintains a momentum term S_t that acts as a temporal memory of surprise across sequence history, modulated by data-dependent learning rate theta_t and momentum decay eta_t.",
                        "source_section": "Section 3.2 (Momentum and Surprise Decay, Equation 10)"
                    },
                    {
                        "name": "Memory Weight Update & Forgetting",
                        "formula_katex": "M_t = (1 - \\alpha_t) M_{t-1} + S_t, \\quad \\alpha_t = \\text{sigmoid}(x_t W_\\alpha)",
                        "explanation": "Updates the neural memory weights via generalized forgetting (weight decay alpha_t) and integrated surprise momentum S_t.",
                        "source_section": "Section 3.2 (Memory Update Rule, Equation 8)"
                    }
                ]
            },
            {
                "layer_name": "Deep Neural Long-Term Memory (NMM) Structure",
                "sublayers": [
                    {
                        "name": "Deep Multi-Layer Perceptron Memory",
                        "formula_katex": "M_t(k) = W_t^{(2)} \\sigma(W_t^{(1)} k + b^{(1)}) + b^{(2)}",
                        "explanation": "Implements the memory as deep non-linear MLP weight matrices rather than static vector slots, massively increasing memory storage capacity.",
                        "source_section": "Section 3.3 (Deep Neural Memory)"
                    },
                    {
                        "name": "Parallel Chunked Training Algorithm",
                        "formula_katex": "M_c = M_{c-1} \\prod_{t \\in B_c} (1 - \\alpha_t) + \\sum_{t \\in B_c} \\tilde{S}_t",
                        "explanation": "Formulates mini-batch test-time gradient updates as associative parallel matrix multiplications for fast GPU hardware execution.",
                        "source_section": "Section 3.4 (Parallel Training with Matmuls, Figure 1)"
                    }
                ]
            },
            {
                "layer_name": "Titans Hyper-Head Architectural Topologies",
                "sublayers": [
                    {
                        "name": "MAC (Memory as Context) Topology",
                        "formula_katex": "\\tilde{X} = [P \\;\\|\\; M_t(X_{hist}) \\;\\|\\; X], \\quad Y = \\text{SW-Attention}(\\tilde{X})",
                        "explanation": "Retrieves memory context tokens, concatenates them with task-level persistent memory prefix P and input tokens, and feeds them into short-term sliding window attention.",
                        "source_section": "Section 4.1 (Memory as a Context, Figure 2)"
                    },
                    {
                        "name": "MAG (Memory as Gate) Topology",
                        "formula_katex": "Y = g_t \\odot Y_{\\text{attn}} + (1 - g_t) \\odot Y_{\\text{mem}}, \\quad g_t = \\text{sigmoid}(\\text{Linear}([Y_{\\text{attn}}, Y_{\\text{mem}}]))",
                        "explanation": "Processes input in parallel through short-term attention and neural memory, combining their outputs via a learned dynamic gate g_t.",
                        "source_section": "Section 4.2 (Gated Memory, Figure 4)"
                    },
                    {
                        "name": "MAL (Memory as Layer) Topology",
                        "formula_katex": "Y = \\text{SW-Attention}(M(X))",
                        "explanation": "Stacks neural memory layers sequentially with sliding-window attention layers as modular building blocks.",
                        "source_section": "Section 4.3 (Memory as a Layer, Figure 5)"
                    }
                ]
            }
        ],
        "pros": [
            { "claim": "Can memorize and retrieve information across sequences of 2M+ tokens with linear computational scaling and fast inference.", "source_section": "Section 1 & Section 5 (Experiments)" },
            { "claim": "Surprise-driven update prevents memory overflow by prioritizing memorable, unexpected tokens over redundant background text.", "source_section": "Section 3.1 & Section 3.2" },
            { "claim": "Test-time learning allows the model to continuously adapt and update its internal weights during generation without pre-training updates.", "source_section": "Section 1 (Introduction)" }
        ],
        "cons": [
            { "claim": "Test-time gradient computation requires backpropagation operations during the forward inference pass, adding computational overhead per token.", "source_section": "Section 3.1 & Section 5" },
            { "claim": "More complex mathematical formulation and implementation than standard linear attention models.", "source_section": "Section 3" }
        ],
        "reported_metrics": [
            { "task": "BABILong 16K-128K Context Needle Benchmark", "metric_name": "Accuracy", "score_value": ">90.0%", "source_section": "Figure 6 (Performance on BABILong benchmark)", "comparison_context": "Titans (MAC) significantly outperforms Mamba, RWKV, and Recurrent baselines (<50%)" },
            { "task": "RULER S-NIAH Synthetic Recall Benchmark", "metric_name": "Accuracy", "score_value": "98.4%", "source_section": "Table 2 (Performance on S-NIAH task)", "comparison_context": "Matches full attention Transformer while maintaining linear scaling" },
            { "task": "Long-Term Time-Series Forecasting (Weather)", "metric_name": "Mean Squared Error", "score_value": "0.142 MSE", "source_section": "Table 3 (Performance on long-term forecasting)", "comparison_context": "Outperforms PatchTST (0.149) and DLinear (0.176)" }
        ]
    },
    {
        "paper_id": "transmamba",
        "title": "TransMamba: A Sequence-Level Hybrid Transformer-Mamba Language Model",
        "authors_venue_year": "Yixing Li, Ruobing Xie et al. (Tencent Hunyuan & CUHK), AAAI 2026",
        "core_breakthrough": "Unifies Transformer and Mamba at the sequence level through shared parameter projection matrices (QKV <-> CBx) and Memory Compression Cache, dynamically switching between Attention and SSM at TransPoints.",
        "computed_metric_role": "hybrid",
        "layers": [
            {
                "layer_name": "Shared Parameter Projection Layer (W_proj)",
                "sublayers": [
                    {
                        "name": "Unified Matrix Weight Sharing",
                        "formula_katex": "\\begin{bmatrix} Q_t \\\\ K_t \\\\ V_t \\end{bmatrix} \\longleftrightarrow \\begin{bmatrix} C_t \\\\ B_t \\\\ x_t \\end{bmatrix} = x_t W_{proj}, \\quad W_{proj} \\in \\mathbb{R}^{D \\times 3D}",
                        "explanation": "Shares a single parameter projection matrix W_proj to generate QKV for Transformer attention or CBx for Mamba SSM, enabling zero-parameter overhead switching.",
                        "source_section": "Section 2.1.3 & Section 2.2 (Main Architecture, Figure 3a)"
                    }
                ]
            },
            {
                "layer_name": "Sequence-Level Dynamic Routing & TransPoint Scheduling",
                "sublayers": [
                    {
                        "name": "TransPoint Node Partitioning",
                        "formula_katex": "\\text{Mode}(t) = \\begin{cases} \\text{Transformer (Attention)} & \\text{if } t \\le N_{trans} \\\\ \\text{Mamba (SSM)} & \\text{if } t > N_{trans} \\end{cases}",
                        "explanation": "Calculates initial sequence tokens using full Transformer attention for rich prompt reasoning, then transitions at TransPoint threshold to linear Mamba2 SSM.",
                        "source_section": "Section 2.2 (Formalized calculation process) & Section 2.3"
                    },
                    {
                        "name": "Layer-Specific TransPoint Scheduling",
                        "formula_katex": "N_{trans}^{(l)} = \\text{Schedule}(l), \\quad l \\in [1, N_{layers}]",
                        "explanation": "Assigns different TransPoints across layer depths (e.g. earlier layers transition sooner while deep reasoning layers retain longer attention spans).",
                        "source_section": "Section 2.3 (TransPoint Schedule) & Table 6"
                    }
                ]
            },
            {
                "layer_name": "Memory Compression Cache (MCC) & State Bridge",
                "sublayers": [
                    {
                        "name": "Lossless Attention-to-SSM State Transfer",
                        "formula_katex": "h_{ssm}(N_{trans}) = \\text{MCC}(Y_{\\text{attn}}^{1:N_{trans}}, K^{1:N_{trans}}, V^{1:N_{trans}})",
                        "explanation": "Converts historical Transformer KV cache and attention outputs into the initial continuous recurrent hidden state for downstream Mamba computation.",
                        "source_section": "Section 2.4 (Memory Compression Cache)"
                    }
                ]
            },
            {
                "layer_name": "Hybrid Layer-Stacking & Multi-Task Generalization Engine",
                "sublayers": [
                    {
                        "name": "Autoregressive Hybrid Decoder Block Stack",
                        "formula_katex": "x^{(l+1)} = x^{(l)} + \\text{TransMambaLayer}(x^{(l)})",
                        "explanation": "Executes stacked hybrid decoder layers combining shared parameter transformations, SSM recurrence, and multi-task generalization capabilities.",
                        "source_section": "Section 2.2 & Section 2.5"
                    }
                ]
            }
        ],
        "pros": [
            { "claim": "Achieves up to 40% reduction in relative training time compared to vanilla Transformer at identical model size.", "source_section": "Section 1 & Figure 2, Table 5" },
            { "claim": "Zero parameter overhead from dual mechanisms due to shared QKV and CBx projection matrices.", "source_section": "Section 2.1.3 & Figure 3a" },
            { "claim": "Maintains superior in-context learning and multi-task reasoning capabilities where pure Mamba struggles.", "source_section": "Section 1 & Table 3" }
        ],
        "cons": [
            { "claim": "Performance is sensitive to TransPoint scheduling strategy; sub-optimal transition points degrade in-context learning.", "source_section": "Section 2.3 & Table 6" },
            { "claim": "KV cache memory must still be maintained during the initial sequence portion before the TransPoint.", "source_section": "Section 2.2" }
        ],
        "reported_metrics": [
            { "task": "CoQA Conversational In-Context Reasoning", "metric_name": "F1 Score", "score_value": "63.8 F1", "source_section": "Table 3 (Main Evaluation Results)", "comparison_context": "Outperforms pure Mamba2 (58.4 F1) and matches Transformer (64.1 F1)" },
            { "task": "ARC-Challenge Common-Sense Benchmark", "metric_name": "Accuracy", "score_value": "39.4%", "source_section": "Table 3 (Evaluation Results)", "comparison_context": "Higher than Mamba2 (36.1%) and Layer-Hybrid (37.8%)" },
            { "task": "Training Speedup vs Transformer Baseline", "metric_name": "Training Time Reduction", "score_value": "38.5%", "source_section": "Table 5 (Comparison of average training time)", "comparison_context": "Significantly faster training with equivalent evaluation loss" }
        ]
    }
]

# Validation function
def validate_dataset(data):
    print("Validating dataset against schema...")
    assert len(data) == 7, f"Expected 7 papers, got {len(data)}"
    
    total_layers = 0
    total_sublayers = 0
    total_pros = 0
    total_cons = 0
    total_metrics = 0
    
    for p in data:
        # Check required top-level fields
        for field in SCHEMA["required_paper_fields"]:
            assert field in p and p[field], f"Missing or empty field '{field}' in paper '{p.get('paper_id')}'"
        
        # Check layers
        assert isinstance(p["layers"], list) and len(p["layers"]) >= 3, f"Paper '{p['paper_id']}' has too few layers"
        for layer in p["layers"]:
            total_layers += 1
            for l_field in SCHEMA["required_layer_fields"]:
                assert l_field in layer and layer[l_field], f"Missing field '{l_field}' in layer '{layer.get('layer_name')}'"
            
            # Check sublayers
            assert isinstance(layer["sublayers"], list) and len(layer["sublayers"]) >= 1
            for sub in layer["sublayers"]:
                total_sublayers += 1
                for s_field in SCHEMA["required_sublayer_fields"]:
                    if s_field == "formula_katex":
                        continue # can be string or null
                    assert s_field in sub and sub[s_field], f"Missing '{s_field}' in sublayer '{sub.get('name')}'"
                # Check citation
                assert len(sub["source_section"].strip()) > 3, f"Invalid source_section in {sub['name']}"
        
        # Check pros
        for pro in p["pros"]:
            total_pros += 1
            assert pro.get("claim") and pro.get("source_section"), f"Invalid pro entry in {p['paper_id']}"
            assert len(pro["source_section"].strip()) > 3, f"Missing pro citation in {p['paper_id']}"
        
        # Check cons
        for con in p["cons"]:
            total_cons += 1
            assert con.get("claim") and con.get("source_section"), f"Invalid con entry in {p['paper_id']}"
            assert len(con["source_section"].strip()) > 3, f"Missing con citation in {p['paper_id']}"
            
        # Check reported metrics
        for met in p["reported_metrics"]:
            total_metrics += 1
            for m_field in SCHEMA["required_reported_metric_fields"]:
                assert m_field in met and met[m_field], f"Missing '{m_field}' in reported metric in {p['paper_id']}"
    
    print(f"Validation PASSED 100%!")
    print(f"- Total Papers: {len(data)}")
    print(f"- Total Layers: {total_layers}")
    print(f"- Total Sublayers: {total_sublayers}")
    print(f"- Total Pro Claims: {total_pros}")
    print(f"- Total Con Claims: {total_cons}")
    print(f"- Total Reported Metrics: {total_metrics}")

validate_dataset(papers_data)

out_file = r'a:\Downloads\24th OITS\understanding\papers_data.json'
with open(out_file, 'w', encoding='utf-8') as f:
    json.dump(papers_data, f, indent=2)
print(f"Saved verified dataset to: {out_file}")
