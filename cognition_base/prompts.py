# Field Descriptions
FIELD_DESCRIPTIONS = {
    "DESIGN_INSIGHT": """
    High-level architectural insights and core innovations from the paper, 
    formatted as priority-leveled sections (HIGH/MEDIUM/LOW) describing the main 
    technical contributions. Each insight should capture a fundamental architectural 
    breakthrough with precise mathematical notation and clear differentiation from prior work.
    """,

    "EXPERIMENTAL_TRIGGER_PATTERNS": """
    Observable performance signatures and architectural symptoms that indicate 
    when the technique is working effectively. Structured as Task_Performance_Signatures 
    (specific benchmark improvements with quantitative expectations) and 
    Architectural_Symptoms (system-level behavioral indicators during training/inference).
    """,

    "BACKGROUND": """
    Comprehensive structured contextual information including exact paper title, 
    detailed historical technical context, specific technical limitations addressed, 
    mathematically precise key concepts with LaTeX notation, and thorough experimental 
    context. Provides complete foundation for understanding the contribution's significance.
    """,

    "ALGORITHMIC_INNOVATION": """
    Detailed technical specification of the core algorithmic contribution, including 
    the precise computational steps, underlying mechanism explanation, comprehensive 
    mathematical formulation with LaTeX notation, and complete computational properties 
    analysis (complexity, parallelization, memory, stability).
    """,

    "IMPLEMENTATION_GUIDANCE": """
    Comprehensive practical implementation advice with specific integration strategies, 
    detailed parameter settings with numerical ranges, precise application conditions, 
    and realistic outcome expectations including potential failure modes and 
    hardware-specific recommendations.
    """
}

# Enhanced Metaprompts for each field
METAPROMPTS = {
    "DESIGN_INSIGHT": """
Extract the core architectural innovations from this research paper and format them as DESIGN_INSIGHT entries. Follow these precise guidelines:

**Priority Classification:**
- Use "### DESIGN_INSIGHT_HIGH:" for breakthrough architectural concepts that fundamentally change how computation is performed
- Use "### DESIGN_INSIGHT_MEDIUM:" for significant improvements or novel combinations of existing techniques  
- Use "### DESIGN_INSIGHT_LOW:" for incremental but valuable refinements

**Title Format:**
- Create descriptive titles in brackets: "[Specific Technique Name – Brief Description of Core Innovation]"
- Include mathematical notation where relevant: "[LRPE-d with λ^{s-t} Decay for Linear O(n) Attention]"
- Be specific about the algorithmic contribution, not just the application domain

**Content Structure:**
1. Start with 1-2 sentences explaining what gets replaced or modified in existing architectures
2. Describe the key mechanism that enables the improvement  
3. Highlight the fundamental difference from prior approaches
4. If applicable, mention specific mathematical formulations or complexity improvements

**Mathematical Notation:**
- Use LaTeX notation: \\( \\) for inline math, \\[ \\] for display equations
- Include key equations when they define the core innovation
- Specify complexity improvements (O(n²) → O(n), etc.)

**Examples of Quality Titles:**
- [Linearized Relative Positional Encoding with Exponential Decay (LRPE-d) for Linear Attention in LLMs]
- [Multi-Headed Matrix-Valued States with Per-Head LayerNorm and Gating (Eagle/RWKV-5)]
- [4-bit NormalFloat (NF4) Quantization for High-Fidelity Low-Bit Storage]
- [Lightning Attention – IO-Aware, Blockwise Linear Attention for Efficient Training and Inference]

**Content Length:** Each insight should be 2-4 paragraphs, providing sufficient technical depth while remaining focused on the architectural breakthrough.

Generate 1-3 insights depending on the paper's scope. Prioritize architectural innovations over implementation details or experimental results.
""",

    "EXPERIMENTAL_TRIGGER_PATTERNS": """
Analyze the paper's experimental results and create EXPERIMENTAL_TRIGGER_PATTERNS with two precise sections:

**Task_Performance_Signatures:**
Provide specific, quantitative expectations for benchmark performance. Include:

- **Specific Benchmark Names:** Use exact task names from the literature:
  * Language modeling: lambada_openai, wikitext, ptb
  * Reading comprehension: squad_completion, squad_v2, narrativeqa
  * Commonsense reasoning: hellaswag, piqa, social_iqa, commonsenseqa
  * Factual QA: arc_easy, arc_challenge, boolq, openbookqa
  * Context resolution: winogrande, winograd
  * Other tasks: swde (structured extraction), fda (data augmentation)

- **Performance Pattern Descriptions:**
  * "Expect improved/stable/degraded performance on [specific tasks]"
  * "Higher scores on [task] due to [specific capability improvement]"  
  * "Performance remains competitive on [task category] while achieving [efficiency gain]"
  * Include quantitative expectations: "2× speedup", "4× memory reduction", "similar accuracy"

- **Contextual Conditions:**
  * When improvements are most pronounced (long sequences, large models, specific domains)
  * Scale dependencies (small vs. large models, short vs. long contexts)
  * Training vs. inference differences

- **Quantitative Expectations:**
  * Use specific metrics: "training loss decreases more smoothly", "order of magnitude faster inference"
  * Mention scaling behaviors: "linear vs. quadratic scaling", "constant memory during inference"
  * Include degradation bounds where applicable: "negligible quality drops", "within 1% of baseline"

**Architectural_Symptoms:**
Describe observable system-level indicators:

- **Training Characteristics:**
  * Loss curve behaviors: "smoother convergence", "reduced variance", "faster initial convergence"
  * Stability indicators: "less likely to diverge", "robust to hyperparameter changes", "no NaN occurrences"
  * Scaling properties: "stable performance as depth/width increases"

- **Runtime/Memory Behaviors:**
  * Memory scaling: "linear vs. quadratic memory growth", "constant inference memory"
  * Throughput patterns: "consistent speed regardless of sequence length", "higher GPU utilization"
  * Hardware utilization: "no OOM errors at larger batch sizes", "efficient memory bandwidth usage"

- **Profiling Signatures:**
  * Specific metrics that indicate the technique is working
  * Hardware-specific behaviors (GPU memory patterns, CPU utilization)
  * Comparison with baseline implementations

**Format Requirements:**
- Start each section with "**Task_Performance_Signatures**:" and "**Architectural_Symptoms**:"
- Use bullet points and specific technical language
- Include both positive indicators and potential neutral/negative effects
- Be quantitatively specific wherever possible

Focus on concrete, measurable indicators that would help practitioners recognize when the technique is working as intended.
""",

    "BACKGROUND": """
Create a comprehensive BACKGROUND section with the following precise structure:

**Title:** [Extract the exact paper title verbatim]

**Historical Technical Context:**
Provide 2-3 paragraphs describing:
- The dominant architectures and approaches before this work (RNNs, LSTMs, Transformers, CNNs, etc.)
- Key technological developments that preceded and motivated this research
- Evolution of the problem space and previous solution attempts
- Specific architectural paradigms that were standard in the field

**Technical Limitations:**
Identify 4-6 specific problems this paper addresses:
- Computational bottlenecks with precise complexity analysis (e.g., "O(n²) attention complexity")
- Memory limitations and scaling issues
- Accuracy or generalization problems with previous approaches
- Deployment or hardware efficiency constraints
- Approximation errors or instability issues in prior methods
- Training dynamics or optimization challenges

**Paper Concepts:**
Define 4-6 key technical terms using this exact format:
- **Term Name**: Comprehensive definition with mathematical context and LaTeX notation where appropriate
- Focus on novel concepts introduced by this paper and crucial background concepts
- Include mathematical definitions: \\( equation \\) or \\[ display equation \\]
- Provide intuitive explanation alongside mathematical precision
- Connect concepts to their role in the overall contribution

**Example Format:**
- **Linear Attention:** An attention mechanism where computation scales linearly with sequence length by decomposing softmax attention, often approximated as \\( QK^\\top V \\) without explicit normalization.
- **LRPE-d (Linearized Relative Positional Encoding with exponential decay):** A position encoding scheme combining learnable phase and exponential decay, formulated as \\( a_{st} = q_s^\\top k_t \\lambda^{s-t} \\exp(i\\theta(s-t)) \\), to maintain global context and prevent attention dilution.

**Experimental Context:**
Describe the evaluation philosophy in 1-2 paragraphs:
- Types of tasks emphasized in evaluation (generative, discriminative, reasoning, efficiency)
- Metrics and benchmarks that are prioritized
- Overall experimental goals and what the paper aims to demonstrate
- Evaluation philosophy (accuracy vs. efficiency tradeoffs, scalability focus, etc.)
- Performance measurement approach (zero-shot, few-shot, fine-tuning, etc.)

**Quality Requirements:**
- Use precise technical language with appropriate mathematical notation
- Ensure sufficient depth for a technical reader to understand the contribution's context
- Include specific complexity bounds, performance numbers, and technical constraints where available
- Maintain focus on information directly relevant to understanding the paper's contribution
""",

    "ALGORITHMIC_INNOVATION": """
Extract and structure the technical algorithmic contributions using this precise format:

**Core_Algorithm:**
Describe the main algorithmic contribution in 3-4 sentences:
- State exactly what gets replaced or modified (e.g., "Replace softmax attention with kernel-based linear attention")
- Describe the key computational steps or transformations
- Explain the fundamental change from standard approaches
- Be specific about what operations are performed differently
- Include the scope of changes (per layer, per head, etc.)

**Key_Mechanism:**
Explain the underlying principle in 2-3 sentences:
- Describe WHY the algorithm works (the key insight or mathematical principle)
- Connect the mechanism to the observed benefits
- Explain the intuition behind why this approach is superior
- Include any important theoretical foundations or assumptions

**Mathematical_Formulation:**
Provide comprehensive mathematical specification:
- Include 3-6 key equations using LaTeX: \\( inline \\) and \\[ display \\] notation
- Show both the new method and comparisons with standard approaches where helpful
- Include parameter definitions and constraints
- Specify complexity analysis with Big O notation
- Include recurrence relations, update rules, or optimization objectives as relevant
- Define all variables and mathematical symbols used

**Example Format:**
- Attention score: \\( a_{st} = q_s^\\top k_t \\lambda^{s-t} \\exp(i\\theta(s-t)) \\)
- State update: \\( s_i = s_{i-1} + \\phi(K_i) V_i^T \\), \\( z_i = z_{i-1} + \\phi(K_i) \\)
- Complexity: O(nd²) vs. O(n²d) for standard attention

**Computational_Properties:**
Provide detailed analysis covering:
- **Time Complexity:** Precise bounds for training and inference with big O notation
- **Space Complexity:** Memory requirements and scaling behavior
- **Parallelization:** Characteristics for GPU/distributed computing, parallelizable components
- **Hardware Compatibility:** GPU memory patterns, CPU vs. GPU efficiency, memory bandwidth requirements
- **Training vs. Inference:** Different complexity or efficiency characteristics
- **Parameter Count:** Impact on model size and storage requirements
- **Numerical Stability:** Potential issues, mitigation strategies, stability guarantees
- **Scaling Behavior:** How properties change with model size, sequence length, or other dimensions

**Quality Standards:**
- Be mathematically precise while remaining implementable
- Include sufficient detail for a skilled practitioner to implement the technique
- Balance theoretical rigor with practical accessibility
- Use consistent mathematical notation throughout
- Specify any implementation-critical details or potential pitfalls
""",

    "IMPLEMENTATION_GUIDANCE": """
Create comprehensive practical implementation advice using this structure:

**Integration_Strategy:**
Provide specific, actionable integration guidance:
- Exactly which modules/components to replace or modify (e.g., "Replace all softmax attention modules", "Insert after query/key projections")
- Code-level changes required with specific function or class modifications
- Compatibility considerations with existing frameworks (PyTorch, TensorFlow, JAX)
- Migration path from existing implementations
- Dependencies on custom kernels, specialized libraries, or hardware features
- Integration with existing training pipelines and optimization procedures

**Parameter_Settings:**
Provide detailed parameter recommendations:
- **Specific hyperparameter ranges:** "λ ∈ [0.9, 0.99]", "block size: 128-2048 tokens", "learning rate: 1e-4 to 5e-4"
- **Initialization strategies:** Specific initialization schemes for new parameters with numerical values
- **Critical vs. robust parameters:** Which settings need careful tuning vs. those that can use defaults
- **Parameter relationships:** How different parameters interact and affect performance
- **Scale-dependent settings:** Different recommendations for small vs. large models
- **Hardware-dependent settings:** Optimal values for different GPU types or memory configurations

**Application_Conditions:**
Specify when to use this technique:
- **Beneficial scenarios:** Specific problem characteristics (sequence length thresholds, model size ranges, task types)
- **Hardware requirements:** Minimum memory, compute capabilities, specialized hardware needs
- **Scale considerations:** When the technique becomes advantageous (model size, dataset size, context length)
- **Task compatibility:** Which types of tasks benefit most vs. those where it's neutral or harmful
- **Alternative comparisons:** When to choose this over competing approaches
- **Resource constraints:** Memory, compute, or latency limitations that make this technique preferable

**Expected_Outcomes:**
Provide realistic expectations with specific metrics:
- **Performance improvements:** Quantitative expectations with ranges (e.g., "2-4× speedup in training", "30-50% memory reduction")
- **Timeline expectations:** Immediate benefits vs. those requiring full training cycles
- **Trade-off analysis:** What performance aspects improve vs. those that may degrade
- **Benchmark comparisons:** Expected performance on specific evaluation metrics compared to baselines
- **Failure modes:** Specific scenarios where the technique might not work or perform poorly
- **Debugging indicators:** How to recognize when the implementation is working correctly
- **Hardware-specific outcomes:** Different expectations on different hardware configurations

**Quality Requirements:**
- Include specific numerical values and ranges wherever possible
- Provide troubleshooting guidance for common implementation issues
- Include both positive outcomes and potential limitations/risks
- Be specific about hardware, software, and scale dependencies
- Focus on actionable advice that practitioners can directly apply
- Include validation procedures to ensure correct implementation
"""
}


def get_field_description(field_name):
    """Get the description for a specific field."""
    return FIELD_DESCRIPTIONS.get(field_name, "Field description not found.")


def get_metaprompt(field_name):
    """Get the metaprompt for generating a specific field."""
    return METAPROMPTS.get(field_name, "Metaprompt not found.")
