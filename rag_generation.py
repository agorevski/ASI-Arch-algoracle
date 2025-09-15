"""
RAG Generation Metaprompts for Research Paper Analysis
====================================================

This module contains metaprompts designed to generate structured JSON outputs
for research paper analysis in the cognition base format.

The pipeline processes research papers (PDF or HTML) and generates JSON files
with the following five key fields:
- DESIGN_INSIGHT
- EXPERIMENTAL_TRIGGER_PATTERNS  
- BACKGROUND
- ALGORITHMIC_INNOVATION
- IMPLEMENTATION_GUIDANCE
"""

# Field Descriptions
FIELD_DESCRIPTIONS = {
    "DESIGN_INSIGHT": """
    High-level architectural insights and core innovations from the paper, 
    typically formatted as titled sections describing the main technical contributions.
    Focus on what makes the approach fundamentally different from prior work.
    """,
    
    "EXPERIMENTAL_TRIGGER_PATTERNS": """
    Observable performance signatures and architectural symptoms that indicate 
    when the technique is working. Divided into task performance patterns 
    (benchmark improvements) and system behavior patterns (training characteristics).
    """,
    
    "BACKGROUND": """
    Structured contextual information including the paper title, historical context, 
    technical limitations being addressed, key concepts/definitions, and experimental methodology.
    Provides comprehensive context for understanding the contribution.
    """,
    
    "ALGORITHMIC_INNOVATION": """
    Technical details of the core algorithm, including the main mechanism, 
    mathematical formulation, and computational properties. The heart of the 
    technical contribution with precise algorithmic and mathematical details.
    """,
    
    "IMPLEMENTATION_GUIDANCE": """
    Practical advice for implementation including integration strategies, 
    parameter settings, application conditions, and expected outcomes. 
    Actionable guidance for practitioners wanting to use the technique.
    """
}

# Metaprompts for each field
METAPROMPTS = {
    "DESIGN_INSIGHT": """
Extract the core architectural innovations from this research paper and format them as DESIGN_INSIGHT entries. For each major contribution:

1. Create a descriptive title in brackets that captures the essence of the innovation
2. Focus on high-level architectural concepts rather than implementation details  
3. Emphasize what makes this approach fundamentally different from prior work
4. Use the format: "### DESIGN_INSIGHT_X: [Descriptive Title]"

Generate 1-3 design insights depending on the paper's scope. Each should be a concise but comprehensive summary of a key architectural contribution.

Examples of good titles:
- [Multi-Query Attention – Sharing Keys and Values Across Attention Heads for Fast Decoding]
- [Replace Attention with Toeplitz-Based Relative Position Token Mixing for Log-Linear Complexity]
- [Fast Attention via Positive Orthogonal Random Features (FAVOR+): Linear-Time, Unbiased Softmax Attention Approximation]

Focus on the fundamental architectural breakthrough rather than incremental improvements.
""",

    "EXPERIMENTAL_TRIGGER_PATTERNS": """
Analyze the paper's experimental results and create EXPERIMENTAL_TRIGGER_PATTERNS that describe:

**Task_Performance_Signatures**: 
- Specific benchmark improvements expected (mention specific task names like lambada_openai, hellaswag, squad_completion, arc_easy/challenge, boolq, piqa, social_iqa, winogrande, openbookqa, fda, swde when relevant)
- Performance patterns across different types of tasks (language modeling, reasoning, context-heavy tasks, etc.)
- Conditions under which improvements are most pronounced (long sequences, large models, etc.)
- Any task categories where performance might be neutral or degraded
- Quantitative expectations where possible (e.g., "order of magnitude faster", "negligible quality drops")

**Architectural_Symptoms**:
- Observable behaviors during training (convergence patterns, loss curves, stability, etc.)
- System-level indicators (memory usage, scaling properties, stability metrics)  
- Runtime characteristics that distinguish this approach (throughput, latency, memory bandwidth)
- Profiling signatures that indicate the technique is working

Focus on concrete, measurable indicators that would help practitioners recognize when the technique is working as intended. Be specific about which metrics improve and under what conditions.
""",

    "BACKGROUND": """
Create a structured BACKGROUND section with the following components:

**Title**: [Extract the exact paper title]

**Historical Technical Context**: Describe the state of the field before this work, including:
- Dominant architectures and approaches (RNNs, LSTMs, Transformers, etc.)
- Key prior developments that led to this work
- Evolution of the problem space and previous solutions

**Technical Limitations**: Identify the specific problems this paper addresses:
- Computational bottlenecks or inefficiencies (e.g., quadratic complexity, memory bandwidth)
- Accuracy or generalization issues
- Scalability or practical deployment challenges
- Approximation errors in previous approaches

**Paper Concepts**: Define 4-6 key technical terms as bullet points:
- Include mathematical definitions where appropriate
- Focus on concepts central to understanding the contribution
- Use format: "- **Term**: Definition with context and mathematical notation if relevant"
- Cover both novel concepts introduced and important background concepts

**Experimental Context**: Describe the evaluation philosophy and methodology:
- Types of tasks used for evaluation (generative, discriminative, reasoning, etc.)
- Metrics and benchmarks emphasized
- Overall experimental approach and goals
- What the evaluation is trying to demonstrate (efficiency, accuracy, scalability, etc.)

Ensure the background provides sufficient context for a technical reader to understand the contribution and its significance.
""",

    "ALGORITHMIC_INNOVATION": """
Extract the technical algorithmic contributions and structure them as:

**Core_Algorithm**: 
- Describe the main algorithmic contribution in 2-3 sentences
- Focus on the key computational steps or transformations
- Explain how it differs from standard approaches (e.g., "Replace X with Y", "Modify Z by...")
- Be specific about what gets computed differently

**Key_Mechanism**:
- Explain WHY the algorithm works (the underlying principle or insight)
- Describe the key insight that enables the improvement
- Connect the mechanism to the observed benefits
- Focus on the intuition behind why this approach is better

**Mathematical_Formulation**:
- Provide key equations using LaTeX notation (\\( \\) for inline, \\[ \\] for display)
- Include both the new method and contrasts with standard approaches where helpful
- Focus on the most important mathematical relationships
- Show complexity analysis where relevant (O(n²) vs O(n), etc.)
- Include parameter definitions and constraints

**Computational_Properties**:
- Time and space complexity analysis with specific bounds
- Parallelization characteristics and hardware compatibility
- Memory requirements and scaling behavior
- Training vs. inference efficiency considerations
- Parameter count implications
- Numerical stability considerations

Be mathematically precise while remaining accessible to practitioners who need to implement the technique.
""",

    "IMPLEMENTATION_GUIDANCE": """
Create practical implementation advice structured as:

**Integration_Strategy**:
- How to incorporate this technique into existing systems (e.g., replace specific modules)
- What components need to be modified or replaced
- Compatibility considerations with existing codebases
- Migration path from existing implementations
- Code-level changes required

**Parameter_Settings**:
- Specific hyperparameter recommendations with typical ranges
- Initialization strategies for new parameters
- Key parameters that need tuning vs. those that can use defaults
- Relationships between parameters and performance
- Guidelines for parameter selection based on problem characteristics

**Application_Conditions**:
- When this technique should be used vs. alternatives
- Problem characteristics that make it most beneficial (sequence length, model size, etc.)
- Scenarios where it might not be appropriate
- Hardware requirements or recommendations
- Scale considerations (small vs. large models)

**Expected_Outcomes**:
- Realistic expectations for performance improvements with specific metrics
- Timeline for seeing benefits (immediate vs. requiring training)
- Trade-offs or limitations to expect
- Comparison with baseline methods on relevant benchmarks
- Potential failure modes or degradation scenarios

Focus on actionable advice that would help a practitioner successfully implement and deploy the technique. Include specific numbers and concrete guidance wherever possible.
"""
}

def get_field_description(field_name):
    """Get the description for a specific field."""
    return FIELD_DESCRIPTIONS.get(field_name, "Field description not found.")

def get_metaprompt(field_name):
    """Get the metaprompt for generating a specific field."""
    return METAPROMPTS.get(field_name, "Metaprompt not found.")

def get_all_metaprompts():
    """Get all metaprompts as a dictionary."""
    return METAPROMPTS.copy()

def get_all_field_descriptions():
    """Get all field descriptions as a dictionary."""
    return FIELD_DESCRIPTIONS.copy()

# Usage example:
if __name__ == "__main__":
    print("Available fields:")
    for field, description in FIELD_DESCRIPTIONS.items():
        print(f"\n{field}:")
        print(description.strip())
    
    print("\n" + "="*50)
    print("Sample metaprompt for DESIGN_INSIGHT:")
    print("="*50)
    print(get_metaprompt("DESIGN_INSIGHT"))
