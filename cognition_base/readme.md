# RAG Generation Pipeline - PDF to JSON

## Overview

This pipeline implements the best-practice approach for converting research papers (PDFs) into structured JSON files for RAG (Retrieval Augmented Generation) systems. It extracts text from PDFs, processes them through Azure OpenAI GPT-5, and generates structured cognition base entries with five key fields.

## Architecture

The pipeline follows this workflow:

```
PDF Input → Text Extraction → Preprocessing → Field Generation → JSON Output
             (PyMuPDF)         (Cleaning)      (Azure OpenAI)     (Structured)
```

### Pipeline Steps

1. **PDF Text Extraction** (PyMuPDF/fitz)
   - Extracts text page-by-page with metadata
   - Preserves structure for born-digital PDFs
   - Tracks page numbers for provenance

2. **Text Preprocessing**
   - Removes headers, footers, and watermarks
   - Cleans excessive whitespace
   - Normalizes line breaks

3. **Field Generation** (Azure OpenAI GPT-5)
   - Generates 5 structured fields using specialized metaprompts
   - Uses temperature=0.3 for consistent, focused output
   - Processes fields in optimal order for context building

4. **JSON Assembly**
   - Validates and structures output
   - Saves with proper encoding (UTF-8)
   - Maintains source file provenance

## Output Schema

Each processed paper generates a JSON file with these five fields:

### 1. DESIGN_INSIGHT

High-level architectural innovations with priority levels (HIGH/MEDIUM/LOW), mathematical notation, and clear differentiation from prior work.

### 2. EXPERIMENTAL_TRIGGER_PATTERNS

Observable performance signatures including:

- **Task_Performance_Signatures**: Specific benchmark expectations
- **Architectural_Symptoms**: System-level behavioral indicators

### 3. BACKGROUND

Comprehensive context including:

- Exact paper title
- Historical technical context
- Technical limitations addressed
- Key concepts with LaTeX notation
- Experimental evaluation philosophy

### 4. ALGORITHMIC_INNOVATION

Detailed technical specifications:

- Core algorithm description
- Key mechanism explanation
- Mathematical formulation (with LaTeX)
- Computational properties analysis

### 5. IMPLEMENTATION_GUIDANCE

Practical implementation advice:

- Integration strategies
- Parameter settings with ranges
- Application conditions
- Expected outcomes and failure modes

## Installation

### Requirements

```bash
cd cognition_base
pip install -r requirements.txt
```

Key dependencies:

- `pymupdf>=1.23.0` - PDF text extraction
- `openai>=1.0.0` - Azure OpenAI API client
- Other dependencies for the RAG service

### Azure OpenAI Configuration

Ensure your `pipeline/config.py` has these settings:

```python
AZURE_ENDPOINT = "https://your-endpoint.openai.azure.com/"
AZURE_DEPLOYMENT_RAG_GENERATION = "gpt-5"  # Your deployment name
API_VERSION = "2025-01-01-preview"
API_KEY = "your-api-key"
```

## Usage

### Basic Usage

```bash
python rag_generation.py -i /path/to/pdf/folder -o /path/to/output/folder
```

### Arguments

- `-i, --input_folder`: Path to folder containing PDF files (required)
- `-o, --output_folder`: Path to output folder for JSON files (required)

### Example

```bash
# Process papers from raw folder, output to cognition folder
python rag_generation.py \
    -i cognition_base/raw/linear_attention \
    -o cognition_base/cognition/linear_attention
```

### Output

For each PDF file `paper.pdf`, the pipeline generates:

- `paper.json` in the output folder
- Console logging showing progress for each step
- Success/failure summary at the end

## Output Format Example

```json
[
    {
        "DESIGN_INSIGHT": "### DESIGN_INSIGHT_HIGH: [Technique Name]...",
        "EXPERIMENTAL_TRIGGER_PATTERNS": "**Task_Performance_Signatures:**...",
        "BACKGROUND": "**Title:** [Paper Title]...",
        "ALGORITHMIC_INNOVATION": "**Core_Algorithm:**...",
        "IMPLEMENTATION_GUIDANCE": "**Integration_Strategy:**..."
    }
]
```

## Processing Details

### Text Extraction

- Uses PyMuPDF (fitz) for reliable text extraction
- Preserves page structure and metadata
- Handles multi-column layouts reasonably well
- Born-digital PDFs work best (vs. scanned PDFs)

### Field Generation Order

Fields are generated in this order for optimal context:

1. BACKGROUND (establishes foundation)
2. DESIGN_INSIGHT (core innovations)
3. ALGORITHMIC_INNOVATION (technical details)
4. EXPERIMENTAL_TRIGGER_PATTERNS (evaluation)
5. IMPLEMENTATION_GUIDANCE (practical advice)

### API Settings

- **Temperature**: 0.3 (focused, consistent output)
- **Max Tokens**: 4000 per field
- **Model**: GPT-5 (or configured deployment)

## Error Handling

The pipeline handles errors gracefully:

- **PDF extraction failures**: Skips file, continues with others
- **Field generation failures**: Marks field with error message, continues
- **JSON saving errors**: Reports error, continues with next file
- **Summary report**: Shows success/failure counts at end

## Performance Considerations

### Token Limits

- Full paper text is sent for each field generation
- Very long papers (>100 pages) may hit token limits
- Consider chunking for extremely long papers if needed

### API Rate Limits

- Processing multiple papers makes sequential API calls
- Add delays if hitting rate limits (can modify code)
- Batch processing is sequential, not parallel

### Processing Time

- Typical paper: 2-5 minutes (5 fields × ~30-60 seconds each)
- Depends on paper length and API response time

## Troubleshooting

### Common Issues

1. **"No supported paper files found"**
   - Check that PDF files are in the input folder
   - Verify file extensions are `.pdf` (case-insensitive)

2. **"Error extracting text from PDF"**
   - PDF may be corrupted or encrypted
   - Scanned PDFs may have issues (consider OCR preprocessing)

3. **"Error generating [FIELD]"**
   - Check Azure OpenAI API connectivity
   - Verify API key and deployment name in config
   - Check for rate limiting or quota issues

4. **Empty or malformed output**
   - Review console logs for specific errors
   - Check that metaprompts are loading correctly
   - Verify paper text extraction was successful

## Best Practices

### Input PDF Quality

- ✅ Born-digital PDFs from arXiv, ACM, IEEE
- ✅ Text-searchable PDFs
- ⚠️ Scanned PDFs (may need OCR preprocessing)
- ❌ Image-only PDFs without OCR

### Output Quality

- Review generated JSON for accuracy
- Check mathematical notation rendering
- Verify field completeness
- Validate against existing examples

### Batch Processing

- Process papers in batches of 10-20 at a time
- Monitor API usage and costs
- Review outputs periodically during long runs

## Future Enhancements

Potential improvements to consider:

1. **Azure Document Intelligence Integration**
   - Better handling of scanned PDFs
   - Improved table and figure extraction
   - More robust structure detection

2. **Chunking Strategy**
   - Intelligent semantic splitting for long papers
   - Section-aware processing
   - Better handling of 100+ page documents

3. **Parallel Processing**
   - Multi-threaded paper processing
   - Batch API calls for efficiency
   - Progress tracking and resumption

4. **Quality Validation**
   - Automated JSON schema validation
   - Field completeness checks
   - Output quality scoring

## Related Files

- `rag_generation.py` - Main pipeline script
- `rag_api.py` - RAG API service
- `rag_service.py` - RAG retrieval service
- `requirements.txt` - Python dependencies
- `cognition/` - Generated JSON outputs

## Support

For issues or questions:

1. Check console output for specific error messages
2. Verify Azure OpenAI configuration
3. Review this README for troubleshooting steps
4. Check example outputs in `cognition/linear_attention/`
