import argparse
import base64
import concurrent.futures
import logging
import fitz  # PyMuPDF
import json
import os
import re
import sys

from functools import partial
from io import BytesIO
from openai import AzureOpenAI
from PIL import Image
from prompts import get_field_description, get_metaprompt
from typing import Dict, List, Optional, Tuple
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config_loader import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(name)s-%(levelname)s-%(message)s')


def extract_text_and_images_from_pdf(pdf_path: str) -> Tuple[str, List[Dict]]:
    """
    Extract text, images, and metadata from a PDF file using PyMuPDF.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Tuple of (full_text, page_metadata_list)
        - full_text: Complete text extracted from the PDF
        - page_metadata_list: List of dicts with page numbers, text, and images per page
    """
    try:
        doc = fitz.open(pdf_path)
        full_text = []
        page_metadata = []
        total_images = 0
        extracted_images = 0

        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text()
            page_images = []

            # Extract images from the page
            # Extract images from the page with size filtering
            image_list = page.get_images()

            # Filter out small images (likely icons, bullets, etc.)
            filtered_images = []
            for img in image_list:
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)

                    # Get image dimensions
                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)

                    # Filter criteria: minimum size thresholds
                    min_width = 200  # pixels
                    min_height = 100  # pixels
                    min_area = 50000  # square pixels

                    if (width >= min_width and height >= min_height and width * height >= min_area):
                        filtered_images.append((img, width * height))

                except Exception:
                    continue

            # Sort by area (largest first) and take top 50
            filtered_images.sort(key=lambda x: x[1], reverse=True)
            filtered_images = [img[0] for img in filtered_images[:50]]

            image_list = filtered_images
            total_images += len(image_list)

            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]

                    # Convert to PIL Image
                    pil_image = Image.open(BytesIO(image_bytes))

                    # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')

                    # Resize image if larger than 150x150 using bicubic interpolation
                    if pil_image.width > 150 or pil_image.height > 150:
                        pil_image = pil_image.resize((150, 150), Image.BICUBIC)

                    # Display image on screen
                    # pil_image.show()

                    # Save as JPEG with quality=80 to BytesIO
                    buffer = BytesIO()
                    pil_image.save(buffer, format='JPEG', quality=80, optimize=True)
                    buffer.seek(0)

                    # Encode as base64
                    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                    page_images.append(image_base64)
                    extracted_images += 1

                except Exception as img_error:
                    # Graceful failure - log warning but continue
                    logging.warning(f"  ⚠ Warning: Could not extract image {img_index + 1} from page {page_num}: {str(img_error)}")
                    continue

            # Store page-level metadata
            page_metadata.append({
                'page_number': page_num,
                'text': page_text,
                'char_count': len(page_text),
                'images': page_images,
                'image_count': len(page_images)
            })

            full_text.append(page_text)

        doc.close()

        combined_text = '\n\n'.join(full_text)

        # Print image extraction summary
        if total_images > 0:
            logging.info(f"  ✓ Extracted {extracted_images}/{total_images} images from PDF")

        return combined_text, page_metadata

    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {str(e)}")
        return "", []


def clean_text(text: str) -> str:
    """
    Clean and preprocess extracted text.

    Args:
        text: Raw text from PDF

    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove page numbers (common patterns)
    text = re.sub(r'\n\d+\n', '\n', text)

    # Remove common header/footer patterns
    text = re.sub(r'(Page \d+ of \d+)', '', text, flags=re.IGNORECASE)

    # Normalize line breaks
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def generate_field_content(client: AzureOpenAI, paper_text: str, field_name: str, page_metadata: List[Dict]) -> Optional[str]:
    """
    Generate content for a specific field using Azure OpenAI with vision support.

    Args:
        client: Azure OpenAI client
        paper_text: Full text of the research paper
        field_name: Name of the field to generate
        page_metadata: List of page metadata including images

    Returns:
        Generated field content as string, or None if generation fails
    """

    field_description = get_field_description(field_name)
    metaprompt = get_metaprompt(field_name)

    system_prompt = f"""You are an expert research paper analyst specializing in extracting 
structured technical information from academic papers in machine learning, deep learning, 
and AI architectures. Your task is to generate the {field_name} field following the 
precise metaprompt instructions using a description of what the field name is about.

You will be provided with both the text content and all images (figures, diagrams, charts, equations) 
from the research paper. Analyze both text and visual elements to provide comprehensive analysis."""

    user_prompt_text = f"""Research Paper Text:
---
{paper_text}
---

Description of {field_name}:
---
{field_description}
---

Metaprompt for {field_name}:
---
{metaprompt}
---

Generate ONLY the {field_name} content following the exact format specified in the metaprompt above.
Analyze both the text and the images provided to extract comprehensive technical information."""

    try:
        logging.info(f"  Generating {field_name}...")

        # Build multimodal content array
        content = [{"type": "text", "text": user_prompt_text}]

        # Add all images from all pages
        total_images_added = 0
        for page_data in page_metadata:
            for image_base64 in page_data.get('images', []):
                if total_images_added < 50:  # Limit to first 50 images
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    })
                    total_images_added += 1

        if total_images_added > 0:
            logging.info(f"  → Including {total_images_added} images in the analysis")

        response = client.chat.completions.create(
            model=Config.AZURE_DEPLOYMENT_RAG_GENERATION,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ],
            temperature=1,
            max_completion_tokens=10000
        )

        content_result = response.choices[0].message.content.strip()

        if content_result:
            logging.info(f"  ✓ {field_name} generated successfully ({len(content_result)} chars)")
            logging.info(f"  → Preview: {content_result[:250]}...")
            return content_result
        else:
            logging.warning(f"  ✗ {field_name} generation returned empty content")
            return None

    except Exception as e:
        logging.error(f"  ✗ Error generating {field_name}: {str(e)}")
        return None


def create_json_output(fields: Dict[str, str]) -> List[Dict]:
    """
    Create the final JSON output structure.

    Args:
        fields: Dictionary mapping field names to their generated content

    Returns:
        List containing a single dictionary with all fields
    """
    output = [{
        "DESIGN_INSIGHT": fields.get("DESIGN_INSIGHT", ""),
        "EXPERIMENTAL_TRIGGER_PATTERNS": fields.get("EXPERIMENTAL_TRIGGER_PATTERNS", ""),
        "BACKGROUND": fields.get("BACKGROUND", ""),
        "ALGORITHMIC_INNOVATION": fields.get("ALGORITHMIC_INNOVATION", ""),
        "IMPLEMENTATION_GUIDANCE": fields.get("IMPLEMENTATION_GUIDANCE", "")
    }]

    return output


def process_single_paper(client: AzureOpenAI, paper_path: str, output_folder: str) -> bool:
    """
    Process a single research paper and generate JSON output.

    Args:
        client: Azure OpenAI client
        paper_path: Path to the PDF file
        output_folder: Directory to save output JSON

    Returns:
        True if processing succeeded, False otherwise
    """
    paper_filename = os.path.basename(paper_path)
    paper_name = os.path.splitext(paper_filename)[0]

    logging.info(f"\n{'='*80}")
    logging.info(f"Processing: {paper_filename}")
    logging.info(f"{'='*80}")

    # Step 1: Extract text from PDF
    logging.info("\n[1/5] Extracting text from PDF...")
    raw_text, page_metadata = extract_text_and_images_from_pdf(paper_path)

    if not raw_text:
        logging.error(f"✗ Failed to extract text from {paper_filename}")
        return False

    logging.info(f"✓ Extracted {len(raw_text)} characters from {len(page_metadata)} pages")

    # Step 2: Clean text
    logging.info("\n[2/5] Cleaning and preprocessing text...")
    cleaned_text = clean_text(raw_text)
    logging.info(f"✓ Cleaned text: {len(cleaned_text)} characters")

    # Step 3: Generate all fields
    logging.info("\n[3/5] Generating structured fields with Azure OpenAI...")
    fields = {}
    field_order = [
        "DESIGN_INSIGHT",
        "EXPERIMENTAL_TRIGGER_PATTERNS",
        "BACKGROUND",
        "ALGORITHMIC_INNOVATION",
        "IMPLEMENTATION_GUIDANCE"
    ]

    # Generate all field data in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(field_order)) as executor:
        # Create a partial function with common parameters
        generate_func = partial(generate_field_content, client, cleaned_text)

        # Submit all field generation tasks
        future_to_field = {
            executor.submit(generate_func, field_name, page_metadata): field_name
            for field_name in field_order
        }

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_field):
            field_name = future_to_field[future]
            try:
                content = future.result()
                if content:
                    fields[field_name] = content
                else:
                    fields[field_name] = f"[Error: Failed to generate {field_name}]"
            except Exception as e:
                logging.error(f"  ✗ Exception generating {field_name}: {str(e)}")
                fields[field_name] = f"[Error: Exception in {field_name}]"

    # Step 4: Create JSON structure
    logging.info("\n[4/5] Creating JSON output structure...")
    json_output = create_json_output(fields)

    # Step 5: Save to file
    logging.info("\n[5/5] Saving output...")
    output_path = os.path.join(output_folder, f"{paper_name}.json")

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, indent=4, ensure_ascii=False)

        logging.info(f"✓ Successfully saved to: {output_path}")
        logging.info(f"\n{'='*80}")
        logging.info(f"✓ Processing completed successfully for {paper_filename}")
        logging.info(f"{'='*80}\n")
        return True

    except Exception as e:
        logging.error(f"✗ Error saving output file: {str(e)}")
        return False


def process_papers(input_folder, output_folder):
    """
    Process research papers from input folder and generate cognition base JSON files.

    Args:
    input_folder (str): Path to folder containing research papers (PDF/HTML)
    output_folder (str): Path to output folder for generated JSON files
    """
    # Validate input folder exists
    if not os.path.exists(input_folder):
        logging.error(f"Error: Input folder '{input_folder}' does not exist.")
        sys.exit(1)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    logging.info(f"Processing papers from: {input_folder}")
    logging.info(f"Output will be saved to: {output_folder}")

    client = AzureOpenAI(
        azure_endpoint=Config.AZURE_ENDPOINT,
        azure_deployment=Config.AZURE_DEPLOYMENT_RAG_GENERATION,
        api_version=Config.API_VERSION,
        api_key=Config.API_KEY,
    )

    # Get list of paper files
    supported_extensions = ['.pdf']
    paper_files = []
    for filename in os.listdir(input_folder):
        if any(filename.lower().endswith(ext) for ext in supported_extensions):
            paper_files.append(os.path.join(input_folder, filename))

    if not paper_files:
        logging.warning(f"No supported paper files found in {input_folder}")
        return

    logging.info(f"Found {len(paper_files)} paper(s) to process\n")

    # Process each paper
    success_count = 0
    fail_count = 0

    for i, paper_path in enumerate(paper_files, 1):
        logging.info(f"\n[Paper {i}/{len(paper_files)}]")
        success = process_single_paper(client, paper_path, output_folder)
        if success:
            success_count += 1
        else:
            fail_count += 1

    # Print summary
    logging.info(f"\n{'='*80}")
    logging.info("PROCESSING COMPLETE")
    logging.info(f"{'='*80}")
    logging.info(f"Total papers: {len(paper_files)}")
    logging.info(f"Successfully processed: {success_count}")
    logging.info(f"Failed: {fail_count}")
    logging.info(f"{'='*80}\n")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate structured JSON outputs for research paper analysis using RAG metaprompts")
    parser.add_argument('--input_folder', '-i', required=True, type=str, help='Path to folder containing research papers (PDF or HTML files)')
    parser.add_argument('--output_folder', '-o', required=True, type=str, help='Path to output folder where JSON files will be saved')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(name)s-%(levelname)s-%(message)s')
    # Suppress httpx INFO logs to avoid showing HTTP request details
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args = parse_arguments()
    # Process the papers
    process_papers(args.input_folder, args.output_folder)
