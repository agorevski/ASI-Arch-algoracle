import os
from typing import Tuple
import logging

from .element import DataElement
from .mongo_database import create_client
from utils.verbose_logger import log_database_operation, log_file_operation, log_error_context
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from config_loader import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(name)s-%(levelname)s-%(message)s')

# Create database instance
db = create_client()


async def program_sample() -> Tuple[str, int]:
    """Sample program using UCT algorithm and generate context.

    Uses the UCT (Upper Confidence bounds applied to Trees) algorithm to select
    a parent node, retrieves reference elements, and builds a context string
    for program generation. The selected node's program is written to the
    configured source file.

    Process:
        1. Use UCT algorithm to select a node as parent node
        2. Get top 2 best results
        3. Get 2-50 random results
        4. Concatenate results into context
        5. The modified file is the program of the node selected by UCT

    Returns:
        Tuple[str, int]: A tuple containing:
            - context (str): The concatenated context string from parent and
              reference elements.
            - parent (int): The index of the selected parent element.

    Raises:
        ValueError: If no elements are available in the database for sampling.
    """
    context = ""

    # Get parent element using UCT sampling with fallback mechanisms
    log_database_operation("Parent Element Sampling", {
        "operation": "UCT algorithm selection",
        "target_range": "1-10",
        "count": 1
    })

    try:
        parent_candidates = db.candidate_sample_from_range(1, 10, 1)
        if not parent_candidates:
            logging.warning("No candidates found in range 1-10, falling back to regular sampling")
            log_database_operation("Fallback to Regular Sampling", {
                "reason": "No candidates in range 1-10"
            })
            parent_candidates = db.sample_from_range(1, 10, 1)

        if not parent_candidates:
            logging.warning("No elements found in regular sampling, using UCT selection")
            log_database_operation("Fallback to UCT Selection", {
                "reason": "No elements in regular sampling"
            })
            parent_element = db.uct_select_node()
            if parent_element is None:
                raise ValueError("No elements available in database for sampling")
        else:
            parent_element = parent_candidates[0]

        log_database_operation("Parent Element Selected", {
            "element_index": parent_element.index,
            "element_name": parent_element.name,
            "program_size": len(parent_element.program)
        })

    except Exception as e:
        log_error_context("Parent Element Sampling", e)
        logging.error(f"Failed to get parent element: {e}")
        # Final fallback: try to get any available element
        try:
            parent_element = db.sample_element()
            if parent_element is None:
                raise ValueError("Database appears to be empty - no elements available")
        except Exception as fallback_error:
            logging.error(f"All fallback methods failed: {fallback_error}")
            raise ValueError("Unable to retrieve any elements from database") from e

    # Get reference elements with error handling
    log_database_operation("Reference Elements Sampling", {
        "operation": "candidate sampling",
        "target_range": "11-50",
        "count": 4
    })

    try:
        ref_elements = db.candidate_sample_from_range(11, 50, 4)
        if not ref_elements:
            logging.warning("No candidates found in range 11-50, falling back to regular sampling")
            log_database_operation("Fallback to Regular Reference Sampling", {"reason": "No candidates in range 11-50"})
            ref_elements = db.sample_from_range(11, 50, 4)

        if not ref_elements:
            logging.warning("No reference elements found, using fewer elements or empty list")
            ref_elements = []

        log_database_operation("Reference Elements Retrieved", {
            "count": len(ref_elements),
            "elements": [{"index": el.index, "name": el.name} for el in ref_elements[:3]]
        })

    except Exception as e:
        log_error_context("Reference Elements Sampling", e)
        logging.warning(f"Failed to get reference elements: {e}, using empty list")
        ref_elements = []

    # Build context from parent and reference elements
    log_database_operation("Context Building", {
        "parent_element": parent_element.index,
        "reference_count": len(ref_elements)
    })

    context += await parent_element.get_context()
    for element in ref_elements:
        context += await element.get_context()

    log_database_operation("Context Generated", {
        "total_context_length": len(context),
        "parent_index": parent_element.index
    })

    parent = parent_element.index

    # Write the program of the UCT selected node
    log_file_operation("WRITE", Config.SOURCE_FILE, content_preview=parent_element.program, size=len(parent_element.program))

    with open(Config.SOURCE_FILE, 'w', encoding='utf-8') as f:
        f.write(parent_element.program)
        logging.info(f"[DATABASE] Implement Changes selected node (index: {parent})")

    return context, parent


def update(result: DataElement) -> bool:
    """Update database with new experimental result.

    Adds a new DataElement to the database, logging the operation details
    and any errors that occur during the update.

    Args:
        result (DataElement): The experimental result to add to the database.
            Should contain name, parent index, analysis, and program data.

    Returns:
        bool: True if the update was successful, False otherwise.
    """
    log_database_operation("Database Update", {"element_name": result.name, "parent_index": result.parent, "analysis_length": len(result.analysis) if result.analysis else 0, "program_size": len(result.program) if result.program else 0})
    try:
        db.add_element_from_dict(result.to_dict())
        log_database_operation("Database Update Complete", {"status": "success", "element_name": result.name})
        return True
    except Exception as e:
        log_error_context("Database Update", e, {"element_name": result.name, "result_dict_keys": list(result.to_dict().keys())})
        return False
