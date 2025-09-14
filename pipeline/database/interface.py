from typing import Tuple
import logging

from config import Config
from .element import DataElement
from .mongo_database import create_client

# Set up logging
logger = logging.getLogger(__name__)


# Create database instance
db = create_client()


async def program_sample() -> Tuple[str, int]:
    """
    Sample program using UCT algorithm and generate context.
    
    Process:
    1. Use UCT algorithm to select a node as parent node
    2. Get top 2 best results
    3. Get 2-50 random results
    4. Concatenate results into context
    5. The modified file is the program of the node selected by UCT
    
    Returns:
        Tuple containing context string and parent index
    """
    context = ""

    # Get parent element using UCT sampling with fallback mechanisms
    try:
        parent_candidates = db.candidate_sample_from_range(1, 10, 1)
        if not parent_candidates:
            logger.warning("No candidates found in range 1-10, falling back to regular sampling")
            parent_candidates = db.sample_from_range(1, 10, 1)
        
        if not parent_candidates:
            logger.warning("No elements found in regular sampling, using UCT selection")
            parent_element = db.uct_select_node()
            if parent_element is None:
                raise ValueError("No elements available in database for sampling")
        else:
            parent_element = parent_candidates[0]
            
    except Exception as e:
        logger.error(f"Failed to get parent element: {e}")
        # Final fallback: try to get any available element
        try:
            parent_element = db.sample_element()
            if parent_element is None:
                raise ValueError("Database appears to be empty - no elements available")
        except Exception as fallback_error:
            logger.error(f"All fallback methods failed: {fallback_error}")
            raise ValueError("Unable to retrieve any elements from database") from e

    # Get reference elements with error handling
    try:
        ref_elements = db.candidate_sample_from_range(11, 50, 4)
        if not ref_elements:
            logger.warning("No candidates found in range 11-50, falling back to regular sampling")
            ref_elements = db.sample_from_range(11, 50, 4)
        
        if not ref_elements:
            logger.warning("No reference elements found, using fewer elements or empty list")
            ref_elements = []
            
    except Exception as e:
        logger.warning(f"Failed to get reference elements: {e}, using empty list")
        ref_elements = []

    # Build context from parent and reference elements
    context += await parent_element.get_context()
    for element in ref_elements:
        context += await element.get_context()

    parent = parent_element.index
    
    # Write the program of the UCT selected node
    # If no node is selected, use the best result
    with open(Config.SOURCE_FILE, 'w', encoding='utf-8') as f:
        f.write(parent_element.program)
        print(f"[DATABASE] Implement Changes selected node (index: {parent})")
    
    return context, parent


def update(result: DataElement) -> bool:
    """
    Update database with new experimental result.
    
    Args:
        result: DataElement containing experimental results
        
    Returns:
        True if update successful
    """
    db.add_element_from_dict(result.to_dict())
    return True
