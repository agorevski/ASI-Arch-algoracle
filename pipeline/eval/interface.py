import logging
import os
import sys
from typing import Tuple
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from config_loader import Config
from utils.verbose_logger import verbose_log_agent_run, log_training_progress, log_file_operation, log_error_context
from .model import debugger, trainer
from .prompts import Debugger_input

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(name)s-%(levelname)s-%(message)s')


async def evaluation(name: str, motivation: str) -> bool:
    """Evaluate training performance for a given experiment.

    Runs the training process for the specified experiment and saves the
    results if successful.

    Args:
        name (str): Experiment name used for identification and saving.
        motivation (str): Experiment motivation or description.

    Returns:
        bool: True if training completed successfully, False otherwise.
    """
    success, error_msg = await run_training(name, motivation)
    if not success:
        logging.error(f"Training failed: {error_msg}")
        return False
    save(name)
    return True


async def run_training(name: str, motivation: str) -> Tuple[bool, str]:
    """Run training script with debugging retry mechanism.

    Executes the training script, retrying with automated debugging if
    failures occur. The process continues until success or maximum
    debug attempts are exhausted.

    Args:
        name (str): Experiment name used for logging and identification.
        motivation (str): Experiment motivation passed to the debugger.

    Returns:
        Tuple[bool, str]: A tuple containing:
            - bool: True if training succeeded, False otherwise.
            - str: Empty string on success, error message on failure.

    Raises:
        No exceptions are raised; errors are caught and returned in the tuple.
    """
    log_training_progress("Training Started", f"Experiment: {name}")

    try:
        debug = True
        previous_error = ""

        for attempt in range(Config.MAX_DEBUG_ATTEMPT):
            log_training_progress(f"Training Attempt {attempt + 1}", f"Debug enabled: {debug}")

            if debug:
                log_training_progress("Debug Phase", "Running debugger to fix issues")
                debug_result = await verbose_log_agent_run(
                    "debugger",
                    debugger,
                    Debugger_input(motivation, previous_error)
                )

                changes_made = debug_result.final_output.changes_made
                log_training_progress("Debug Complete", f"Changes made: {changes_made}")
                logging.info(f"Debug changes for {name}: {changes_made}")

            log_training_progress("Training Execution", f"Running python: {Config.TRAINING_SCRIPT}")
            train_result = await verbose_log_agent_run(
                "trainer",
                trainer,
                f"""Please run the training script:
                1. Execute python {Config.TRAINING_SCRIPT} with parameter: {name}
                2. Only return success=True if script exits with code 0"""
            )

            if train_result.final_output.success:
                log_training_progress("Training Success", f"Experiment {name} completed successfully", success=True)
                logging.info(f"Training successful for {name}")
                return True, ""
            else:
                debug = True
                log_training_progress("Training Failed", f"Attempt {attempt + 1} failed", success=False)

                # Read debug file content as detailed error information
                try:
                    # Find the most recent training_error_*.txt file in DEBUG_FOLDER
                    debug_folder = Path(Config.DEBUG_FOLDER)
                    debug_folder.mkdir(parents=True, exist_ok=True)

                    error_files = list(debug_folder.glob("training_error_*.txt"))

                    if not error_files:
                        # No error files found, create empty placeholder
                        previous_error = "Training failed. No error details available."
                        log_training_progress("Error File Missing", "No training error files found")
                    else:
                        # Get the most recent error file by modification time
                        latest_error_file = max(error_files, key=lambda p: p.stat().st_mtime)

                        log_file_operation("READ", str(latest_error_file))
                        with open(latest_error_file, 'r', encoding='utf-8') as f:
                            debug_content = f.read()

                        log_file_operation("READ", str(latest_error_file), size=len(debug_content))
                        previous_error = f"Training failed. Debug info from {latest_error_file.name}:\n{debug_content}"

                except Exception as e:
                    log_error_context("Debug File Read", e, {"debug_folder": Config.DEBUG_FOLDER})
                    previous_error = (
                        f"Training failed. Cannot read debug files from {Config.DEBUG_FOLDER}: {str(e)}"
                    )

                log_training_progress("Error Analysis", previous_error[:200])
                logging.info(f"Training failed for {name} (attempt {attempt + 1}): {previous_error}")

                # If this is the last attempt, return failure
                if attempt == Config.MAX_DEBUG_ATTEMPT - 1:
                    final_error = (
                        f"Training failed after {Config.MAX_DEBUG_ATTEMPT} attempts. "
                        f"Final error: {previous_error}"
                    )
                    log_training_progress("Training Failed Permanently", final_error, success=False)
                    return False, final_error

                continue

    except Exception as e:
        error_msg = f"Unexpected error during training: {str(e)}"
        log_error_context("Training Process", e, {"name": name, "motivation": motivation[:200]})
        log_training_progress("Training Error", error_msg, success=False)
        logging.error(error_msg)
        return False, error_msg


def save(name: str) -> None:
    """Save source file content to code pool with given name.

    Reads the content from the configured source file and writes it to
    the code pool directory with the specified name.

    Args:
        name (str): File name (without extension) to save the content as.

    Returns:
        None
    """
    # Read source file
    with open(Config.SOURCE_FILE, "r", encoding='utf-8') as f:
        content = f.read()

    # Write to code pool
    output_path = f"{Config.CODE_POOL}/{name}.py"
    with open(output_path, "w", encoding='utf-8') as f:
        f.write(content)

    log_file_operation("WRITE", output_path, f"Saved experiment code for {name}", size=len(content))
