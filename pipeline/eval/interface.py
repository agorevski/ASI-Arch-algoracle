import os
from typing import Tuple

from config import Config
from utils.verbose_logger import verbose_log_agent_run, log_training_progress, log_file_operation, log_error_context
from .model import debugger, trainer
from .prompts import Debugger_input


async def evaluation(name: str, motivation: str) -> bool:
    """
    Evaluate training performance for a given experiment.

    Args:
        name: Experiment name
        motivation: Experiment motivation

    Returns:
        True if training successful, False otherwise
    """
    success, error_msg = await run_training(name, motivation)
    if not success:
        print(f"Training failed: {error_msg}")
        return False
    save(name)
    return True


async def run_training(name: str, motivation: str) -> Tuple[bool, str]:
    """
    Run training script with debugging retry mechanism.

    Args:
        name: Experiment name
        motivation: Experiment motivation

    Returns:
        Tuple of (success_flag, error_message)
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
                print(f"Debug changes for {name}: {changes_made}")

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
                print(f"Training successful for {name}")
                return True, ""
            else:
                debug = True
                log_training_progress("Training Failed", f"Attempt {attempt + 1} failed", success=False)

                # Read debug file content as detailed error information
                try:
                    # If debug file doesn't exist, create an empty file
                    if not os.path.exists(Config.DEBUG_FILE):
                        with open(Config.DEBUG_FILE, 'w', encoding='utf-8') as f:
                            f.write("")

                    log_file_operation("READ", Config.DEBUG_FILE)
                    with open(Config.DEBUG_FILE, 'r', encoding='utf-8') as f:
                        debug_content = f.read()

                    log_file_operation("READ", Config.DEBUG_FILE, size=len(debug_content))

                    previous_error = f"Training failed. Debug info:\n{debug_content}"
                except Exception as e:
                    log_error_context("Debug File Read", e, {"debug_file": Config.DEBUG_FILE})
                    previous_error = (
                        f"Training failed. Cannot read debug file {Config.DEBUG_FILE}: {str(e)}"
                    )

                log_training_progress("Error Analysis", previous_error[:200])
                print(f"Training failed for {name} (attempt {attempt + 1}): {previous_error}")

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
        print(error_msg)
        return False, error_msg


def save(name: str) -> None:
    """
    Save source file content to code pool with given name.

    Args:
        name: File name to save as
    """
    # Read source file
    with open(Config.SOURCE_FILE, "r", encoding='utf-8') as f:
        content = f.read()

    # Write to code pool
    output_path = f"{Config.CODE_POOL}/{name}.py"
    with open(output_path, "w", encoding='utf-8') as f:
        f.write(content)

    log_file_operation("WRITE", output_path, f"Saved experiment code for {name}", size=len(content))
