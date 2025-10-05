import asyncio
import os

from agents import set_default_openai_api, set_default_openai_client, set_tracing_disabled
from openai import AsyncAzureOpenAI

from analyse import analyse
from database import program_sample, update
from eval import evaluation
from evolve import evolve
from utils.agent_logger import end_pipeline, log_error, log_info, log_step, log_warning, start_pipeline
from utils.verbose_logger import log_pipeline_step, log_error_context
import traceback
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config_loader import Config


client = AsyncAzureOpenAI(
    azure_endpoint=Config.AZURE_ENDPOINT,
    azure_deployment=Config.AZURE_DEPLOYMENT,
    api_version=Config.API_VERSION,
    api_key=Config.API_KEY,
)
set_default_openai_client(client)
set_default_openai_api("chat_completions")
set_tracing_disabled(True)


async def run_single_experiment() -> bool:
    """Run single experiment loop - using pipeline categorized logging."""
    # Start a new pipeline process
    pipeline_id = start_pipeline("experiment")

    try:
        # Step 1: Program sampling
        log_pipeline_step("Program Sampling", "Sampling program from database using UCT algorithm")
        log_step("Program Sampling", "Start sampling program from database")
        context, parent = await program_sample()
        log_info(f"Program sampling completed, context length: {len(str(context))}")

        # Step 2: Evolution
        log_pipeline_step("Program Evolution", f"Evolving new program from parent index: {parent}")
        log_step("Program Evolution", "Start evolving new program")
        name, motivation = await evolve(context)
        if name == "Failed":
            log_error("Program evolution failed")
            end_pipeline(False, "Evolution failed")
            return False
        log_info(f"Program evolution successful, generated program: {name}")
        log_info(f"Evolution motivation: {motivation}")

        # Step 3: Evaluation
        log_pipeline_step("Program Evaluation", f"Training and evaluating program: {name}")
        log_step("Program Evaluation", f"Start evaluating program {name}")
        success = await evaluation(name, motivation)
        if not success:
            log_error(f"Program {name} evaluation failed")
            end_pipeline(False, "Evaluation failed")
            return False
        log_info(f"Program {name} evaluation successful")

        # Step 4: Analysis
        log_pipeline_step("Result Analysis", f"Analyzing results for program: {name}")
        log_step("Result Analysis", f"Start analyzing program {name} results")
        result = await analyse(name, motivation, parent=parent)
        log_info(f"Analysis completed, result: {result}")

        # Step 5: Update database
        log_pipeline_step("Database Update", f"Storing results for program: {name}")
        log_step("Database Update", "Update results to database")
        update(result)
        log_info("Database update completed")

        # Successfully complete pipeline
        log_pipeline_step("Pipeline Complete", f"Experiment {name} completed successfully!")
        log_info("Experiment pipeline completed successfully")
        end_pipeline(True, f"Experiment completed successfully, program: {name}, result: {result}")
        return True

    except KeyboardInterrupt:
        log_error_context("Pipeline Execution", KeyboardInterrupt("User interrupted"), {"pipeline_id": pipeline_id})
        log_warning("User interrupted experiment")
        end_pipeline(False, "User interrupted experiment")
        return False
    except Exception as e:
        log_error_context("Pipeline Execution", e, {
            "pipeline_id": pipeline_id,
            "traceback": traceback.format_exc()
        })
        log_error(f"Experiment pipeline unexpected error: {str(e)}")
        log_error(f"Traceback: {traceback.format_exc()}")
        end_pipeline(False, f"Unexpected error: {str(e)}")
        return False


async def main():
    """Main function - continuous experiment execution."""
    set_tracing_disabled(True)

    log_info("Starting continuous experiment pipeline...")

    # Run plot.py first
    log_info("Running plot scripts...")
    log_info("Plot scripts completed")

    experiment_count = 0
    while True:
        try:
            experiment_count += 1
            log_info(f"Starting experiment {experiment_count}")

            success = await run_single_experiment()
            if success:
                log_info(f"Experiment {experiment_count} completed successfully, starting next experiment...")
            else:
                log_warning(f"Experiment {experiment_count} failed, retrying in 60 seconds...")
                await asyncio.sleep(60)

        except KeyboardInterrupt:
            log_warning("Continuous experiment interrupted by user")
            break
        except Exception as e:
            log_error(f"Main loop unexpected error: {e}")
            log_info("Retrying in 60 seconds...")
            await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(main())
