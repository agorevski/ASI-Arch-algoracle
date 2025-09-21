import argparse
import logging
import os

from azure.ai.ml import MLClient, load_component, dsl
from azure.ai.ml.entities import UserIdentityConfiguration
from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential
from typing import Dict

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s-%(name)s-%(levelname)s-%(message)s'
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
                        type=str,
                        required=True,
                        help='Model name')
    args = parser.parse_args()
    for k,v in args.__dict__.items():
        logging.info(f'{k}: {v}')
    return args


def configure_logging() -> None:
    log_handlers = []
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    log_handlers.append(console_handler)
    logging.basicConfig(
        level=logging._nameToLevel["INFO"],
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=log_handlers
        )
    # Suppress noisy logs from azure, urllib3, msrest, etc.
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
    logging.getLogger("msrest.serialization").setLevel(logging.ERROR)
    logging.getLogger("azure.identity").setLevel(logging.WARNING)
    logging.getLogger("azure.ai.ml").setLevel(logging.WARNING)


def create_pipeline_job(component_inputs: Dict) -> object:
    @dsl.pipeline(display_name="ASI-Arch Training Job")
    def training_job_wrapper(model_name: str):
        run_lec_func = load_component(
            source=os.path.join(os.path.dirname(__file__), "aml_training_component.yaml"))
        run_step = run_lec_func(model_name=model_name)
        return {"raw_results": run_step.outputs.output_folder}
    return training_job_wrapper(model_name=component_inputs['model_name'])


def get_credential() -> TokenCredential:
    log_key = "get_credential()"
    credential = None
    try:
        url = "https://management.azure.com/.default"
        logging.info(f"{log_key}: Attempting to get credential using default credential")
        credential = DefaultAzureCredential()
        logging.info(f"{log_key}: Attempting to get token for {url}")
        credential.get_token(url)
    except Exception as e:
        logging.error(f"{log_key}: Failed with exception {e}")
    return credential


def submit_job(
    ml_client: MLClient,
    pipeline_job: object,
    cluster_name: str,
    experiment_name: str,
    stream_job: bool = True,
    tags: dict = None
) -> object:
    pipeline_job.settings.default_compute = cluster_name
    pipeline_job.identity = UserIdentityConfiguration()
    pipeline_job = ml_client.jobs.create_or_update(job=pipeline_job, experiment_name=experiment_name, tags=tags)
    logging.info(f"Pipeline job Name: {pipeline_job.name}")
    logging.info(f"Experiment Name: {experiment_name}")
    logging.info(f"Job link: {pipeline_job.studio_url}")
    if stream_job:
        ml_client.jobs.stream(pipeline_job.name)
    return pipeline_job


if __name__ == "__main__":
    configure_logging()
    args = parse_arguments()
    if args is None:
        logging.error("Failed to parse arguments.")
        exit()

    cred = get_credential()
    if cred is None:
        raise RuntimeError("Failed to obtain credentials")

    ml_client = MLClient(
        credential=cred,
        subscription_id=Config.AML_SUBSCRIPTION_ID,
        resource_group_name=Config.AML_RESOURCE_GROUP,
        workspace_name=Config.AML_WORKSPACE_NAME
    )
    ml_client.compute.get(Config.AML_CLUSTER_NAME)

    component_inputs = {
        "model_name": args.model_name
    }

    pipeline_job = create_pipeline_job(component_inputs)

    tags = {}
    tags["model_name"] = args.model_name

    job = submit_job(
        ml_client=ml_client,
        pipeline_job=pipeline_job,
        cluster_name=Config.AML_CLUSTER_NAME,
        experiment_name="ASI-Arch Training Job",
        stream_job=True,
        tags=tags
    )

    # Extract just the job name (after the last '/')
    job_identifier = job.id.split('/')[-1]
    logging.info(f"Job ID: {job_identifier}")
    # Wait for job completion
    logging.info("Waiting for job to complete...")
    ml_client.jobs.stream(job.name)

    # Get the completed job to access outputs
    completed_job = ml_client.jobs.get(job.name)

    # Download the output from raw_results
    if hasattr(completed_job, 'outputs') and 'raw_results' in completed_job.outputs:
        output_uri = completed_job.outputs['raw_results']
        logging.info(f"Downloading output from URI: {output_uri}")

        # Create local download directory
        download_path = os.path.join(os.path.dirname(__file__), f"job_outputs_{job_identifier}")
        os.makedirs(download_path, exist_ok=True)

        # Download the output
        ml_client.jobs.download(name=job.name, download_path=download_path, output_name="raw_results")
        logging.info(f"Output downloaded to: {download_path}")
    else:
        logging.error("raw_results output not found in completed job")
