import datetime
import logging
import requests
import os
import subprocess
from typing import Any, Dict

from agents import function_tool
import traceback
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from config_loader import Config


@function_tool
def read_code_file() -> Dict[str, Any]:
    """Read a code file and return its contents.

    Reads the source file specified in the Config.SOURCE_FILE configuration.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - success (bool): True if the file was read successfully.
            - content (str): The file contents if successful.
            - error (str): Error message if unsuccessful.
    """
    source_file = Config.SOURCE_FILE
    try:
        with open(source_file, 'r') as f:
            content = f.read()
        return {'success': True, 'content': content}
    except Exception as e:
        return {'success': False, 'error': str(e)}


@function_tool
def read_csv_file(file_path: str) -> Dict[str, Any]:
    """Read a CSV file and return its contents.

    Args:
        file_path: The path to the CSV file to read.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - success (bool): True if the file was read successfully.
            - content (str): The file contents if successful.
            - error (str): Error message if unsuccessful.
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        return {'success': True, 'content': content}
    except Exception as e:
        return {'success': False, 'error': str(e)}


@function_tool
def write_code_file(content: str) -> Dict[str, Any]:
    """Write content to a code file.

    Writes the provided content to the source file specified in Config.SOURCE_FILE.

    Args:
        content: The content to write to the file.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - success (bool): True if the file was written successfully.
            - message (str): Success message if successful.
            - error (str): Error message if unsuccessful.
    """
    source_file = Config.SOURCE_FILE
    try:
        with open(source_file, 'w') as f:
            f.write(content)
        return {'success': True, 'message': 'Successfully written'}
    except Exception as e:
        return {'success': False, 'error': str(e)}


@function_tool
def run_training_script(name: str, script_path: str) -> Dict[str, Any]:
    """Run the training script and return its output.

    Executes a training script locally for up to 60 seconds as a sanity check,
    then schedules an Azure ML pipeline job for complete training.

    Args:
        name: The model name to pass to the training script.
        script_path: The relative path to the training script.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - success (bool): True if the script executed successfully.
            - message (str): Success message if successful.
            - output (str): The combined stdout/stderr output from the script.
            - error (str): Error message if unsuccessful.
    """
    try:
        script_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', script_path)

        args = ['python', script_path, '--model_name', name, '--sanity_test', True]
        logging.info(args)
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
        start_time = datetime.datetime.now()
        output_lines = []
        # Print output in real-time as it's generated
        for line in iter(process.stdout.readline, ''):
            logging.info(line.strip())
            output_lines.append(line.strip())
            cur_time = datetime.datetime.now()
            if cur_time - start_time > datetime.timedelta(seconds=60):  # Run for up to 60 seconds..
                logging.info("Local process is working, switching to AML...")
                process.kill()
                break
        process.wait()

        # Start a fresh proc that schedules an AML job for a complete training session
        args = ['python', './utils/schedule_aml_pipeline.py', '--model_name', name]
        logging.info(args)
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
        for line in iter(process.stdout.readline, ''):
            logging.info(line.strip())
            output_lines.append(line.strip())

        process.wait()

        if process.returncode == 0:
            return {
                'success': True,
                'message': 'Training script executed successfully',
                'output': '\n'.join(output_lines)
            }
        else:
            return {
                'success': False,
                'error': f'Script failed with return code {process.returncode}',
                'output': '\n'.join(output_lines)
            }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


@function_tool
def run_plot_script(script_path: str) -> Dict[str, Any]:
    """Run the plotting script.

    Executes a Python plotting script and captures its output.

    Args:
        script_path: The path to the plotting script to execute.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - success (bool): True if the script executed successfully.
            - output (str): The stdout output from the script.
            - error (str): The stderr output from the script.
    """
    try:
        args = ['python', script_path]
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=True
        )
        return {'success': True, 'output': result.stdout, 'error': result.stderr}
    except subprocess.CalledProcessError as e:
        return {'success': False, 'output': e.stdout, 'error': e.stderr}


def run_rag(query: str) -> Dict[str, Any]:
    """Run RAG (Retrieval-Augmented Generation) and return the results.

    Sends a query to the RAG service and retrieves relevant documents.

    Args:
        query: The search query to send to the RAG service.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - success (bool): True if the query was successful.
            - results (dict): The search results from the RAG service if successful.
            - error (str): Error message if unsuccessful.
    """
    try:
        response = requests.post(
            f'{Config.RAG}/search',
            headers={'Content-Type': 'application/json'},
            json={'query': query, 'k': 3, 'similarity_threshold': 0.5}
        )

        response.raise_for_status()
        results = response.json()

        return {'success': True, 'results': results}

    except Exception as e:
        logging.error(f"Traceback: {traceback.format_exc()}")
        return {'success': False, 'error': str(e)}
