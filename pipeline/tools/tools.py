import logging
import requests
import os
import subprocess
from typing import Any, Dict

from agents import function_tool
from config import Config
import traceback


@function_tool
def read_code_file() -> Dict[str, Any]:
    """Read a code file and return its contents."""
    source_file = Config.SOURCE_FILE
    try:
        with open(source_file, 'r') as f:
            content = f.read()
        return {'success': True, 'content': content}
    except Exception as e:
        return {'success': False, 'error': str(e)}


@function_tool
def read_csv_file(file_path: str) -> Dict[str, Any]:
    """Read a CSV file and return its contents."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        return {'success': True, 'content': content}
    except Exception as e:
        return {'success': False, 'error': str(e)}


@function_tool
def write_code_file(content: str) -> Dict[str, Any]:
    """Write content to a code file."""
    source_file = Config.SOURCE_FILE
    try:
        with open(source_file, 'w') as f:
            f.write(content)
        return {'success': True, 'message': 'Successfully written'}
    except Exception as e:
        return {'success': False, 'error': str(e)}


@function_tool
def run_training_script(name: str, script_path: str) -> Dict[str, Any]:
    """Run the training script and return its output."""
    try:
        script_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', script_path)

        args = ['python', script_path, '--model_name', name]
        print(args)

        # Use Popen to capture output in real-time
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)

        output_lines = []
        # Print output in real-time as it's generated
        for line in iter(process.stdout.readline, ''):
            print(line.strip())
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
    """Run the plotting script."""
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
    """Run RAG and return the results."""
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
