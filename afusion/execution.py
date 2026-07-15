# afusion/execution.py

import subprocess

from config import (
    SINGULARITY_CONTAINER,
    DEFAULT_ALPHAFOLDARAMS,
    DEFAULT_AF_INPUT_PATH,
    DEFAULT_AF_OUTPUT_PATH,
)


def run_alphafold(command, placeholder=None):
    """
    Runs the AlphaFold command (Docker or Singularity) and captures output.
    Uses placeholder to update output in real-time if provided.
    """
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=True
    )
    output_lines = []
    for line in iter(process.stdout.readline, ""):
        if line:
            output_lines.append(line)
            logger.debug(line.strip())
            # Update placeholder if provided
            if placeholder is not None:
                placeholder.markdown(f"```\n{''.join(output_lines)}\n```")
    process.stdout.close()
    process.wait()
    return "".join(output_lines)


def build_singularity_command(input_json_path, output_dir, use_gpu=True):
    """
    Builds a Singularity command to run AlphaFold 3 with the given parameters.

    :param input_json_path: Path to the input JSON file
    :param output_dir: Path to the output directory
    :param use_gpu: Whether to use GPU (default: True)
    :return: Singularity command string
    """
    # Build the base Singularity command
    singularity_command = (
        f"singularity exec {SINGULARITY_CONTAINER} python run_alphafold.py"
    )

    # Add the basic required parameters
    singularity_command += f" --json_path={input_json_path}"
    singularity_command += f" --output_dir={output_dir}"
    singularity_command += f" --model_dir={DEFAULT_ALPHAFOLDARAMS['model_dir']}"

    # Add database parameters from the config
    singularity_command += f" --db_dir={DEFAULT_ALPHAFOLDARAMS['db_dir']}"

    # Add all the database paths and z-values from the constant config
    for param_name, param_value in DEFAULT_ALPHAFOLDARAMS.items():
        if param_name.endswith('_database_path') or param_name.endswith('_z_value') or param_name.endswith('_n_cpu') or param_name.endswith('_max_parallel_shards'):
            singularity_command += f" --{param_name}={param_value}"

    # Add force_output_dir flag (this is a boolean flag, no value needed)
    if DEFAULT_ALPHAFOLDARAMS.get('force_output_dir', False):
        singularity_command += " --force_output_dir"

    # Add GPU flag if needed
    if use_gpu:
        singularity_command += " --nv"

    return singularity_command
