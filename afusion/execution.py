# afusion/execution.py

import subprocess
import tempfile
import os
from loguru import logger
import re

from afusion.config import (
    SINGULARITY_CONTAINER,
    DEFAULT_ALPHAFOLDARAMS,
    DEFAULT_AF_INPUT_PATH,
    DEFAULT_AF_OUTPUT_PATH,
)

def extract_job_id(output: str) -> str | None:
    match = re.search(r"Submitted batch job (\d+)", output)
    return match.group(1) if match else None

def extract_output_dir(cmd: str) -> str | None:
    match = re.search(r"--output_dir=(\S+)", cmd)
    return match.group(1) if match else None

def run_alphafold(command, placeholder=None):
    """
    Runs the AlphaFold command (Docker or Singularity) and captures output.
    Uses placeholder to update output in real-time if provided.
    Submits the command to SLURM via sbatch instead of using subprocess.
    """
    # Create a temporary SLURM script
    output_dir = extract_output_dir(command)
    log_dir = os.path.join(output_dir,"log")
    os.makedirs(log_dir,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        slurm_script_path = f.name
        # Write SLURM script content
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --job-name=alphafold\n")
        f.write(f"#SBATCH --output={log_dir}/slurm_output.log\n")
        f.write(f"#SBATCH --error={log_dir}/slurm_error.log\n")
        f.write("#SBATCH --nodes=1\n")
        f.write("#SBATCH --ntasks=1\n")
        f.write("#SBATCH --time=24:00:00\n")
        f.write("#SBATCH --partition=normal\n")
        f.write("#SBATCH --mem=180G\n")
        f.write("#SBATCH --nodelist=cssblivuke105\n")
        f.write("#SBATCH --gres=gpu:1\n")
        f.write("\n")
        f.write(f"{command}\n")
    try:
        # Make the script executable
        os.chmod(slurm_script_path, 0o755)

        # Submit the job to SLURM
        sbatch_command = f"sbatch {slurm_script_path}"
        process = subprocess.Popen(
            sbatch_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=True
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

        # Get the job ID for monitoring
        jobid = extract_job_id("".join(output_lines))
        logger.debug(f"SBATCH jobid={jobid}")

        return jobid
    finally:
        # Clean up the temporary script
        try:
            os.unlink(slurm_script_path)
        except:
            pass


def build_singularity_command(input_json_path, output_dir, use_gpu=True):
    """
    Builds a Singularity command to run AlphaFold 3 with the given parameters.

    :param input_json_path: Path to the input JSON file
    :param output_dir: Path to the output directory
    :param use_gpu: Whether to use GPU (default: True)
    :return: Singularity command string
    """
    # Build the base Singularity command
    # Add GPU flag if needed
    if use_gpu:
        nv_flag = " --nv"
    else:
        nv_flag = ""

    singularity_command = (
        f"singularity exec {nv_flag} --bind {output_dir}:{output_dir} --bind {DEFAULT_ALPHAFOLDARAMS['db_dir']}:{DEFAULT_ALPHAFOLDARAMS['db_dir']} --bind {DEFAULT_ALPHAFOLDARAMS['model_dir']}:{DEFAULT_ALPHAFOLDARAMS['model_dir']}  {SINGULARITY_CONTAINER} python /app/alphafold/run_alphafold.py"
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



    return singularity_command
