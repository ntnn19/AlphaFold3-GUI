# afusion/execution.py
import subprocess
import tempfile
import os
import time
from loguru import logger

def run_alphafold(command, slurm_log_dir, placeholder=None, slurm_time="48:00:00"):
    """
    Runs the AlphaFold command via Slurm and captures output.
    Uses placeholder to update output in real-time if provided.

    Parameters:
    - command: The singularity command to execute.
    - slurm_log_dir: Directory for Slurm log files.
    - placeholder: Optional streamlit placeholder for live output.
    - slurm_time: Maximum time for the job (default: '48:00:00').
    """

    # Create a temporary Slurm batch script
    slurm_script_path = tempfile.mktemp(suffix=".sh", prefix="alphafold_slurm_")
    slurm_script_content = f"""#!/bin/bash
#SBATCH --job-name=alphafold_run
#SBATCH --output={slurm_log_dir}/run/stdout-%j.log
#SBATCH --error={slurm_log_dir}/run/stderr-%j.log
#SBATCH --time={slurm_time}
#SBATCH --gpus=1

# Run the command
{command}
"""

    with open(slurm_script_path, "w") as slurm_script:
        slurm_script.write(slurm_script_content)

    logger.info(f"Slurm script created at: {slurm_script_path}")

    try:
        # Submit the job and capture job ID
        result = subprocess.run(f"sbatch {slurm_script_path}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            logger.error(f"Failed to submit Slurm job: {result.stderr}")
            return result.stderr

        job_id = result.stdout.strip().split()[-1]
        logger.info(f"Slurm job submitted with ID: {job_id}")

        # Poll for Slurm job completion
        stderr_path = f"{slurm_log_dir}/run/stderr-{job_id}.log"
        output_lines = []  # Store all lines
        last_read_pos = 0  # Track last read position

        while True:
            # Check job status using sacct
            status_result = subprocess.run(f"sacct -j {job_id} --format=State --noheader", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            job_status = status_result.stdout.strip()

            if "COMPLETED" in job_status:
                logger.info(f"Slurm job {job_id} completed.")
                break
            elif "FAILED" in job_status or "CANCELLED" in job_status:
                logger.error(f"Slurm job {job_id} failed or was cancelled.")
                return job_status

            # Read only new lines from Slurm log
            if os.path.exists(stderr_path):
                with open(stderr_path, "r") as output_file:
                    output_file.seek(last_read_pos)  # Move to last read position
                    new_lines = output_file.readlines()
                    last_read_pos = output_file.tell()  # Update last read position

                    output_lines.extend(new_lines)  # Store all lines

                    for line in new_lines:
                        logger.info(line.strip())  # Log new lines
                        if placeholder is not None:
                            placeholder.markdown(f"```\n{''.join(output_lines)}\n```")

            time.sleep(5)  # Reduce polling frequency

        return ''.join(output_lines)

    except Exception as e:
        logger.error(f"Error running Slurm job: {e}")
        return str(e)
