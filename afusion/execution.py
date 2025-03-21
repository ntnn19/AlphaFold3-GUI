# afusion/execution.py


import subprocess
import os
from loguru import logger


def run_snakemake(command, job_dir, placeholder):
    # Make sure job_dir is an absolute path
    job_dir = os.path.abspath(job_dir)

    # Ensure the job directory exists
    if not os.path.exists(job_dir):
        os.makedirs(job_dir)

    # Run the Snakemake command and capture output in the specified job_dir
    process = subprocess.Popen(command, shell=True, cwd=job_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Poll the process and update the output placeholder with the stdout
    while True:
        output = process.stdout.readline()
        if output == b'' and process.poll() is not None:
            break
        if output:
            placeholder.text(output.decode())

    # Get stderr if any error occurs
    stderr_output = process.stderr.read().decode()
    if stderr_output:
        placeholder.text(f"Error: {stderr_output}")

    return process.returncode

def run_alphafold(command, placeholder=None):
    """
    Runs the AlphaFold Docker command and captures output.
    Uses placeholder to update output in real-time if provided.
    """
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=True)
    output_lines = []
    for line in iter(process.stdout.readline, ''):
        if line:
            output_lines.append(line)
            logger.debug(line.strip())
            # Update placeholder if provided
            if placeholder is not None:
                placeholder.markdown(f"```\n{''.join(output_lines)}\n```")
    process.stdout.close()
    process.wait()
    return ''.join(output_lines)
