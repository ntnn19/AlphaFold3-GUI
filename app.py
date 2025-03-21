# simplified app.py for streamlit cloud

import streamlit as st
import json
import re
import os
import sys
from loguru import logger

# Import your modules (make sure they are correctly installed in your environment)
from afusion.execution import run_alphafold
from afusion.sequence_input import (
    collect_protein_sequence_data,
    collect_rna_sequence_data,
    collect_dna_sequence_data,
    collect_ligand_sequence_data
)
from afusion.bonds import handle_bond
from afusion.utils import compress_output_folder

# Configure the logger
logger.add("afusion.log", rotation="1 MB", level="DEBUG")

def main():
    # Set page configuration and theme
    st.set_page_config(
        page_title="AFusion: AlphaFold 3 GUI",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS styling
    st.markdown("""
        <style>
        /* Remove padding */
        .css-18e3th9 {
            padding-top: 0rem;
            padding-bottom: 0rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        /* Header styling */
        .css-1v3fvcr {
            font-size: 2rem;
            color: #2c3e50;
        }
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #f2f4f5;
        }
        /* Button styling */
        .stButton>button {
            background-color: #2c3e50;
            color: white;
            border-radius: 5px;
        }
        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-thumb {
            background: #2c3e50;
            border-radius: 10px;
        }
        </style>
        """, unsafe_allow_html=True)

    # Title and subtitle
    st.markdown("<h1 style='text-align: center;'>🔬 AFusion: AlphaFold 3 GUI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 16px;'>A convenient GUI for running AlphaFold 3 predictions</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 14px;'>If this project helps you, please ⭐️ <a href='https://github.com/Hanziwww/AlphaFold3-GUI' target='_blank'>my project</a>!</p>", unsafe_allow_html=True)

    #### Sidebar Navigation ####
    with st.sidebar:
        st.header("Navigation")
        st.sidebar.markdown("---")
        sections = {
            "Job Settings": "job_settings",
            "Sequences": "sequences",
            "Bonded Atom Pairs": "bonded_atom_pairs",
            "User Provided CCD": "user_ccd",
            "Generated JSON": "json_content",
            "Execution Settings": "execution_settings",
            "Run AlphaFold 3": "run_alphafold",
        }
        for section_name, section_id in sections.items():
            st.markdown(f"<a href='#{section_id}' style='text-decoration: none;'>{section_name}</a>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("<small>Created by Hanzi 2024.</small>", unsafe_allow_html=True)

    # Main Content
    st.markdown('<div id="home"></div>', unsafe_allow_html=True)
    st.markdown("### Welcome to AFusion!")
    st.markdown("Use this GUI to generate input JSON files and run AlphaFold 3 predictions with ease. Please [install AlphaFold 3](https://github.com/google-deepmind/alphafold3/blob/main/docs/installation.md) before using.")

    st.markdown('<div id="job_settings"></div>', unsafe_allow_html=True)
    st.header("📝 Job Settings")
    with st.expander("Configure Job Settings", expanded=True):
        job_name = st.text_input("Job Name", value="My AlphaFold Job", help="Enter a descriptive name for your job.")
        logger.info(f"Job name set to: {job_name}")
        model_seeds = st.text_input("Model Seeds (comma-separated)", value="1,2,3", help="Provide integer seeds separated by commas.")
        logger.debug(f"Model seeds input: {model_seeds}")
        model_seeds_list = [int(seed.strip()) for seed in model_seeds.split(",") if seed.strip().isdigit()]
        if not model_seeds_list:
            st.error("Please provide at least one valid model seed.")
            logger.error("No valid model seeds provided.")
            st.stop()
        logger.info(f"Model seeds list: {model_seeds_list}")

    st.markdown('<div id="sequences"></div>', unsafe_allow_html=True)
    st.header("📄 Sequences")
    sequences = []
    num_entities = st.number_input("Number of Entities", min_value=1, step=1, value=1, help="Select the number of entities you want to add.")
    logger.info(f"Number of entities set to: {num_entities}")

    for i in range(int(num_entities)):
        st.markdown(f"### Entity {i+1}")
        with st.expander(f"Entity {i+1} Details", expanded=True):
            entity_type = st.selectbox(f"Select Entity Type", ["Protein 🧬", "RNA 🧫", "DNA 🧬", "Ligand 💊"], key=f"entity_type_{i}")
            logger.info(f"Entity {i+1} type: {entity_type}")

            copy_number = st.number_input(f"Copy Number", min_value=1, step=1, value=1, key=f"copy_number_{i}", help="Specify the number of copies of this sequence.")
            logger.info(f"Entity {i+1} copy number: {copy_number}")

            # Collect sequence data
            if entity_type.startswith("Protein"):
                sequence_data = collect_protein_sequence_data(i)
            elif entity_type.startswith("RNA"):
                sequence_data = collect_rna_sequence_data(i)
            elif entity_type.startswith("DNA"):
                sequence_data = collect_dna_sequence_data(i)
            elif entity_type.startswith("Ligand"):
                sequence_data = collect_ligand_sequence_data(i)
            else:
                st.error(f"Unknown entity type: {entity_type}")
                logger.error(f"Unknown entity type: {entity_type}")
                continue  # Skip to next entity

            # Handle IDs based on copy number
            entity_ids = []
            if copy_number >= 1:
                # Allow user to input multiple IDs
                entity_id = st.text_input(f"Entity ID(s) (comma-separated)", key=f"entity_id_{i}", help="Provide entity ID(s), separated by commas if multiple.")
                if not entity_id.strip():
                    st.error("Entity ID is required.")
                    logger.error("Entity ID missing.")
                    continue
                entity_ids = re.split(r"\s*,\s*", entity_id)
                if len(entity_ids) != copy_number:
                    st.error(f"Please provide {copy_number} ID(s) separated by commas.")
                    logger.error(f"Number of IDs provided does not match copy number for Entity {i+1}.")
                    continue
                logger.debug(f"Entity {i+1} IDs: {entity_ids}")

                for copy_id in entity_ids:
                    # Clone the sequence data and set the ID
                    sequence_entry = sequence_data.copy()
                    sequence_entry['id'] = copy_id
                    # Wrap the entry appropriately
                    if entity_type.startswith("Protein"):
                        sequences.append({"protein": sequence_entry})
                    elif entity_type.startswith("RNA"):
                        sequences.append({"rna": sequence_entry})
                    elif entity_type.startswith("DNA"):
                        sequences.append({"dna": sequence_entry})
                    elif entity_type.startswith("Ligand"):
                        sequences.append({"ligand": sequence_entry})
            else:
                st.error("Copy number must be at least 1.")
                logger.error("Invalid copy number.")
                continue

    st.markdown('<div id="bonded_atom_pairs"></div>', unsafe_allow_html=True)
    st.header("🔗 Bonded Atom Pairs (Optional)")
    bonded_atom_pairs = []
    add_bonds = st.checkbox("Add Bonded Atom Pairs")
    if add_bonds:
        num_bonds = st.number_input("Number of Bonds", min_value=1, step=1, key="num_bonds")
        for b in range(int(num_bonds)):
            st.markdown(f"**Bond {b+1}**")
            bond = handle_bond(b)
            if bond:
                bonded_atom_pairs.append(bond)
                logger.debug(f"Added bond: {bond}")

    st.markdown('<div id="user_ccd"></div>', unsafe_allow_html=True)
    st.header("🧩 User Provided CCD (Optional)")
    user_ccd = st.text_area("User CCD (mmCIF format)")
    if user_ccd:
        logger.debug("User provided CCD data.")

    # Generate JSON Data
    alphafold_input = {
        "name": job_name,
        "modelSeeds": model_seeds_list,
        "sequences": sequences,
        "dialect": "alphafold3",
        "version": 1
    }

    if bonded_atom_pairs:
        alphafold_input["bondedAtomPairs"] = bonded_atom_pairs

    if user_ccd:
        alphafold_input["userCCD"] = user_ccd

    # Convert JSON to string
    json_output = json.dumps(alphafold_input, indent=2)
    logger.debug(f"Generated JSON:\n{json_output}")

    st.markdown('<div id="json_content"></div>', unsafe_allow_html=True)
    st.header("📄 Generated JSON Content")
    st.code(json_output, language="json")

    st.markdown('<div id="execution_settings"></div>', unsafe_allow_html=True)
    st.header("⚙️ AlphaFold 3 Execution Settings")
    with st.expander("Configure Execution Settings", expanded=True):
        # Paths for execution
        af_input_path = st.text_input("AF Input Path", value=os.path.expanduser("~/af_input"), help="Path to AlphaFold input directory.")
        af_output_path = st.text_input("AF Output Path", value=os.path.expanduser("~/af_output"), help="Path to AlphaFold output directory.")
        model_parameters_dir = st.text_input("Model Parameters Directory", value="/path/to/models", help="Path to model parameters directory.")
        databases_dir = st.text_input("Databases Directory", value="/path/to/databases", help="Path to databases directory.")

        logger.debug(f"Execution settings: af_input_path={af_input_path}, af_output_path={af_output_path}, model_parameters_dir={model_parameters_dir}, databases_dir={databases_dir}")

        # Additional options
        run_data_pipeline = st.checkbox("Run Data Pipeline (CPU only, time-consuming)", value=True)
        run_inference = st.checkbox("Run Inference (requires GPU)", value=True)

        logger.info(f"Run data pipeline: {run_data_pipeline}, Run inference: {run_inference}")

        # Bucket Sizes Configuration
        use_custom_buckets = st.checkbox("Specify Custom Compilation Buckets", value=False)
        if use_custom_buckets:
            buckets_input = st.text_input(
                "Bucket Sizes (comma-separated)",
                value="256,512,768,1024,1280,1536,2048,2560,3072,3584,4096,4608,5120",
                help="Specify bucket sizes separated by commas. Example: 256,512,768,..."
            )
            # Parse buckets
            bucket_sizes = [int(size.strip()) for size in buckets_input.split(",") if size.strip().isdigit()]
            if not bucket_sizes:
                st.error("Please provide at least one valid bucket size.")
                logger.error("No valid bucket sizes provided.")
                st.stop()
            logger.debug(f"Custom bucket sizes: {bucket_sizes}")
        else:
            bucket_sizes = []  # Empty list indicates default buckets

    st.markdown('<div id="run_alphafold"></div>', unsafe_allow_html=True)
    st.header("🚀 Run AlphaFold 3")
    # Save JSON to file
    json_save_path = os.path.join(af_input_path, "fold_input.json")
    try:
        os.makedirs(af_input_path, exist_ok=True)
        with open(json_save_path, "w") as json_file:
            json.dump(alphafold_input, json_file, indent=2)
        st.success(f"JSON file saved to {json_save_path}")
        logger.info(f"JSON file saved to {json_save_path}")
    except Exception as e:
        st.error(f"Error saving JSON file: {e}")
        logger.error(f"Error saving JSON file: {e}")

    # Run AlphaFold 3
    if st.button("Run AlphaFold 3 Now ▶️"):
        # Build the Docker command
        docker_command = (
            f"docker run --rm "
            f"--volume {af_input_path}:/root/af_input "
            f"--volume {af_output_path}:/root/af_output "
            f"--volume {model_parameters_dir}:/root/models "
            f"--volume {databases_dir}:/root/public_databases "
            f"--gpus all "
            f"alphafold3 "
            f"python run_alphafold.py "
            f"--json_path=/root/af_input/fold_input.json "
            f"--model_dir=/root/models "
            f"--output_dir=/root/af_output "
            f"{'--run_data_pipeline' if run_data_pipeline else ''} "
            f"{'--run_inference' if run_inference else ''} "
            f"{'--buckets ' + ','.join(map(str, bucket_sizes)) if bucket_sizes else ''}"
        )

        st.markdown("#### Docker Command:")
        st.code(docker_command, language="bash")
        logger.debug(f"Docker command: {docker_command}")

        # Run the command and display output in a box
        with st.spinner('AlphaFold 3 is running...'):
            output_placeholder = st.empty()
            output = run_alphafold(docker_command, placeholder=output_placeholder)

        # Display the output in an expander box
        st.markdown("#### Command Output:")
        with st.expander("Show Command Output 📄", expanded=False):
            st.text_area("Command Output", value=output, height=400)

        logger.info("AlphaFold 3 execution completed.")

        # Check if the output directory exists
        job_output_folder_name = job_name.lower().replace(' ', '_')
        output_folder_path = os.path.join(af_output_path, job_output_folder_name)

        if os.path.exists(output_folder_path):
            st.success("AlphaFold 3 execution completed successfully.")
            st.info(f"Results are saved in: {output_folder_path}")
            logger.info(f"Results saved in: {output_folder_path}")

            # Provide download option
            st.markdown("### Download Results 📥")
            zip_data = compress_output_folder(output_folder_path, job_output_folder_name)
            st.download_button(
                label="Download ZIP",
                data=zip_data,
                file_name=f"{job_output_folder_name}.zip",
                mime="application/zip"
            )
            logger.info("User downloaded the results ZIP file.")

            # Provide instructions to run the Visualization App
            st.markdown("### Visualize Your Results")
            st.write("To visualize your results, run the following command:")
            st.code(f"afusion visualization --output_folder_path '{output_folder_path}'", language="bash")
            st.write("Or launch the Visualization App and enter the output folder path.")
            logger.info("Provided instructions to the user to run the Visualization App.")

        else:
            st.error("AlphaFold 3 execution did not complete successfully. Please check the logs.")
            logger.error("AlphaFold 3 execution did not complete successfully.")
    else:
        st.info("Click the 'Run AlphaFold 3 Now ▶️' button to execute the command.")

    st.markdown("---")

    # Provide access to the log file
    st.markdown("### Download Log File 📥")
    with open('afusion.log', 'r') as log_file:
        log_content = log_file.read()
    st.download_button(
        label="Download Log File",
        data=log_content,
        file_name='afusion.log',
        mime='text/plain'
    )

    # Display log content in the app
    with st.expander("Show Log Content 📄", expanded=False):
        st.text_area("Log Content", value=log_content, height=200)

    st.markdown("---")
    # Add footer
    st.markdown("<p style='text-align: center; font-size: 12px; color: #95a5a6;'>© 2024 Hanzi. All rights reserved.</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
