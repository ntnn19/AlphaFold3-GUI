import streamlit as st
import subprocess
import os
import sys
import time

# Import logging library
from loguru import logger

# Configure the logger
logger.add("install.log", rotation="1 MB", level="DEBUG")

# Set page configuration and theme
st.set_page_config(
    page_title="AlphaFold 3 Installation Guide",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown("""
    <style>
    /* Center the title */
    .title {
        text-align: center;
    }
    /* Button styling */
    .stButton>button {
        background-color: #2c3e50;
        color: white;
        border-radius: 5px;
    }
    /* Code blocks */
    .code-block {
        background-color: #f2f4f5;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and subtitle
st.markdown("<h1 class='title'>🧬 AlphaFold 3 Installation Guide</h1>", unsafe_allow_html=True)
st.markdown("<p class='title'>Please follow the steps below to install AlphaFold 3</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 14px;'>If this project helps you, please ⭐️ <a href='https://github.com/Hanziwww/AlphaFold3-GUI' target='_blank'>my project</a>!</p>", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("---")
steps = {
    "1️⃣ Environment Preparation": "env_prep",
    "2️⃣ Install Docker": "install_docker",
    "3️⃣ Install Singularity": "install_singularity",
    "4️⃣ Install NVIDIA Drivers": "install_nvidia",
    "5️⃣ Download AlphaFold 3 Source Code": "download_code",
    "6️⃣ Obtain Genetic Databases": "download_db",
    "7️⃣ Obtain Model Parameters": "obtain_models",
    "8️⃣ Build Docker Container": "build_docker",
    "9️⃣ Run Test": "run_test",
    "1️⃣0️⃣Run singularity Test": "run_singularity_test",
}
for step_name, step_id in steps.items():
    st.sidebar.markdown(f"<a href='#{step_id}' style='text-decoration: none;'>{step_name}</a>", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown("<small>© 2024 Your Name. All rights reserved.</small>", unsafe_allow_html=True)

# Main content
st.markdown("## 1️⃣ Environment Preparation", unsafe_allow_html=True)
st.markdown('<div id="env_prep"></div>', unsafe_allow_html=True)

st.markdown("Before getting started, please ensure that your system meets the following requirements:")
st.markdown("""
- Running **Linux** operating system (Ubuntu 22.04 LTS is recommended)
- **NVIDIA GPU** installed with Compute Capability **8.0** or higher
- At least **64GB** of RAM
- At least **1TB** of disk space (**SSD** is recommended)
""")

with st.expander("Check System Environment", expanded=False):
    st.markdown("Click the button below to check your system environment.")
    if st.button("Check Environment", key="env_check"):
        # Check Operating System
        logger.info("Checking Operating System...")
        os_info = subprocess.getoutput("lsb_release -a")
        st.markdown("#### Operating System Info:")
        st.code(os_info, language="bash")
        logger.debug(f"Operating System Info:\n{os_info}")

        # Check GPU
        logger.info("Checking GPU...")
        gpu_info = subprocess.getoutput("nvidia-smi")
        st.markdown("#### GPU Info:")
        st.code(gpu_info, language="bash")
        logger.debug(f"GPU Info:\n{gpu_info}")

        # Check Memory
        logger.info("Checking Memory...")
        mem_info = subprocess.getoutput("free -h")
        st.markdown("#### Memory Info:")
        st.code(mem_info, language="bash")
        logger.debug(f"Memory Info:\n{mem_info}")

        # Check Disk Space
        logger.info("Checking Disk Space...")
        disk_info = subprocess.getoutput("df -h")
        st.markdown("#### Disk Space Info:")
        st.code(disk_info, language="bash")
        logger.debug(f"Disk Space Info:\n{disk_info}")

st.markdown("## 2️⃣ Install Docker", unsafe_allow_html=True)
st.markdown('<div id="install_docker"></div>', unsafe_allow_html=True)

st.markdown("AlphaFold 3 depends on Docker for execution. Next, we'll install and configure Docker.")

with st.expander("Install Docker", expanded=False):
    st.markdown("Click the button below to begin installing Docker.")
    st.warning("**Note:** Some commands require `sudo` privileges. Please run these commands in your terminal when prompted.")
    if st.button("Install Docker", key="install_docker"):
        st.info("Please run the following commands in your terminal:")
        docker_commands = """
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \\
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \\
  https://download.docker.com/linux/ubuntu \\
  $(lsb_release -cs) stable" | \\
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io
"""
        st.code(docker_commands, language="bash")
        st.info("After running the above commands, click the 'Verify Docker Installation' button below.")
    if st.button("Verify Docker Installation", key="verify_docker"):
        try:
            docker_version = subprocess.getoutput("docker --version")
            st.success(f"Docker installed successfully! {docker_version}")
            logger.info(f"Docker installed successfully! {docker_version}")
        except Exception as e:
            st.error("Docker is not installed correctly. Please ensure you've run the commands above.")
            logger.error(f"Docker installation verification error: {e}")


st.markdown("## 3️⃣ Install Singularity", unsafe_allow_html=True)
st.markdown('<div id="install_singularity"></div>', unsafe_allow_html=True)

st.markdown("The automated workflow depends on Singularity for execution. Next, we'll install and configure Singularity.")

with st.expander("Install SingularityCE (optional. Required for the automated workflow)", expanded=False):
    st.markdown("Click the button below to begin installing SingularityCE from source.")
    st.warning("**Note:** Some commands require `sudo` privileges. Please run these commands in your terminal when prompted.")
    if st.button("Install SingularityCE", key="install_singularity"):
        st.info("Please run the following commands in your terminal:")
        singularity_commands = """
# Install system dependencies
sudo apt-get update
sudo apt-get install -y \
    autoconf \
    automake \
    cryptsetup \
    fuse2fs \
    git \
    fuse \
    libfuse-dev \
    libseccomp-dev \
    libtool \
    pkg-config \
    runc \
    squashfs-tools \
    squashfs-tools-ng \
    uidmap \
    wget \
    zlib1g-dev

# For Ubuntu 24.04+ install additional dependencies
sudo apt-get install -y libsubid-dev

# Install Go (if not already installed)
export VERSION=1.24.0 OS=linux ARCH=amd64  # Change version if needed
wget -O /tmp/go${VERSION}.${OS}-${ARCH}.tar.gz \
  https://dl.google.com/go/go${VERSION}.${OS}-${ARCH}.tar.gz
sudo tar -C /usr/local -xzf /tmp/go${VERSION}.${OS}-${ARCH}.tar.gz
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
source ~/.bashrc

# Clone SingularityCE repo
git clone --recurse-submodules https://github.com/sylabs/singularity.git
cd singularity

# Build and install SingularityCE
./mconfig
make -C builddir
sudo make -C builddir install

# Verify the installation
singularity --version
"""
        st.code(singularity_commands, language="bash")
        st.info("After running the above commands, you should have SingularityCE installed. Check the installation with 'singularity --version'.")

st.markdown("## 4️⃣ Install NVIDIA Drivers", unsafe_allow_html=True)
st.markdown('<div id="install_nvidia"></div>', unsafe_allow_html=True)

st.markdown("Next, we'll install the appropriate NVIDIA drivers for your GPU.")

with st.expander("Install NVIDIA Drivers", expanded=False):
    st.markdown("Click the button below to begin installing NVIDIA drivers.")
    st.warning("**Note:** This step requires `sudo` privileges. Please run the commands in your terminal when prompted.")
    if st.button("Install NVIDIA Drivers", key="install_nvidia"):
        st.info("Please run the following commands in your terminal:")
        nvidia_commands = """
sudo apt-get update
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers install
"""
        st.code(nvidia_commands, language="bash")
        st.info("After running the above commands, reboot your system and then click the 'Verify NVIDIA Installation' button below.")
    if st.button("Verify NVIDIA Installation", key="verify_nvidia"):
        try:
            nvidia_info = subprocess.getoutput("nvidia-smi")
            if "NVIDIA-SMI" in nvidia_info:
                st.success("NVIDIA drivers installed successfully!")
                st.code(nvidia_info, language="bash")
                logger.info("NVIDIA drivers installed successfully!")
            else:
                st.error("NVIDIA drivers are not installed correctly. Please ensure you've run the commands above and rebooted your system.")
                logger.error("NVIDIA drivers verification failed.")
        except Exception as e:
            st.error(f"An error occurred while verifying NVIDIA drivers: {e}")
            logger.error(f"NVIDIA drivers verification error: {e}")

st.markdown("## 5️⃣Download AlphaFold 3 Source Code", unsafe_allow_html=True)
st.markdown('<div id="download_code"></div>', unsafe_allow_html=True)

st.markdown("Now, we'll clone the AlphaFold 3 source code from GitHub.")

with st.expander("Download Source Code", expanded=False):
    st.markdown("Click the button below to begin downloading the AlphaFold 3 source code.")
    st.warning("**Note:** Some commands may require `sudo` privileges. Please run these commands in your terminal when prompted.")
    if st.button("Download Source Code", key="clone_code"):
        st.info("Please run the following commands in your terminal:")
        git_commands = """
sudo apt-get install -y git
git clone https://github.com/google-deepmind/alphafold3.git
"""
        st.code(git_commands, language="bash")
        st.info("After running the above commands, ensure the 'alphafold3' directory has been created.")

st.markdown("## 6️⃣Obtain Genetic Databases", unsafe_allow_html=True)
st.markdown('<div id="download_db"></div>', unsafe_allow_html=True)

st.markdown("AlphaFold 3 requires multiple genetic databases to run. Downloading and setting up these databases may take some time.")

with st.expander("Download Genetic Databases", expanded=False):
    st.markdown("Click the button below to begin downloading the genetic databases.")
    db_dir = st.text_input("Please enter the database download directory (absolute path):", value=f"{os.path.expanduser('~')}/alphafold3_databases")
    st.warning("**Note:** This step requires `sudo` privileges. Please run the commands in your terminal when prompted.")
    if st.button("Download Databases", key="download_db"):
        st.info("Please run the following commands in your terminal:")
        db_commands = f"""
sudo apt-get install -y wget zstd
cd alphafold3  # Navigate to the AlphaFold 3 directory
./fetch_databases.sh {db_dir}
"""
        st.code(db_commands, language="bash")
        st.info("This process may take a long time. Please be patient and ensure you have sufficient disk space.")

st.markdown("## 7️⃣ Obtain Model Parameters", unsafe_allow_html=True)
st.markdown('<div id="obtain_models"></div>', unsafe_allow_html=True)

st.markdown("The AlphaFold 3 model parameters require requesting access.")

with st.expander("Request Access to Model Parameters", expanded=False):
    st.markdown("Please [**click here**](https://forms.gle/svvpY4u2jsHEwWYS6) to fill out the form to request access to AlphaFold 3 model parameters.")
    st.markdown("Once you have obtained the model parameters, please place them in an appropriate directory.")

st.markdown("## 8️⃣Build Docker Container", unsafe_allow_html=True)
st.markdown('<div id="build_docker"></div>', unsafe_allow_html=True)

st.markdown("Now, we'll build the Docker container for AlphaFold 3.")

with st.expander("Build Docker Container", expanded=False):
    st.markdown("Click the button below to begin building the Docker container. This may take some time. Please be patient.")
    st.warning("**Note:** This step may require `sudo` privileges. Please run the commands in your terminal when prompted.")
    if st.button("Build Docker Container", key="build_docker"):
        st.info("Please run the following commands in your terminal:")
        build_commands = """
cd alphafold3  # Navigate to the AlphaFold 3 directory
sudo docker build -t alphafold3 -f docker/Dockerfile .
"""
        st.code(build_commands, language="bash")
        st.info("After running the above commands, the Docker container should be built successfully.")

st.markdown("## 9️⃣Run Docker Test", unsafe_allow_html=True)
st.markdown('<div id="run_test"></div>', unsafe_allow_html=True)

st.markdown("Finally, we'll run a simple test to verify that AlphaFold 3 has been installed successfully.")

with st.expander("Run Docker Test", expanded=False):
    st.markdown("Click the button below to begin the test.")
    input_dir = st.text_input("Please enter the input directory (absolute path):", value=f"{os.path.expanduser('~')}/af_input")
    output_dir = st.text_input("Please enter the output directory (absolute path):", value=f"{os.path.expanduser('~')}/af_output")
    model_dir = st.text_input("Please enter the model parameters directory (absolute path):", value=f"{os.path.expanduser('~')}/alphafold3_models")
    db_dir = st.text_input("Please enter the database directory (absolute path):", value=f"{os.path.expanduser('~')}/alphafold3_databases")
    st.warning("**Note:** This step may require `sudo` privileges. Please run the commands in your terminal when prompted.")
    if st.button("Run Test", key="run_test"):
        st.info("Please ensure that you have prepared an appropriate input JSON file in the input directory.")
        st.info("Please run the following command in your terminal:")
        test_command = f"""
sudo docker run -it \\
    --volume {input_dir}:/root/af_input \\
    --volume {output_dir}:/root/af_output \\
    --volume {model_dir}:/root/models \\
    --volume {db_dir}:/root/public_databases \\
    --gpus all \\
    alphafold3 \\
    python run_alphafold.py \\
    --json_path=/root/af_input/fold_input.json \\
    --model_dir=/root/models \\
    --output_dir=/root/af_output
"""
        st.code(test_command, language="bash")
        st.info("After running the above command, the test should execute. Please check the output directory for results.")


st.markdown("## 1️⃣0️⃣Run Singularity Test", unsafe_allow_html=True)
st.markdown('<div id="run_singularity_test"></div>', unsafe_allow_html=True)
st.markdown(
    "Finally, we'll run a simple test to verify that the Singularity container can run a Docker image successfully.")

with st.expander("Run Singularity Test with Docker Image", expanded=False):
    st.markdown("Click the button below to begin the test.")
    input_dir = st.text_input("Please enter the input directory (absolute path):",
                              value=f"{os.path.expanduser('~')}/af_input", key="input_dir")
    output_dir = st.text_input("Please enter the output directory (absolute path):",
                               value=f"{os.path.expanduser('~')}/af_output", key="output_dir")
    model_dir = st.text_input("Please enter the model parameters directory (absolute path):",
                              value=f"{os.path.expanduser('~')}/alphafold3_models", key="model_dir")
    db_dir = st.text_input("Please enter the database directory (absolute path):",
                           value=f"{os.path.expanduser('~')}/alphafold3_databases", key="db_dir")
    st.warning(
        "**Note:** This step may require `sudo` privileges. Please run the commands in your terminal when prompted.")

    if st.button("Run Test", key="run_singularity_test"):
        st.info("Please ensure that you have prepared an appropriate input JSON file in the input directory.")
        st.info("Please run the following command in your terminal:")
        test_command = f"""
 singularity run \\
    --bind {input_dir}:/root/af_input \\
    --bind {output_dir}:/root/af_output \\
    --bind {model_dir}:/root/models \\
    --bind {db_dir}:/root/public_databases \\
    docker://alphafold3 \\
    python run_alphafold.py \\
    --json_path=/root/af_input/fold_input.json \\
    --model_dir=/root/models \\
    --output_dir=/root/af_output
"""
        st.code(test_command, language="bash")
        st.info(
            "After running the above command, the test should execute. Please check the output directory for results.")

st.markdown("<p style='text-align: center;'>🎉 Congratulations, AlphaFold 3 has been successfully installed!</p>", unsafe_allow_html=True)
st.markdown("---")

# Provide access to the log file
st.markdown("### Download Log File 📥")
if os.path.exists('install.log'):
    with open('install.log', 'r') as log_file:
        log_content = log_file.read()
    st.download_button(
        label="Download Log File",
        data=log_content,
        file_name='install.log',
        mime='text/plain'
    )

    # Display log content in the app
    with st.expander("Show Log Content 📄", expanded=False):
        st.text_area("Log Content", value=log_content, height=200)
else:
    st.info("Log file not found. It will be created after you run some steps in the installation.")
st.markdown("---")
# Add footer
st.markdown("<p style='text-align: center; font-size: 12px; color: #95a5a6;'>© 2024 Your Name. All rights reserved.</p>", unsafe_allow_html=True)
