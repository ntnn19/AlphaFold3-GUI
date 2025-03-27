from setuptools import setup, find_packages

setup(
    name='afusion',
    version='1.3.0',
    author='Han Wang',
    author_email='marspenman@gmail.com',
    description='AFusion: AlphaFold 3 GUI & Toolkit with Visualization',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Hanziwww/AlphaFold3-GUI',
    packages=find_packages(include=['afusion', 'afusion.*']),
    include_package_data=True,
    install_requires=[
        'yaml',  # Direct pip package dependencies
        'streamlit',
        'pandas',
        'loguru>=0.7.2',  # Ensure this matches your specified version
        'numpy',
        'snakemake>=8.14.0',  # Specific version for Snakemake
        'snakemake-executor-plugin-slurm',  # Snakemake executor plugin for Slurm
        'py3Dmol',
        'biopython',
        'plotly',
        'streamlit-authenticator',  # Add streamlit-authenticator
        'git+https://github.com/ntnn19/AlphaFold3-GUI.git@original_main#egg=afusion'
    ],
    entry_points={
        'console_scripts': [
            'afusion = afusion.cli:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: POSIX :: Linux',
    ],
    python_requires='>=3.10',
)
