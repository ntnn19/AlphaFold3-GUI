# afusion/config.py

# Configuration settings for AlphaFold 3 execution

# Singularity container path
SINGULARITY_CONTAINER = "/data/singularity/alphafold3/alphafold3_v3.0.3_cbdd420.sif"

# Default database paths (these should match the server config)
DEFAULT_DB_DIR = "/var/lib/databases/af3db_sharded/af3db_sharded"
DEFAULT_MODEL_DIR = "/data/singularity/alphafold3/weights"

# Default AlphaFold parameters from server config
DEFAULT_ALPHAFOLDARAMS = {
    "db_dir": DEFAULT_DB_DIR,
    "model_dir": DEFAULT_MODEL_DIR,
    "force_output_dir": True,
    "small_bfd_database_path": "/var/lib/databases/af3db_sharded/af3db_sharded/bfd-first_non_consensus_sequences.fasta.shuffled.split/bfd-first_non_consensus_sequences.fasta@64",
    "small_bfd_z_value": 65984053,
    "mgnify_database_path": "/var/lib/databases/af3db_sharded/af3db_sharded/mgy_clusters_2022_05.fa.shuffled.split/mgy_clusters_2022_05.fa@512",
    "mgnify_z_value": 623796864,
    "uniprot_cluster_annot_database_path": "/var/lib/databases/af3db_sharded/af3db_sharded/uniprot_all_2021_04.fa.shuffled.split/uniprot_all_2021_04.fa@256",
    "uniprot_cluster_annot_z_value": 225619586,
    "uniref90_database_path": "/var/lib/databases/af3db_sharded/af3db_sharded/uniref90_2022_05.fa.shuffled.split/uniref90_2022_05.fa@128",
    "uniref90_z_value": 153742194,
    "ntrna_database_path": "/var/lib/databases/af3db_sharded/af3db_sharded/nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta.shuffled.split/nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta@256",
    "ntrna_z_value": 76752.808514,
    "rfam_database_path": "/var/lib/databases/af3db_sharded/af3db_sharded/rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta.shuffled.split/rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta@16",
    "rfam_z_value": 138.115553,
    "rna_central_database_path": "/var/lib/databases/af3db_sharded/af3db_sharded/rnacentral_active_seq_id_90_cov_80_linclust.fasta.shuffled.split/rnacentral_active_seq_id_90_cov_80_linclust.fasta@64",
    "rna_central_z_value": 13271.415730,
    "jackhmmer_n_cpu": 2,
    "jackhmmer_max_parallel_shards": 16,
    "nhmmer_n_cpu": 2,
    "nhmmer_max_parallel_shards": 16,
}

# Default input/output paths
DEFAULT_AF_INPUT_PATH = "/tmp/af_input"
DEFAULT_AF_OUTPUT_PATH = "/tmp/af_output"
