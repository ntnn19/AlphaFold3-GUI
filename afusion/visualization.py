# visualization.py
import streamlit as st

from Bio import PDB
import py3Dmol
import io
import json
import numpy as np
import plotly.express as px
import pandas as pd
from loguru import logger
    
from Bio.PDB import *
from scipy.spatial import ConvexHull
from stl import mesh
import warnings
# Configure logger
logger.add("afusion_visualization.log", rotation="1 MB", level="DEBUG")

# ========================================
# Functions that accept file paths
# ========================================

def read_cif_file(file_path):
    parser = PDB.MMCIFParser(QUIET=True)
    with open(file_path, 'r') as file:
        content = file.read()
    file_like = io.StringIO(content)
    structure = parser.get_structure('protein', file_like)
    return structure, content

def extract_pae_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    pae = np.array(data.get("pae", []), dtype=np.float16)  # Use np.float16
    token_chain_ids = data.get("token_chain_ids", [])
    return pae, token_chain_ids

def extract_summary_confidences(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# ========================================
# Functions that accept file objects (uploaded files)
# ========================================

def read_cif_file_obj(file_obj):
    parser = PDB.MMCIFParser(QUIET=True)
    content = file_obj.read().decode('utf-8')
    file_like = io.StringIO(content)
    structure = parser.get_structure('protein', file_like)
    return structure, content

def extract_pae_from_json_obj(file_obj):
    """Extract PAE data from JSON file object."""
    try:
        data = json.load(file_obj)
        # Check if data is a list (sometimes AF2 outputs PAE in different formats)
        if isinstance(data, list):
            pae_data = data[0]  # Take first element if it's a list
        else:
            pae_data = data
            
        # Get PAE matrix and chain IDs
        pae = np.array(pae_data.get("predicted_aligned_error", []), dtype=np.float16)
        token_chain_ids = pae_data.get("max_predicted_aligned_error", [])
        
        if len(pae) == 0:  # If PAE not found in first format, try alternate keys
            pae = np.array(pae_data.get("pae", []), dtype=np.float16)
            token_chain_ids = pae_data.get("token_chain_ids", [])
            
        return pae, token_chain_ids
    except Exception as e:
        st.error(f"Error extracting PAE data: {str(e)}")
        # Return empty data as fallback
        return np.array([[]], dtype=np.float16), []

def extract_summary_confidences_obj(file_obj):
    """Extract summary confidence data from JSON file object."""
    try:
        data = json.load(file_obj)
        # Check if data is a list
        if isinstance(data, list):
            summary_data = data[0]  # Take first element if it's a list
        else:
            summary_data = data
            
        # Convert any numpy arrays to lists for JSON serialization
        processed_data = {}
        for key, value in summary_data.items():
            if isinstance(value, np.ndarray):
                processed_data[key] = value.tolist()
            else:
                processed_data[key] = value
                
        return processed_data
    except Exception as e:
        st.error(f"Error extracting summary confidence data: {str(e)}")
        return {}  # Return empty dict as fallback

# ========================================
# Common functions
# ========================================

def extract_residue_bfactors(structure):
    residue_bfactors = {}
    ligands = []
    for model in structure:
        for chain in model:
            for residue in chain:
                hetfield = residue.get_id()[0]
                resseq = residue.get_id()[1]
                resname = residue.get_resname()
                chain_id = chain.id

                atom_bfactors = []

                for atom in residue:
                    bfactor = atom.get_bfactor()

                    if hetfield.strip() == "":
                        atom_bfactors.append(bfactor)
                    else:
                        # Ligand atoms
                        ligands.append({
                            'chain_id': chain_id,
                            'resseq': resseq,
                            'resname': resname,
                            'atom_name': atom.get_name(),
                            'bfactor': bfactor
                        })

                if hetfield.strip() == "" and atom_bfactors:
                    avg_bfactor = sum(atom_bfactors) / len(atom_bfactors)
                    residue_bfactors[(chain_id, resseq)] = {
                        'avg_bfactor': avg_bfactor,
                        'resname': resname
                    }

    return residue_bfactors, ligands

def get_color_from_bfactor(bfactor):
    # Define color mapping
    color_mapping = [
        {'range': [90, 100], 'color': '#106dff'},   # Very high (pLDDT > 90)
        {'range': [70, 90],  'color': '#10cff1'},   # Confident (90 > pLDDT > 70)
        {'range': [50, 70],  'color': '#f6ed12'},   # Low (70 > pLDDT > 50)
        {'range': [0, 50],   'color': '#ef821e'}    # Very low (pLDDT < 50)
    ]
    for mapping in color_mapping:
        b_min, b_max = mapping['range']
        if b_min <= bfactor < b_max:
            return mapping['color']
    return 'grey'  # Default color

def visualize_structure(residue_bfactors, ligands, cif_content, background_color='#000000'):
    # Create a py3Dmol view
    view = py3Dmol.view(width='100%', height=600)
    view.addModel(cif_content, 'cif')

    # Set default style
    view.setStyle({'model': -1}, {'cartoon': {'color': 'grey'}})

    # Apply coloring based on B-factors
    for (chain_id, resseq), info in residue_bfactors.items():
        avg_bfactor = info['avg_bfactor']
        color = get_color_from_bfactor(avg_bfactor)
        selection = {'chain': chain_id, 'resi': resseq}
        style = {'cartoon': {'color': color}}
        view.addStyle(selection, style)

    # Color ligands
    for ligand in ligands:
        bfactor = ligand['bfactor']
        color = get_color_from_bfactor(bfactor)
        selection = {
            'chain': ligand['chain_id'],
            'resi': ligand['resseq'],
            'atom': ligand['atom_name']
        }
        style = {'stick': {'color': color}}
        view.addStyle(selection, style)

    # Set background color
    view.setBackgroundColor(background_color)

    # Generate HTML content
    view_html = view._make_html()
    return view_html

def display_visualization_header():
    st.write("### Visualization")
    color_mapping_html = """
    <div style='display: flex; flex-wrap: wrap; align-items: center; justify-content: center;'>
        <div style='display: flex; align-items: center; margin: 5px;'>
            <div style='width: 20px; height: 20px; background-color: #106dff;'></div>
            <span style='margin-left: 5px;'>Very high (pLDDT > 90)</span>
        </div>
        <div style='display: flex; align-items: center; margin: 5px;'>
            <div style='width: 20px; height: 20px; background-color: #10cff1;'></div>
            <span style='margin-left: 5px;'>Confident (90 > pLDDT > 70)</span>
        </div>
        <div style='display: flex; align-items: center; margin: 5px;'>
            <div style='width: 20px; height: 20px; background-color: #f6ed12;'></div>
            <span style='margin-left: 5px;'>Low (70 > pLDDT > 50)</span>
        </div>
        <div style='display: flex; align-items: center; margin: 5px;'>
            <div style='width: 20px; height: 20px; background-color: #ef821e;'></div>
            <span style='margin-left: 5px;'>Very low (pLDDT < 50)</span>
        </div>
    </div>
    """
    st.markdown(color_mapping_html, unsafe_allow_html=True)

def visualize_pae(pae_matrix, token_chain_ids):
    """Visualize PAE matrix with error handling."""
    st.write("### Predicted Aligned Error (PAE)")
    
    if len(pae_matrix) == 0:
        st.warning("No PAE data available")
        return
        
    try:
        fig = px.imshow(
            pae_matrix,
            color_continuous_scale='Greens_r',
            labels=dict(
                x="Residue index",
                y="Residue index",
                color="PAE (Å)"
            ),
            zmin=0.0,
            zmax=31.75
        )
        
        # Draw chain boundaries if available
        if token_chain_ids:
            chain_boundaries = []
            prev_chain = token_chain_ids[0]
            for idx, chain_id in enumerate(token_chain_ids):
                if chain_id != prev_chain:
                    chain_boundaries.append(idx - 0.5)
                    prev_chain = chain_id
            
            for boundary in chain_boundaries:
                fig.add_shape(
                    type="line",
                    x0=boundary,
                    y0=0,
                    x1=boundary,
                    y1=len(token_chain_ids),
                    line=dict(color="red", width=1)
                )
                fig.add_shape(
                    type="line",
                    x0=0,
                    y0=boundary,
                    x1=len(token_chain_ids),
                    y1=boundary,
                    line=dict(color="red", width=1)
                )
                
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error visualizing PAE matrix: {str(e)}")

def display_summary_data(summary_data, chain_ids):
    st.write("### Summary of Confidence Metrics")

    # Map chain-level metrics to chain IDs
    chain_metrics = {}
    for key in ['chain_iptm', 'chain_ptm']:
        if key in summary_data:
            values = summary_data[key]
            if len(values) == len(chain_ids):
                chain_metrics[key] = dict(zip(chain_ids, values))
            else:
                st.warning(f"The number of values in {key} does not match the number of chains.")

    # Display chain-level metrics
    for key, metrics in chain_metrics.items():
        st.write(f"#### {key}")
        df = pd.DataFrame.from_dict(metrics, orient='index', columns=[key])
        # Format numbers to two decimal places
        df_style = df.style.format("{:.2f}").set_table_styles(
            [{'selector': 'th, td', 'props': [('border', '1px solid black')]}]
        ).set_properties(**{'text-align': 'center'})
        # Center the table and set a fixed width
        st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
        st.table(df_style)
        st.markdown("</div>", unsafe_allow_html=True)

    # Display pair metrics as matrices and visualize them
    for key in ['chain_pair_iptm', 'chain_pair_pae_min']:
        pair_metrics = summary_data.get(key, {})
        if pair_metrics:
            st.write(f"#### {key}")
            if len(pair_metrics) == len(chain_ids):
                df = pd.DataFrame(pair_metrics, index=chain_ids, columns=chain_ids)
                # Format numbers to two decimal places
                df_style = df.style.format("{:.2f}").set_table_styles(
                    [{'selector': 'th, td', 'props': [('border', '1px solid black')]}]
                ).set_properties(**{'text-align': 'center'})
                # Create two columns
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
                    st.table(df_style)
                    st.markdown("</div>", unsafe_allow_html=True)
                with col2:
                    # Visualize the matrix
                    fig = px.imshow(
                        df,
                        x=chain_ids,
                        y=chain_ids,
                        color_continuous_scale='Viridis',
                        text_auto=".2f",
                        labels={'x': 'Chain', 'y': 'Chain', 'color': key}
                    )
                    fig.update_layout(autosize=True)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"The dimensions of {key} do not match the number of chains.")

    # Display other metrics
    other_metrics = {k: v for k, v in summary_data.items() if k not in ['chain_iptm', 'chain_ptm', 'chain_pair_iptm', 'chain_pair_pae_min']}
    if other_metrics:
        st.write("#### Other Metrics")
        df = pd.DataFrame(list(other_metrics.items()), columns=['Metric', 'Value'])
        # Format 'Value' column to two decimal places if numeric
        df['Value'] = df['Value'].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
        df_style = df.style.set_table_styles(
            [{'selector': 'th, td', 'props': [('border', '1px solid black')]}]
        ).set_properties(**{'text-align': 'center'})
        # Center the table and set a fixed width
        st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
        st.table(df_style)
        st.markdown("</div>", unsafe_allow_html=True)



def extract_sequence(structure):
    """Extract amino acid sequence from structure."""
    aa_dict = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }
    
    sequence = {}
    try:
        # Get first model if it's a structure object
        if isinstance(structure, PDB.Structure.Structure):
            models = structure.get_list()
            if models:
                structure = models[0]
        
        # Extract sequence for each chain
        for chain in structure.get_chains():
            chain_seq = ""
            for residue in chain:
                if residue.id[0] == " ":  # Only standard amino acids
                    resname = residue.get_resname()
                    if resname in aa_dict:
                        chain_seq += aa_dict[resname]
            if chain_seq:  # Only add chains that have a sequence
                sequence[chain.id] = chain_seq
    except Exception as e:
        st.error(f"Error extracting sequence: {str(e)}")
        return {"A": ""}  # Return dummy sequence if extraction fails
    
    return sequence

def format_sequence(sequence, width=80):
    """Format sequence with line numbers and fixed width."""
    lines = []
    for i in range(0, len(sequence), width):
        chunk = sequence[i:i + width]
        line_num = str(i + 1).rjust(5)
        lines.append(f"{line_num} {chunk}")
    return '\n'.join(lines)

def add_visualization_controls(structure):
    """Add visualization controls to the sidebar."""
    st.sidebar.markdown("### Visualization Controls")
    
    # Style selection
    style = st.sidebar.selectbox(
        "Select Style",
        ["cartoon", "stick", "line", "sphere"],
        help="Choose the visualization style for the protein structure"
    )
    
    # Color scheme selection
    color_scheme = st.sidebar.selectbox(
        "Color Scheme",
        ["confidence", "chain", "secondary", "rainbow", "custom"],
        help="Choose how to color the structure"
    )
    
    # Custom color if selected
    custom_color = None
    if color_scheme == "custom":
        custom_color = st.sidebar.color_picker("Choose custom color", "#00FF00")
    
    # Extract and display sequence
    try:
        sequences = extract_sequence(structure)
        if not sequences:
            st.sidebar.warning("No sequence information available")
            return {
                'style': style,
                'color_scheme': color_scheme,
                'custom_color': custom_color,
                'selected_residues': None,
                'selection_color': "#FF0000"
            }
        
        # Sequence selection controls
        st.sidebar.markdown("### Sequence Selection")
        
        # Chain selection
        selected_chain = st.sidebar.selectbox(
            "Select Chain",
            list(sequences.keys()),
            help="Choose which chain to select residues from"
        )
        
        # Display selected chain sequence
        st.sidebar.markdown("#### Chain Sequence")
        formatted_seq = format_sequence(sequences[selected_chain])
        st.sidebar.code(formatted_seq)
        
        # Residue selection
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_res = st.number_input("Start Residue", 1, len(sequences[selected_chain]), 1)
        with col2:
            end_res = st.number_input("End Residue", start_res, len(sequences[selected_chain]), 
                                     min(start_res + 10, len(sequences[selected_chain])))
        
        # Selection color
        selection_color = st.sidebar.color_picker("Selection Color", "#FF0000")
        
        selected_residues = None
        if st.sidebar.button("Highlight Selection"):
            selected_residues = {
                selected_chain: list(range(start_res, end_res + 1))
            }
            
        return {
            'style': style,
            'color_scheme': color_scheme,
            'custom_color': custom_color,
            'selected_residues': selected_residues,
            'selection_color': selection_color
        }
        
    except Exception as e:
        st.sidebar.error(f"Error setting up visualization controls: {str(e)}")
        return {
            'style': style,
            'color_scheme': color_scheme,
            'custom_color': custom_color,
            'selected_residues': None,
            'selection_color': "#FF0000"
        }
def visualize_structure(residue_bfactors, ligands, cif_content, 
                       style="cartoon", color_scheme="confidence", 
                       custom_color=None, selected_residues=None, 
                       selection_color="#FF0000", background_color='#000000'):
    """Enhanced structure visualization with multiple styles and coloring options."""
    view = py3Dmol.view(width='100%', height=600)
    view.addModel(cif_content, 'cif')
    
    # Set base style
    style_settings = {}
    if style == "cartoon":
        style_settings = {"cartoon": {}}
    elif style == "stick":
        style_settings = {"stick": {}}
    elif style == "line":
        style_settings = {"line": {}}
    elif style == "sphere":
        style_settings = {"sphere": {}}
    
    # Apply base style
    view.setStyle({'model': -1}, style_settings)
    
    # Apply coloring based on scheme
    style_key = list(style_settings.keys())[0]  # Get the current style type
    
    if color_scheme == "confidence":
        # Use existing B-factor based coloring
        for (chain_id, resseq), info in residue_bfactors.items():
            avg_bfactor = info['avg_bfactor']
            color = get_color_from_bfactor(avg_bfactor)
            selection = {'chain': chain_id, 'resi': resseq}
            view.addStyle(selection, {style_key: {'color': color}})
    
    elif color_scheme == "chain":
        # Color by chain
        colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff']
        chains = set(chain_id for (chain_id, _) in residue_bfactors.keys())
        for i, chain in enumerate(chains):
            color = colors[i % len(colors)]
            view.addStyle({'chain': chain}, {style_key: {'color': color}})
    
    elif color_scheme == "secondary":
        # Color by secondary structure
        view.addStyle({'ss': 'h'}, {style_key: {'color': '#FF0000'}})  # helices
        view.addStyle({'ss': 's'}, {style_key: {'color': '#FFFF00'}})  # sheets
        view.addStyle({'ss': 'l'}, {style_key: {'color': '#00FF00'}})  # loops
    
    elif color_scheme == "rainbow":
        view.setStyle({}, {style_key: {'color': 'spectrum'}})
    
    elif color_scheme == "custom" and custom_color:
        view.setStyle({}, {style_key: {'color': custom_color}})
    
    # Highlight selected residues if any
    if selected_residues:
        for chain_id, residues in selected_residues.items():
            view.addStyle(
                {'chain': chain_id, 'resi': residues},
                {style_key: {'color': selection_color}}
            )
    
    # Add ligands
    for ligand in ligands:
        selection = {
            'chain': ligand['chain_id'],
            'resi': ligand['resseq'],
            'atom': ligand['atom_name']
        }
        view.addStyle(selection, {'stick': {'color': get_color_from_bfactor(ligand['bfactor'])}})
    
    # Set background color and view options
    view.setBackgroundColor(background_color)
    view.zoomTo()
    
    return view._make_html()

def add_export_controls():
    """Add export controls to the sidebar."""
    st.sidebar.markdown("### Export 3D Model")
    export_format = st.sidebar.selectbox(
        "Export Format",
        ["GLTF/GLB (3D Web)", "VRML (WRL)", "STL", "OBJ"],
        help="Choose format to export the 3D model"
    )
    return export_format

def export_structure(structure, format_type="pdb"):
    """
    Export structure in various formats with proper error handling and data conversion.
    
    Args:
        structure: Bio.PDB Structure object
        format_type: str, one of ["pdb", "mmcif", "stl"]
        
    Returns:
        tuple: (binary_data, filename, mime_type) or (None, None, None) on error
    """
    try:
        import io
        from Bio.PDB import PDBIO, MMCIFWriter
        
        # Create in-memory buffer
        buffer = io.StringIO()
        
        if format_type == "pdb":
            # Export as PDB
            io_handler = PDBIO()
            io_handler.set_structure(structure)
            io_handler.save(buffer)
            return (buffer.getvalue().encode('utf-8'), 
                   "structure.pdb", 
                   "chemical/x-pdb")
                   
        elif format_type == "mmcif":
            # Export as mmCIF
            writer = MMCIFWriter()
            writer.save(structure, buffer)
            return (buffer.getvalue().encode('utf-8'),
                   "structure.cif",
                   "chemical/x-cif")
                   
        elif format_type == "stl":
            try:
                import numpy as np
                from stl import mesh
                
                # Get atom coordinates
                atoms = []
                for model in structure:
                    for chain in model:
                        for residue in chain:
                            for atom in residue:
                                atoms.append(atom.get_coord())
                
                if not atoms:
                    raise ValueError("No atoms found in structure")
                
                # Convert to numpy array
                points = np.array(atoms)
                
                # Create mesh using convex hull
                from scipy.spatial import ConvexHull
                hull = ConvexHull(points)
                
                # Create faces from convex hull
                vertices = points[hull.vertices]
                faces = []
                for simplex in hull.simplices:
                    faces.append([simplex[0], simplex[1], simplex[2]])
                
                # Create STL mesh
                stl_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
                for i, face in enumerate(faces):
                    for j in range(3):
                        stl_mesh.vectors[i][j] = vertices[face[j]]
                
                # Save to binary buffer
                buffer = io.BytesIO()
                stl_mesh.save(buffer, mode=mesh.Mode.BINARY)
                
                return (buffer.getvalue(),
                       "structure.stl",
                       "model/stl")
                       
            except ImportError:
                print("Required packages for STL export not found. Install numpy-stl and scipy.")
                return None, None, None
            except Exception as e:
                print(f"Error creating STL mesh: {str(e)}")
                return None, None, None
                
        else:
            raise ValueError(f"Unsupported format: {format_type}")
            
    except Exception as e:
        print(f"Error exporting structure: {str(e)}")
        return None, None, None
    
def export_to_3d_formats(cif_content):
    """Export structure to PDB format."""
    try:
        # Create a temporary file to store the CIF content
        with open("temp.cif", "w") as f:
            f.write(cif_content)
        
        # Use Bio.PDB to read and convert the structure
        parser = PDB.MMCIFParser()
        structure = parser.get_structure("protein", "temp.cif")
        
        # Write as PDB
        io = PDB.PDBIO()
        io.set_structure(structure)
        io.save("temp.pdb")
        
        # Read the converted PDB file
        with open("temp.pdb", "r") as f:
            pdb_data = f.read()
            
        # Clean up temporary files
        import os
        os.remove("temp.cif")
        os.remove("temp.pdb")
        
        return pdb_data.encode('utf-8')  # Return as bytes
        
    except Exception as e:
        st.error(f"Error converting to PDB format: {str(e)}")
        return None

def export_for_cad(cif_content):
    """Export structure for CAD software."""
    try:
        # Create a temporary file to store the CIF content
        with open("temp.cif", "w") as f:
            f.write(cif_content)
        
        # Use Bio.PDB to read the structure
        parser = PDB.MMCIFParser()
        structure = parser.get_structure("protein", "temp.cif")
        
        # Write as STL using Bio.PDB
        # Note: This is a simplified surface representation
        io = PDB.PDBIO()
        io.set_structure(structure)
        io.save("temp.stl")
        
        # Read the STL file as binary
        with open("temp.stl", "rb") as f:
            stl_data = f.read()
            
        # Clean up temporary files
        import os
        os.remove("temp.cif")
        os.remove("temp.stl")
        
        return stl_data
        
    except Exception as e:
        st.error(f"Error converting to STL format: {str(e)}")
        return None
    
def convert_to_stl(structure, style="cartoon"):
    """Convert structure to STL format with proper binary data handling.
    
    Args:
        structure: Bio.PDB Structure object
        style: String indicating visualization style ("cartoon" or "surface")
        
    Returns:
        bytes: Binary STL data
    """
    try:
        import numpy as np
        from Bio.PDB import PDBIO
        import io
        
        # First try mesh-based approach (more reliable)
        try:
            from stl import mesh
            from scipy.spatial import ConvexHull
            
            # Get atom coordinates
            atoms = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            atoms.append(atom.get_coord())
            
            if not atoms:
                raise ValueError("No atoms found in structure")
            
            # Convert to numpy array
            points = np.array(atoms)
            
            # Create a surface mesh using convex hull
            hull = ConvexHull(points)
            
            # Create mesh from convex hull
            vertices = points[hull.vertices]
            faces = []
            for simplex in hull.simplices:
                faces.append([simplex[0], simplex[1], simplex[2]])
            
            # Create STL mesh
            stl_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
            for i, face in enumerate(faces):
                for j in range(3):
                    stl_mesh.vectors[i][j] = vertices[face[j]]
            
            # Save to binary buffer
            buffer = io.BytesIO()
            stl_mesh.save(buffer, mode=mesh.Mode.BINARY)
            return buffer.getvalue()
            
        except ImportError:
            # If numpy-stl is not available, try py3Dmol
            import py3Dmol
            
            # Save structure as PDB first
            pdb_buffer = io.StringIO()
            pdb_io = PDBIO()
            pdb_io.set_structure(structure)
            pdb_io.save(pdb_buffer)
            
            # Create py3Dmol view
            view = py3Dmol.view(width=800, height=600)
            view.addModel(pdb_buffer.getvalue(), "pdb")
            
            # Set style
            if style == "cartoon":
                view.setStyle({'': {'cartoon': {'color': 'spectrum'}}})
            else:  # surface
                view.setStyle({'': {'surface': {'opacity': 0.8}}})
            
            # Get binary STL data
            stl_data = view.zoomTo()._convert_to_binary('stl')
            if not isinstance(stl_data, bytes):
                raise ValueError("Failed to get binary STL data from py3Dmol")
                
            return stl_data
            
    except Exception as e:
        print(f"Error converting to STL: {str(e)}")
        return None
    


def create_protein_mesh(structure, style="surface", resolution=2.0):
    """Creates a detailed 3D mesh of a protein structure suitable for CAD software."""
    try:
        import numpy as np
        from scipy.spatial import ConvexHull
        from stl import mesh
        import io

        # Get atom coordinates and radii
        atoms = []
        atom_radii = []
        
        # Van der Waals radii for common atoms (in Angstroms)
        vdw_radii = {
            'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80,
            'P': 1.80, 'H': 1.20, 'F': 1.47, 'Cl': 1.75
        }

        # Collect atoms based on style
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        if style == "cartoon" and atom.get_name() not in ['CA', 'C', 'N', 'O']:
                            continue
                        if atom.element != 'H':  # Skip hydrogens
                            atoms.append(atom.get_coord())
                            atom_radii.append(vdw_radii.get(atom.element, 1.70))

        if not atoms:
            raise ValueError("No atoms found in structure")

        points = np.array(atoms)
        radii = np.array(atom_radii)

        # Generate surface points
        surface_points = []
        phi = np.linspace(0, 2*np.pi, int(20/resolution))
        theta = np.linspace(0, np.pi, int(10/resolution))
        phi, theta = np.meshgrid(phi, theta)
        
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        sphere_points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

        # Create surface for each atom
        for coord, radius in zip(points, radii):
            atom_surface = sphere_points * radius + coord
            surface_points.extend(atom_surface)

        surface_points = np.array(surface_points)
        hull = ConvexHull(surface_points)
        
        # Create mesh
        faces = hull.simplices
        vertices = surface_points[hull.vertices]
        
        # Generate STL mesh
        stl_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
        for i, face in enumerate(faces):
            for j in range(3):
                stl_mesh.vectors[i][j] = vertices[face[j]]

        # Save to binary buffer
        buffer = io.BytesIO()
        stl_mesh.save(buffer, mode=mesh.Mode.BINARY)
        return buffer.getvalue()

    except Exception as e:
        print(f"Error creating protein mesh: {str(e)}")
        return None


def export_protein_for_cad(structure):
    """
    Creates UI elements for CAD export with different options.
    
    Args:
        structure: Bio.PDB Structure object
    Returns:
        tuple: (success, UI_elements)
    """
    import streamlit as st
    
    st.write("### Export for CAD/3D Printing")
    
    # Export options
    model_type = st.selectbox(
        "Model Type",
        ["Surface (All Atoms)", "Backbone Only"],
        help="Choose what parts of the protein to include in the 3D model"
    )
    
    resolution = st.slider(
        "Resolution",
        min_value=1.0,
        max_value=5.0,
        value=2.0,
        step=0.5,
        help="Higher resolution means more detail but larger file size"
    )
    
    if st.button("Generate 3D Model"):
        with st.spinner("Generating 3D model... This may take a moment."):
            mode = "backbone" if model_type == "Backbone Only" else "surface"
            stl_data = create_protein_mesh(structure, mode=mode, resolution=resolution)
            
            if stl_data:
                st.success("3D model generated successfully!")
                st.download_button(
                    "Download STL File",
                    data=stl_data,
                    file_name="protein_model.stl",
                    mime="model/stl",
                    help="Download for use in CAD software or 3D printing"
                )
                
                st.info("""
                ℹ️ This STL file can be opened in:
                - CAD software (Fusion 360, SolidWorks, AutoCAD)
                - 3D printing slicers (Cura, PrusaSlicer)
                - 3D modeling tools (Blender, Maya)
                
                Tips for best results:
                - Import at 1:1 scale
                - Units are in Angstroms (Å)
                - You may need to scale the model in your CAD software
                - For 3D printing, consider supports for overhanging regions
                """)
                return True
            else:
                st.error("""
                Failed to generate 3D model. Try:
                - Reducing the resolution
                - Using backbone-only mode
                - Checking if the structure has valid coordinates
                """)
                return False
                
    return False


# ========================================
# Main Application
# ========================================

def main():
    # Set page configuration and theme
    st.set_page_config(
        page_title="AFusion Visualization",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS styles
    st.markdown("""
        <style>
        .css-18e3th9 {
            padding-top: 0rem;
            padding-bottom: 0rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .css-10trblm {
            font-size: 2rem;
            color: #2c3e50;
        }
        .css-1d391kg {
            background-color: #f2f4f5;
        }
        .stButton>button {
            background-color: #2c3e50;
            color: white;
            border-radius: 5px;
        }
        </style>
        """, unsafe_allow_html=True)

    # Title and subtitle
    st.markdown("<h1 style='text-align: center;'>🔬 AFusion: Visualization</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 16px;'>Visualize your AlphaFold 3 predictions</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 14px;'>If this project helps you, please ⭐️ <a href='https://github.com/Hanziwww/AlphaFold3-GUI' target='_blank'>my project</a>!</p>", unsafe_allow_html=True)

    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        st.sidebar.markdown("---")
        sections = {
            "Introduction": "introduction",
            "Upload Files": "upload_files",
            "Visualization": "visualization",
            "Summary Metrics": "summary_metrics",
        }
        for section_name, section_id in sections.items():
            st.markdown(f"<a href='#{section_id}' style='text-decoration: none;'>{section_name}</a>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<small>Created by Hanzi 2024.</small>", unsafe_allow_html=True)

    st.markdown('<div id="introduction"></div>', unsafe_allow_html=True)
    st.markdown("### Welcome to AFusion Visualization!")
    st.markdown("Use this app to visualize your AlphaFold 3 prediction results.")

    # File upload section
    st.markdown('<div id="upload_files"></div>', unsafe_allow_html=True)
    st.header("📂 Upload Prediction Files")
    with st.expander("Upload Files", expanded=True):
        model_cif_file = st.file_uploader("Upload model.cif file", type=["cif"])
        confidences_json_file = st.file_uploader("Upload confidences.json file", type=["json"])
        summary_confidences_file = st.file_uploader("Upload summary_confidences.json file", type=["json"])

    if model_cif_file and confidences_json_file and summary_confidences_file:
        try:
            # Read and process files
            structure, cif_content = read_cif_file_obj(model_cif_file)
            residue_bfactors, ligands = extract_residue_bfactors(structure)
            pae_matrix, token_chain_ids = extract_pae_from_json_obj(confidences_json_file)
            summary_data = extract_summary_confidences_obj(summary_confidences_file)
            logger.info("Successfully loaded and processed uploaded files.")

            # Get chain ID list
            chain_ids = list(set(token_chain_ids))
            chain_ids.sort()

            # Add visualization controls
            viz_params = add_visualization_controls(structure)

            # Display visualization results
            st.markdown('<div id="visualization"></div>', unsafe_allow_html=True)
            st.header("🌟 Visualizations")
            display_visualization_header()

            # Create two columns, ratio 3:2
            col1, col2 = st.columns([3, 2])

            with col1:
                st.write("### 3D Model Visualization")
                if residue_bfactors or ligands:
                    view_html = visualize_structure(
                        residue_bfactors, 
                        ligands, 
                        cif_content,
                        style=viz_params['style'],
                        color_scheme=viz_params['color_scheme'],
                        custom_color=viz_params['custom_color'],
                        selected_residues=viz_params['selected_residues'],
                        selection_color=viz_params['selection_color'],
                        background_color='#000000'
                    )
                    st.components.v1.html(view_html, height=600, scrolling=False)
                    
                    # Download section
                    st.write("### Download 3D Model")
                    
                    # Create download buttons
                    download_col1, download_col2, download_col3 = st.columns(3)
                    
                    # CIF Download
                    with download_col1:
                        st.download_button(
                            "Download as CIF",
                            data=cif_content.encode('utf-8'),
                            file_name="protein_model.cif",
                            mime="chemical/x-cif",
                            help="Download structure in CIF format"
                        )
                    
                    # PDB Download
                    with download_col2:
                        pdb_data = export_to_3d_formats(cif_content)
                        if pdb_data:
                            st.download_button(
                                "Download as PDB",
                                data=pdb_data,
                                file_name="protein_model.pdb",
                                mime="chemical/x-pdb",
                                help="Download structure in PDB format"
                            )
                    
                    # STL/CAD Download
                    with download_col3:
                        st.write("Export for 3D Printing/CAD")
                        
                        # Export options
                        style = st.selectbox(
                            "Export Style",
                            ["surface", "cartoon"],
                            help="Choose representation style for 3D export"
                        )
                        
                        resolution = st.slider(
                            "Resolution",
                            min_value=1.0,
                            max_value=5.0,
                            value=2.0,
                            step=0.5,
                            help="Higher resolution = more detail but larger file size"
                        )
                        
                        # Generate STL
                        stl_data = create_protein_mesh(structure, style=style, resolution=resolution)
                        
                        if stl_data:
                            st.download_button(
                                "Download for CAD (STL)",
                                data=stl_data,
                                file_name="protein_structure.stl",
                                mime="model/stl",
                                help="Download as STL file for CAD software"
                            )
                            
                            st.info("""
                            💡 STL file can be used in:
                            - CAD software (Fusion 360, SolidWorks)
                            - 3D printing slicers
                            - 3D modeling tools
                            
                            Note: Model is in Angstroms (Å)
                            """)
                        else:
                            st.error("""
                            Unable to generate STL file. 
                            Try:
                            - Reducing resolution
                            - Using different style
                            - Using PDB format instead
                            """)
                else:
                    st.error("Failed to extract atom data.")
                    logger.error("Failed to extract atom data.")

            with col2:
                # Visualize PAE matrix
                visualize_pae(pae_matrix, token_chain_ids)

            # Display summary data
            st.markdown('<div id="summary_metrics"></div>', unsafe_allow_html=True)
            display_summary_data(summary_data, chain_ids)

        except Exception as e:
            st.error(f"An error occurred while processing the files: {e}")
            logger.error(f"Error processing files: {e}")
    else:
        st.info("Please upload all required files to proceed.")

    st.markdown("---")
    # Provide log file download option
    st.markdown("### Download Log File 📥")
    with open('afusion_visualization.log', 'r') as log_file:
        log_content = log_file.read()
    st.download_button(
        label="Download Log File",
        data=log_content,
        file_name='afusion_visualization.log',
        mime='text/plain'
    )

    # Display log content
    with st.expander("Show Log Content 📄", expanded=False):
        st.text_area("Log Content", value=log_content, height=200)

    st.markdown("---")
    # Add footer
    st.markdown("<p style='text-align: center; font-size: 12px; color: #95a5a6;'>© 2024 Hanzi. All rights reserved.</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
