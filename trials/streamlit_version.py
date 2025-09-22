#!/usr/bin/env python3
"""
Interactive Streamlit DICOM Viewer with Real-time 3D Visualization
"""
import streamlit as st
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from collections import defaultdict
from io import BytesIO
from skimage import measure
import time

# --- VTK Setup ---
try:
    import vtk
    from vtk.util import numpy_support
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False

# --- Core DICOM Logic ---

class DICOMDataManager:
    """Manages DICOM data loading, scanning, and caching."""

    @staticmethod
    @st.cache_data
    def scan_for_series(directory_path):
        """Scans a directory for DICOM files and groups them by series."""
        p = Path(directory_path)
        if not p.is_dir():
            st.error(f"Error: The provided path '{directory_path}' is not a valid directory.")
            return {}

        series_map = defaultdict(list)
        
        # Look for various DICOM file extensions
        dicom_extensions = ['*.dcm', '*.DCM', '*.dicom', '*.DICOM', '*']
        dicom_files = []
        
        for ext in dicom_extensions:
            dicom_files.extend(list(p.glob(f'**/{ext}')))
        
        # Remove duplicates and filter for likely DICOM files
        dicom_files = list(set(dicom_files))
        
        if not dicom_files:
            st.warning("No files found in the specified directory.")
            return {}
            
        with st.spinner(f"Scanning {len(dicom_files)} files..."):
            valid_files = 0
            for f in dicom_files:
                try:
                    # Try to read as DICOM
                    ds = pydicom.dcmread(f, stop_before_pixels=True)
                    series_uid = ds.SeriesInstanceUID
                    series_map[series_uid].append(f)
                    valid_files += 1
                except Exception as e:
                    # Skip files that are not valid DICOM
                    continue
            
            st.info(f"Found {valid_files} valid DICOM files out of {len(dicom_files)} files scanned.")
        
        # Create a more descriptive map for the UI
        series_info = {}
        for series_uid, files in series_map.items():
            if not files: continue
            
            # Read metadata from the first file of the series
            try:
                ds_sample = pydicom.dcmread(files[0])
                description = ds_sample.get("SeriesDescription", "N/A")
                num_slices = len(files)
                patient_id = ds_sample.get("PatientID", "N/A")
                
                # Create a unique, descriptive key for the selectbox
                series_key = f"Patient {patient_id} - {description} ({num_slices} slices)"
                series_info[series_key] = {
                    "files": files,
                    "uid": series_uid
                }
            except Exception as e:
                st.warning(f"Could not read metadata from {files[0]}: {e}")
                continue

        return series_info

    @staticmethod
    @st.cache_data
    def load_dicom_volume(file_paths):
        """Loads and sorts a list of DICOM files into a 3D NumPy volume."""
        if not file_paths:
            st.error("No files provided to load.")
            return None, None
            
        slices = []
        failed_files = 0
        
        for f in file_paths:
            try:
                slice_data = pydicom.dcmread(f)
                # Check if the file has pixel data
                if hasattr(slice_data, 'pixel_array'):
                    slices.append(slice_data)
                else:
                    failed_files += 1
            except Exception as e:
                st.warning(f"Could not read file: {f} - {e}")
                failed_files += 1
                continue

        if not slices:
            st.error("No valid DICOM slices with pixel data found!")
            return None, None
            
        if failed_files > 0:
            st.warning(f"Failed to load {failed_files} out of {len(file_paths)} files.")

        # Sort slices based on Image Position (Patient) for robustness
        try:
            slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
            st.info("Sorted slices by ImagePositionPatient[2]")
        except (AttributeError, IndexError, TypeError):
            try:
                slices.sort(key=lambda s: int(s.InstanceNumber))
                st.info("Sorted slices by InstanceNumber")
            except (AttributeError, TypeError):
                st.warning("Could not sort slices. Using original order.")
        
        # Stack the slices into a 3D volume
        try:
            # Get pixel arrays
            pixel_arrays = []
            for s in slices:
                pixel_array = s.pixel_array
                # Ensure 2D array
                if len(pixel_array.shape) == 3:
                    pixel_array = pixel_array[:,:,0]  # Take first channel if RGB
                pixel_arrays.append(pixel_array)
            
            volume = np.stack(pixel_arrays, axis=0)
            
            # Apply rescale slope and intercept if available
            if hasattr(slices[0], 'RescaleSlope') and hasattr(slices[0], 'RescaleIntercept'):
                slope = float(slices[0].RescaleSlope)
                intercept = float(slices[0].RescaleIntercept)
                volume = volume * slope + intercept
                st.info(f"Applied rescale: slope={slope}, intercept={intercept}")
            
            st.success(f"Successfully created 3D volume with shape: {volume.shape}")
            
        except Exception as e:
            st.error(f"Error creating 3D volume: {e}")
            return None, None

        return slices[0], volume

# --- Visualization Classes ---
class ImageVisualization:
    @staticmethod
    def display_slice(volume, slice_idx):
        if volume.shape[0] <= slice_idx:
            st.error(f"Slice index {slice_idx} is out of range for volume with {volume.shape[0]} slices.")
            return None
            
        slice_data = volume[slice_idx, :, :]
        fig = px.imshow(slice_data, color_continuous_scale='gray', title=f"Axial Slice {slice_idx}")
        fig.update_layout(width=500, height=500, title_x=0.5)
        return fig

class Interactive3DViewer:
    """Handles interactive 3D visualization using Plotly."""
    
    @staticmethod
    @st.cache_data
    def generate_mesh(volume, iso_value, step_size=2):
        """Generate 3D mesh using marching cubes with caching."""
        try:
            # Downsample for performance if volume is large
            if volume.shape[0] > 100 or volume.shape[1] > 512 or volume.shape[2] > 512:
                volume_ds = volume[::step_size, ::step_size, ::step_size]
                spacing_factor = step_size
            else:
                volume_ds = volume
                spacing_factor = 1
            
            # Use scikit-image marching cubes (faster than VTK for smaller volumes)
            verts, faces, _, _ = measure.marching_cubes(volume_ds, level=iso_value)
            
            # Scale vertices back up if we downsampled
            verts = verts * spacing_factor
            
            return verts, faces
        except Exception as e:
            st.error(f"Error generating mesh: {e}")
            return None, None
    
    @staticmethod
    def create_interactive_plot(volume, iso_value, step_size=2):
        """Create an interactive 3D plot using Plotly."""
        verts, faces = Interactive3DViewer.generate_mesh(volume, iso_value, step_size)
        
        if verts is None or faces is None:
            return None
        
        # Create the 3D mesh plot
        fig = go.Figure(data=[
            go.Mesh3d(
                x=verts[:, 2],  # Swap axes for better orientation
                y=verts[:, 1],
                z=verts[:, 0],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color='lightblue',
                opacity=0.8,
                lighting=dict(
                    ambient=0.18,
                    diffuse=1,
                    fresnel=0.1,
                    specular=1,
                    roughness=0.1,
                ),
                lightposition=dict(x=100, y=200, z=0)
            )
        ])
        
        # Update layout for better 3D experience
        fig.update_layout(
            title="Interactive 3D DICOM Model",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y", 
                zaxis_title="Z",
                camera=dict(
                    eye=dict(x=1.2, y=1.2, z=0.6)
                ),
                aspectmode='data'
            ),
            width=800,
            height=600,
            margin=dict(r=0, b=0, l=0, t=40)
        )
        
        return fig

class VolumeRenderer:
    """Handles volume rendering for more sophisticated visualization."""
    
    @staticmethod
    def create_volume_plot(volume, opacity_scale=0.1, surface_count=10):
        """Create a volume rendering using multiple isosurfaces."""
        # Calculate percentile-based isovalues for better visualization
        percentiles = np.linspace(50, 95, surface_count)
        iso_values = [np.percentile(volume, p) for p in percentiles]
        
        # Create multiple mesh objects with varying opacity
        mesh_objects = []
        colors = px.colors.sample_colorscale("viridis", surface_count)
        
        for i, iso_val in enumerate(iso_values):
            try:
                verts, faces = Interactive3DViewer.generate_mesh(volume, iso_val, step_size=3)
                if verts is not None and len(verts) > 0:
                    mesh_objects.append(
                        go.Mesh3d(
                            x=verts[:, 2],
                            y=verts[:, 1], 
                            z=verts[:, 0],
                            i=faces[:, 0],
                            j=faces[:, 1],
                            k=faces[:, 2],
                            color=colors[i],
                            opacity=opacity_scale * (i + 1) / surface_count,
                            name=f"Level {i+1}",
                            showlegend=False
                        )
                    )
            except:
                continue
        
        if not mesh_objects:
            return None
            
        fig = go.Figure(data=mesh_objects)
        fig.update_layout(
            title="Volume Rendering (Multiple Isosurfaces)",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                camera=dict(eye=dict(x=1.2, y=1.2, z=0.6)),
                aspectmode='data'
            ),
            width=800,
            height=600,
            margin=dict(r=0, b=0, l=0, t=40)
        )
        
        return fig


# --- Streamlit UI ---

def main():
    st.set_page_config(page_title="Interactive DICOM 3D Viewer", layout="wide")
    st.title("🏥 Interactive DICOM 3D Viewer")

    # --- Sidebar for Data Loading ---
    st.sidebar.header("📂 Data Loading")
    
    directory_path = st.sidebar.text_input(
        "Enter path to DICOM parent directory:", 
        "manifest-1617826161202/Pseudo-PHI-DICOM-Data"
    )

    if st.sidebar.button("Scan Directory"):
        if not directory_path.strip():
            st.sidebar.error("Please enter a valid directory path.")
        else:
            series_info = DICOMDataManager.scan_for_series(directory_path.strip())
            st.session_state.series_info = series_info
            # Clear cached 3D data when scanning new directory
            if 'volume' in st.session_state:
                del st.session_state.volume
            if 'current_mesh' in st.session_state:
                del st.session_state.current_mesh
                
            if not series_info:
                st.sidebar.error("No DICOM series found in the specified directory.")

    if 'series_info' in st.session_state and st.session_state.series_info:
        series_options = list(st.session_state.series_info.keys())
        selected_series_key = st.sidebar.selectbox(
            "Select a DICOM Series to load:",
            options=series_options,
            index=0
        )
        
        if st.sidebar.button("Load Selected Series"):
            selected_series_files = st.session_state.series_info[selected_series_key]["files"]
            with st.spinner("Loading DICOM series..."):
                metadata, volume = DICOMDataManager.load_dicom_volume(selected_series_files)
                if metadata and volume is not None and volume.size > 0:
                    st.session_state.metadata = metadata
                    st.session_state.volume = volume
                    # Clear cached mesh when loading new series
                    if 'current_mesh' in st.session_state:
                        del st.session_state.current_mesh
                    st.sidebar.success("✅ Series loaded successfully!")
                else:
                    st.sidebar.error("❌ Failed to load series or series is empty!")
    else:
        st.info("👆 Scan a directory to find available DICOM series.")

    # --- Main Panel for Visualization ---
    if 'volume' in st.session_state and st.session_state.volume.size > 0:
        volume = st.session_state.volume
        metadata = st.session_state.metadata
        
        # Additional validation
        if len(volume.shape) != 3:
            st.error(f"Expected 3D volume, got shape: {volume.shape}")
            return
            
        if volume.shape[0] == 0:
            st.error("Volume has no slices!")
            return
        
        st.header(f"Viewing: {metadata.get('SeriesDescription', 'N/A')}")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Metadata", "🖼️ 2D Slice Viewer", "🎮 Interactive 3D", "🌊 Volume Rendering"])
        
        with tab1:
            col1, col2 = st.columns(2)
            col1.metric("Patient ID", metadata.get('PatientID', 'N/A'))
            col1.metric("Modality", metadata.get('Modality', 'N/A'))
            col1.metric("Study Date", metadata.get('StudyDate', 'N/A'))
            col2.metric("Volume Shape", f"{volume.shape[0]}×{volume.shape[1]}×{volume.shape[2]}")
            col2.metric("Data Type", str(volume.dtype))
            col2.metric("Value Range", f"{volume.min():.2f} to {volume.max():.2f}")
        
        with tab2:
            # Safe slider creation
            max_slice = volume.shape[0] - 1
            if max_slice >= 0:
                slice_idx = st.slider(
                    "Slice Index:", 
                    min_value=0, 
                    max_value=max_slice, 
                    value=max_slice // 2,
                    help=f"Navigate through {volume.shape[0]} slices"
                )
                
                fig = ImageVisualization.display_slice(volume, slice_idx)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("No slices available to display.")

        with tab3:
            st.subheader("🎮 Interactive 3D Visualization")
            st.info("Drag to rotate • Scroll to zoom • Shift+drag to pan")
            
            # Create columns for better layout
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.markdown("**Controls**")
                
                # Smart ISO value suggestions
                min_val, max_val = float(volume.min()), float(volume.max())
                
                # Pre-calculate good ISO values
                percentiles = [25, 50, 75, 85, 95]
                iso_suggestions = {f"{p}th percentile": float(np.percentile(volume, p)) for p in percentiles}
                
                # ISO value selection with presets
                iso_method = st.radio(
                    "ISO Value Selection:",
                    ["Quick Presets", "Custom Value"],
                    help="Quick presets are calculated from your data"
                )
                
                if iso_method == "Quick Presets":
                    preset_options = list(iso_suggestions.keys())
                    selected_preset = st.selectbox("Choose preset:", preset_options, index=3)
                    iso_value = iso_suggestions[selected_preset]
                    st.info(f"Value: {iso_value:.2f}")
                else:
                    iso_value = st.slider(
                        "Custom ISO Value:",
                        min_value=min_val,
                        max_value=max_val,
                        value=float(np.percentile(volume, 85)),
                        help="Higher values = denser structures"
                    )
                
                # Performance settings
                st.markdown("**Performance**")
                step_size = st.selectbox(
                    "Quality vs Speed:",
                    options=[1, 2, 3, 4],
                    index=1,
                    format_func=lambda x: {1: "Best Quality", 2: "Balanced", 3: "Fast", 4: "Fastest"}[x],
                    help="Lower values = better quality but slower"
                )
                
                # Auto-update option
                auto_update = st.checkbox("Auto-update on change", value=True)
                manual_update = st.button("🔄 Update 3D View", disabled=auto_update)
            
            with col2:
                # Determine if we should update the plot
                update_needed = auto_update or manual_update or 'last_iso' not in st.session_state
                
                if update_needed:
                    # Store current settings
                    st.session_state.last_iso = iso_value
                    st.session_state.last_step = step_size
                    
                    with st.spinner("Generating 3D mesh..."):
                        fig = Interactive3DViewer.create_interactive_plot(volume, iso_value, step_size)
                        if fig:
                            st.session_state.current_3d_fig = fig
                        else:
                            st.error("Failed to generate 3D visualization")
                
                # Display the current figure
                if 'current_3d_fig' in st.session_state:
                    st.plotly_chart(st.session_state.current_3d_fig, use_container_width=True)
                    
                    # Add download button
                    if st.button("📥 Save 3D View as HTML"):
                        html_buffer = BytesIO()
                        st.session_state.current_3d_fig.write_html(html_buffer)
                        st.download_button(
                            label="Download Interactive 3D",
                            data=html_buffer.getvalue(),
                            file_name="dicom_3d_view.html",
                            mime="text/html"
                        )

        with tab4:
            st.subheader("🌊 Volume Rendering")
            st.info("Multiple semi-transparent layers reveal internal structures")
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.markdown("**Volume Settings**")
                opacity = st.slider("Opacity Scale:", 0.05, 0.3, 0.1, 0.05)
                layers = st.slider("Number of Layers:", 3, 15, 8)
                
                if st.button("🎨 Generate Volume Rendering"):
                    with st.spinner("Creating volume rendering..."):
                        vol_fig = VolumeRenderer.create_volume_plot(volume, opacity, layers)
                        if vol_fig:
                            st.session_state.volume_fig = vol_fig
                        else:
                            st.error("Failed to create volume rendering")
            
            with col2:
                if 'volume_fig' in st.session_state:
                    st.plotly_chart(st.session_state.volume_fig, use_container_width=True)
                else:
                    st.info("Click 'Generate Volume Rendering' to create a multi-layer visualization")

    else:
        if 'volume' in st.session_state:
            st.error("Loaded volume is empty or invalid. Please try loading a different series.")
        else:
            st.info("Load a DICOM series to begin visualization.")

if __name__ == "__main__":
    main()