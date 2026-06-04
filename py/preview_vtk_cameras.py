"""
Script for previewing camera views and patches from VTK files before caching.
This uses the same data loading as main.py for consistency.
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
import GlobVar as GV
from Agent_class import Agent
from classes import FlyByDataset, pad_verts_faces
from ALIDDM_utils import GenPhongRenderer, GenDataSet
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

# Configure multiprocessing for CUDA/PyTorch3D
if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass


def preview_with_dataloader(dataloader, agent, lst_label,lm_type, output_dir=None, limit_samples=5):
    """
    Preview VTK files using the dataloader - exactly like main.py does it.
    
    Args:
        dataloader: PyTorch DataLoader
        agent: Agent instance for rendering
        lst_label: List of label IDs (teeth) to preview
        output_dir: Directory to save preview images (optional)
        limit_samples: Limit number of samples to preview
    """
    n_cameras = len(agent.camera_points)
    agent.renderer.to(GV.DEVICE)
    agent.renderer2.to(GV.DEVICE)
    
    print(f"\n{'='*70}")
    print(f"PREVIEW MODE - Using DataLoader (exactly like main.py)")
    print(f"{'='*70}")
    
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader)):
            if sample_count >= limit_samples:
                break
            
            # Unpack batch - exactly like main.py line 188
            S, V, F, RI, CN, LP, MR, SF = batch_data
            
            # Minimal GPU transfer - exactly like main.py line 191-194
            V_gpu = V.to(GV.DEVICE).float() if not isinstance(V, list) else [v.to(GV.DEVICE).float() for v in V]
            F_gpu = F.to(GV.DEVICE) if not isinstance(F, list) else [f.to(GV.DEVICE) for f in F]
            CN_gpu = CN.to(GV.DEVICE).float()
            
            # Process each patient in the batch - exactly like main.py line 197-227
            batch_size = len(S)
            for patient_idx in range(batch_size):
                if sample_count >= limit_samples:
                    break
                
                # Extract single patient data - exactly like main.py line 202-206
                V_single = V_gpu[patient_idx:patient_idx+1]
                F_single = F_gpu[patient_idx:patient_idx+1]
                CN_single = CN_gpu[patient_idx:patient_idx+1]
                RI_single = [RI[patient_idx]] if isinstance(RI, list) else RI[patient_idx:patient_idx+1]
                
                # Create input mesh - exactly like main.py line 208
                mesh_input = Meshes(verts=V_single, faces=F_single, textures=TexturesVertex(verts_features=CN_single))
                
                patient_name = os.path.basename(S[patient_idx]).split('.')[0]
                print(f"\n{'='*70}")
                print(f"Patient: {patient_name}")
                print(f"{'='*70}")
                
                # Get patient's region IDs for info
                ri_tensor = RI[patient_idx] if isinstance(RI, list) else RI[patient_idx]
                unique_regions = torch.unique(ri_tensor).tolist()
                print(f"✓ Region IDs in mesh: {unique_regions}")
                
                for label in lst_label:
                    label_str = str(label)
                    print(f"\n  → Rendering camera views for tooth {label_str}...")
                    
                    # Position agent - exactly like main.py line 211
                    agent.positions = agent.position_agent(RI_single, V_single, label)
                    
                    # Render inputs - exactly like main.py line 214
                    inputs_raw = agent.GetView(mesh_input,lm_type)  # [1, Cam, C, H, W]
                    images_np = inputs_raw[0].cpu().numpy()  # [Cam, C, H, W]
                    
                    print(f"    ✓ Rendered {n_cameras} camera views (shape: {images_np.shape})")
                    
                    # Visualize
                    fig = visualize_camera_views(
                        images_np,
                        f"{patient_name} - Tooth {label_str}",
                        ri_tensor,
                        V_single[0].cpu(),
                        label=int(label)
                    )
                    
                    # Save if output_dir provided
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                        output_path = os.path.join(output_dir, f"{patient_name}_tooth_{label_str}_preview.png")
                        fig.savefig(output_path, dpi=150, bbox_inches='tight')
                        print(f"    ✓ Saved preview to: {output_path}")
                    
                    plt.show()
                
                # Clean up GPU memory - exactly like main.py line 226-227
                del mesh_input, V_single, F_single, CN_single
                torch.cuda.empty_cache()
                
                sample_count += 1
    
    print(f"\n{'='*70}")
    print(f"Preview completed for {sample_count} samples")
    print(f"{'='*70}\n")


def visualize_camera_views(images, patient_name, region_id, verts, label=None, figsize=(18, 12)):
    """
    Create a visualization of all camera views with region information.
    
    Args:
        images: Numpy array [Cam, C, H, W]
        patient_name: Name of the patient
        region_id: Region IDs tensor
        verts: Vertices tensor
        label: Label/tooth ID being previewed (optional)
    
    Returns:
        matplotlib figure
    """
    n_cameras = images.shape[0]
    n_cols = min(3, n_cameras)
    n_rows = (n_cameras + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=figsize)
    fig.suptitle(f'{patient_name} - Camera Views & Region Info', fontsize=16, fontweight='bold')
    
    gs = GridSpec(n_rows, n_cols + 1, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot camera views
    for cam_idx in range(n_cameras):
        ax = fig.add_subplot(gs[cam_idx // n_cols, cam_idx % n_cols])
        
        # Extract RGB (first 3 channels)
        rgb = images[cam_idx, :3, :, :].transpose(1, 2, 0)
        rgb = np.clip(rgb, 0, 1)
        
        # Check if 4th channel (z-buffer) exists
        if images.shape[1] >= 4:
            zbuf = images[cam_idx, 3, :, :]
            # Add z-buffer as overlay
            ax.imshow(rgb, alpha=0.8)
            im = ax.imshow(zbuf, cmap='viridis', alpha=0.3)
            plt.colorbar(im, ax=ax, label='Z-buffer')
        else:
            ax.imshow(rgb)
        
        ax.set_title(f'Camera {cam_idx}', fontweight='bold')
        ax.axis('off')
    
    # Add region info on the right
    ax_info = fig.add_subplot(gs[:, n_cols])
    ax_info.axis('off')
    
    unique_regions = torch.unique(region_id)
    info_text = f"File: {os.path.basename(str(patient_name))}\n\n"
    if label is not None:
        info_text += f"🦷 Target Tooth: {label}\n\n"
    info_text += f"Vertices: {verts.shape[0]}\n"
    info_text += f"Unique Regions: {len(unique_regions)}\n"
    info_text += f"Region IDs: {unique_regions.tolist()}\n\n"
    info_text += f"Verts bounds:\n"
    info_text += f"  X: [{verts[:, 0].min():.3f}, {verts[:, 0].max():.3f}]\n"
    info_text += f"  Y: [{verts[:, 1].min():.3f}, {verts[:, 1].max():.3f}]\n"
    info_text += f"  Z: [{verts[:, 2].min():.3f}, {verts[:, 2].max():.3f}]\n\n"
    info_text += f"Region distribution:\n"
    for region in unique_regions:
        count = (region_id == region).sum().item()
        pct = 100.0 * count / len(region_id)
        marker = " ← TARGET" if label is not None and region.item() == label else ""
        info_text += f"  Region {region}: {count} verts ({pct:.1f}%){marker}\n"
    
    ax_info.text(0.1, 0.95, info_text, transform=ax_info.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    return fig


def setup_agent_and_renderer(jaw_type='Lower', landmark_type='O', image_size=224, blur_radius=0, faces_per_pixel=1, sphere_radius=0.2):
    """
    Initialize the agent with proper renderers - exactly like main.py
    
    Args:
        jaw_type: Type of jaw ('Upper' or 'Lower')
        landmark_type: Type of landmarks ('O' for Occlusal or 'C' for Cervical)
        image_size: Image size for rendering (default: 224)
        blur_radius: Blur radius for rendering (default: 0)
        faces_per_pixel: Faces per pixel for rendering (default: 1)
        sphere_radius: Sphere radius for camera positioning (default: 0.2)
    
    Returns:
        Agent instance
    """
    print(f"\nSetting up renderers for {jaw_type} jaw ({landmark_type})...")
    
    # Setup renderers using GenPhongRenderer - exactly like main.py line 303
    phong_renderer, mask_renderer = GenPhongRenderer(image_size, blur_radius, faces_per_pixel, GV.DEVICE)
    print(f"  ✓ Renderers initialized | Image size: {image_size}x{image_size}")
    
    # Get camera positions
    jaw_str = 'L' if jaw_type == "Lower" else 'U'
    cam_pos = GV.dic_cam[landmark_type][jaw_str]
    
    # Convert to numpy array if needed
    if isinstance(cam_pos, list):
        cam_positions = np.array(cam_pos, dtype=np.float32)
    else:
        cam_positions = np.asarray(cam_pos, dtype=np.float32)
    
    print(f"  ✓ Cameras: {len(cam_positions)} | Sphere radius: {sphere_radius}")
    
    # Create agent - exactly like main.py line 335-340
    agent = Agent(
        renderer=phong_renderer,
        renderer2=mask_renderer,
        radius=sphere_radius,
        verbose=True,
        camera_positions=cam_positions
    )
    
    return agent


def main():
    parser = argparse.ArgumentParser(
        description='Preview VTK files with camera views before caching - Using DataLoader like main.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    input_param = parser.add_argument_group('input files')
    input_param.add_argument('--dir_patients', type=str, help='Meshes directory', 
                            default='/home/luciacev/Desktop/training ios files/all data')
    input_param.add_argument('--csv_file', type=str, help='CSV file with patient data',
                            default='/home/luciacev/Desktop/training ios files/all data/csv files/data_lower_fold_0_O.csv')
    
    input_param.add_argument('--jaw', type=str, default='Lower', choices=['Upper', 'Lower'])
    input_param.add_argument('-lm', '--lm_typ', type=str, default='C', choices=['O', 'C'],
                            help="Landmark type: 'O' for Occlusal or 'C' for Cervical")
    input_param.add_argument('--image_size', type=int, default=224)
    input_param.add_argument('--blur_radius', type=int, default=0)
    input_param.add_argument('--faces_per_pixel', type=int, default=1)
    input_param.add_argument('-sr', '--sphere_radius', type=float, default=0.25,
                            help='Sphere radius for camera positioning (default: 0.2)')
    input_param.add_argument('-bs', '--batch_size', type=int, default=4)
    
    input_param.add_argument('--labels', type=str, default='',
                            help='Comma-separated list of labels (teeth) to preview')
    input_param.add_argument('--output-dir', type=str, default=None,
                            help='Directory to save preview images')
    input_param.add_argument('--limit', type=int, default=1,
                            help='Limit number of samples to preview')
    input_param.add_argument('--num_device', type=str, default='0')
    
    args = parser.parse_args()
    
    # Setup device
    if args.num_device == '-1' or not torch.cuda.is_available():
        GV.DEVICE = torch.device("cpu")
    else:
        GV.DEVICE = torch.device(f"cuda:{args.num_device}")
    
    print(f"\n{'='*70}")
    print(f"VTK PREVIEW TOOL - DataLoader Mode")
    print(f"{'='*70}")
    print(f"Device: {GV.DEVICE}")
    print(f"Jaw type: {args.jaw}")
    print(f"Landmark type: {args.lm_typ}")
    
    # Determine labels to preview
    if args.labels:
        lst_label = [s.strip() for s in args.labels.split(',')]
    else:
        # Use default labels based on jaw type
        if args.jaw == 'Upper':
            lst_label = ["2","3","4","5","6","7","8","9","10","11","12","13","14","15"]
        else:  # Lower
            lst_label = ["18","19","20","21","22","23","24","25","26","27","28","29","30","31"]
    
    print(f"Labels to preview: {lst_label[:5]}{'...' if len(lst_label) > 5 else ''}")
    
    # Load data
    print(f"\nLoading data from: {args.csv_file}")
    df = pd.read_csv(args.csv_file)
    print(f"  ✓ Loaded {len(df)} samples from CSV")
    
    # Create dataset
    dataset, _ = GenDataSet(df, args.dir_patients, FlyByDataset, 'cpu', landmark_type=args.lm_typ)
    print(f"  ✓ Dataset created with {len(dataset)} samples")
    
    # Create dataloader - exactly like main.py
    optimal_workers = min(8, os.cpu_count() or 4)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=pad_verts_faces,
        num_workers=optimal_workers,
        pin_memory=True,
        persistent_workers=(optimal_workers > 0),
        prefetch_factor=2
    )
    print(f"  ✓ DataLoader created | Batch size: {args.batch_size} | Workers: {optimal_workers}")
    
    # Setup agent
    agent = setup_agent_and_renderer(
        jaw_type=args.jaw,
        landmark_type=args.lm_typ,
        image_size=args.image_size,
        blur_radius=args.blur_radius,
        faces_per_pixel=args.faces_per_pixel,
        sphere_radius=args.sphere_radius
    )
    
    # Preview with dataloader
    preview_with_dataloader(dataloader, agent, lst_label, args.jaw,args.output_dir, args.limit)
    
    print("✓ Preview completed!")


if __name__ == '__main__':
    main()
