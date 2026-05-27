#!/usr/bin/env python3
"""
Visualize cached inputs and targets from disk.
Shows 5 camera views side by side for a selected patient.
"""
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np

CACHE_BASE_DIR = '/media/luciacev/Data/ALI_IOS cache'

def get_cached_files(cache_dir):
    """List all cached files in a directory."""
    if not os.path.exists(cache_dir):
        return []
    return sorted([f for f in os.listdir(cache_dir) if f.endswith('.pth')])

def load_tensor(filepath):
    """Load a tensor from disk."""
    try:
        return torch.load(filepath, weights_only=True)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def visualize_patient(patient_name, label, jawtype='L', lm_typ='O', fold_idx=None, cache_type=None):
    """
    Visualize inputs and targets for a patient and tooth label.
    Automatically finds the correct fold and cache_type if not specified.
    
    Args:
        patient_name: e.g., "A10_T1_L_SegOrReg"
        label: tooth label, e.g., "18"
        jawtype: 'L' or 'U'
        lm_typ: 'O' or 'C'
        fold_idx: fold index (0-4), auto-detect if None
        cache_type: 'train' or 'val', auto-detect if None
    """
    # Determine number of cameras based on landmark type
    region_str = 'cervical' if lm_typ=='C' else 'occlusal'
    n_cameras = 5 if lm_typ.upper() == 'O' else 12
    
    # Load input from global cache
    input_dir = os.path.join(CACHE_BASE_DIR,region_str, f'global_inputs_{jawtype}')
    input_file = f"input_{patient_name}_{label}.pth"
    input_path = os.path.join(input_dir, input_file)
    
    # Check if input exists
    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        return
    
    # Auto-detect fold and cache_type if not specified
    target_path = None
    if fold_idx is not None and cache_type is not None:
        target_dir = os.path.join(CACHE_BASE_DIR,region_str, f'fold_{fold_idx}_targets_{cache_type}_{jawtype}_{lm_typ}')
        target_file = f"target_{patient_name}_{label}.pth"
        target_path = os.path.join(target_dir, target_file)
    else:
        # Search for target in all folds and cache types
        for fi in range(5):
            for ct in ['train', 'val']:
                target_dir = os.path.join(CACHE_BASE_DIR,region_str, f'fold_{fi}_targets_{ct}_{jawtype}_{lm_typ}')
                target_file = f"target_{patient_name}_{label}.pth"
                test_path = os.path.join(target_dir, target_file)
                if os.path.exists(test_path):
                    target_path = test_path
                    fold_idx = fi
                    cache_type = ct
                    break
            if target_path is not None:
                break
    
    if target_path is None:
        print(f"❌ Target file not found in any fold!")
        print(f"   Searched for: target_{patient_name}_{label}.pth")
        return
    
    # Load tensors
    input_tensor = load_tensor(input_path)
    target_tensor = load_tensor(target_path)
    
    if input_tensor is None or target_tensor is None:
        return
    
    print(f"✅ Loaded {patient_name} | Tooth {label} | Mode {lm_typ} | Fold {fold_idx} ({cache_type})")
    print(f"   Input shape: {input_tensor.shape}")
    print(f"   Target shape: {target_tensor.shape}")
    
    # input_tensor: [n_cameras, 4, 224, 224] = cameras × (RGB + Z)
    # target_tensor: [n_cameras, 4, 224, 224] = cameras × (RGB + Z)
    
    # Extract RGB channels (first 3 channels, drop Z-buffer)
    input_rgb = input_tensor[:, :3, :, :]  # [n_cameras, 3, 224, 224]
    target_rgb = target_tensor[:, :3, :, :]  # [n_cameras, 3, 224, 224]
    
    # Create figure with 2 rows × n_cameras columns
    fig = plt.figure(figsize=(4*n_cameras, 8))
    gs = GridSpec(2, n_cameras, figure=fig, hspace=0.3, wspace=0.1)
    
    # Plot inputs (top row)
    for cam_idx in range(n_cameras):
        ax = fig.add_subplot(gs[0, cam_idx])
        img = input_rgb[cam_idx].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_title(f"Input - Cam {cam_idx}", fontsize=8, fontweight='bold')
        ax.axis('off')
    
    # Plot targets (bottom row)
    for cam_idx in range(n_cameras):
        ax = fig.add_subplot(gs[1, cam_idx])
        img = target_rgb[cam_idx].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        
        # Add color legend based on mode
        if lm_typ.upper() == 'O':
            legend_text = "🔴 O, 🟢 MB, 🔵 DB"
        else:  # 'C'
            legend_text = "🟣 CL, 🟠 CB"
        
        ax.set_title(f"Target ({lm_typ}) - Cam {cam_idx}\n{legend_text}", fontsize=7, fontweight='bold')
        ax.axis('off')
    
    fig.suptitle(f"Patient: {patient_name} | Tooth: {label} | Jaw: {jawtype} | Mode: {lm_typ} ({n_cameras} cams) | Fold: {fold_idx} ({cache_type})", 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def list_available_patients(jawtype='L',region = 'O'):
    """List all available patients in the global input cache."""
    region_str = 'cervical' if region=='C' else 'occlusal'
    input_dir = os.path.join(CACHE_BASE_DIR,region_str, f'global_inputs_{jawtype}')
    files = get_cached_files(input_dir)
    
    # Extract unique patient names
    patients = set()
    for f in files:
        # Extract patient name from "input_PATIENT_LABEL.pth"
        parts = f.replace('input_', '').replace('.pth', '').rsplit('_', 1)
        if len(parts) == 2:
            patient_name = parts[0]
            patients.add(patient_name)
    
    return sorted(list(patients))

def main():
    print("\n" + "="*80)
    print("          CACHE VISUALIZATION TOOL")
    print("="*80)
    
    # Configuration
    jawtype = 'L'  # 'L' for Lower, 'U' for Upper
    lm_typ = 'C'   # 'O' for Occlusal, 'C' for Cervical
    fold_idx = 0
    cache_type = 'train'
    
    # List available patients
    patients = list_available_patients(jawtype,lm_typ)
    print(f"\n✅ Found {len(patients)} unique patients in {jawtype} jaw cache")
    print(f"   First 5 patients: {patients[:5]}")
    
    if len(patients) == 0:
        print("❌ No patients found. Make sure pre-rendering is complete.")
        return
    
    # Select first patient and all teeth
    patient_name = patients[0]
    print(f"\n📋 Visualizing patient: {patient_name}")
    
    # List available teeth for this patient
    input_dir = os.path.join(CACHE_BASE_DIR,"cervical", f'global_inputs_{jawtype}')
    files = get_cached_files(input_dir)
    
    teeth = set()
    for f in files:
        if patient_name in f:
            # Extract tooth label from "input_PATIENT_LABEL.pth"
            label = f.replace('input_', '').replace('.pth', '').replace(patient_name + '_', '')
            teeth.add(label)
    
    teeth = sorted(list(teeth))
    print(f"   Available teeth: {teeth}")
    
    # Visualize first few teeth
    for tooth_idx, label in enumerate(teeth[:15]):
        print(f"\n▶️  Visualizing tooth {label}...")
        visualize_patient(patient_name, label, jawtype=jawtype, lm_typ=lm_typ)  # Auto-detect fold
        
        if tooth_idx >= 10:  # Show max 3 teeth
            break

if __name__ == '__main__':
    main()
