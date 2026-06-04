import argparse
import os
import torch
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import torch.multiprocessing as mp
from scipy import linalg
from torch.utils.data import DataLoader

from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from torch.utils.tensorboard import SummaryWriter
from monai.losses import DiceCELoss
from torch.optim import Adam
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

from ALIDDM_utils import *
from classes import *
import GlobVar as GV
from Agent_class import *
from training import Model, Training, Validation
from ALIDDM_utils import Gen_Full_Mask_Mesh

# Global cache directory for all cache files
CACHE_BASE_DIR = '/media/luciacev/Data/ALI_IOS cache_3channelsout_cam'

# Configure multiprocessing for CUDA/PyTorch3D
if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    Stops training if validation loss doesn't improve for a specified number of epochs.
    """
    def __init__(self, patience=20, min_delta=0.001, verbose=True):
        """
        Args:
            patience: Number of epochs with no improvement after which training will be stopped
            min_delta: Minimum change to qualify as an improvement (in loss, lower is better)
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = -1

    def __call__(self, current_loss, epoch):
        """
        Check if training should stop based on validation loss.
        
        Args:
            current_loss: Current validation loss value (lower is better)
            epoch: Current epoch number
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = current_loss
            self.best_epoch = epoch
        elif current_loss < self.best_loss - self.min_delta:
            # Loss improved (decreased)
            self.best_loss = current_loss
            self.counter = 0
            self.best_epoch = epoch
            if self.verbose:
                print(f"✓ Validation loss improved to {current_loss:.6f}")
        else:
            # No improvement in loss
            self.counter += 1
            if self.verbose:
                print(f"No improvement for {self.counter}/{self.patience} validations | Best loss: {self.best_loss:.6f}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"EARLY STOPPING: No improvement for {self.patience} validations")
                return True
        
        return False

    def reset(self):
        """Reset early stopping state for next fold"""
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = -1

def check_cache_complete(input_dir, mask_dir, dataloader, lst_label):
    """
    Check if cache is complete (all expected files exist).
    Returns True if all inputs and targets are cached.
    Also validates cache files and removes corrupted ones.
    
    STRATEGY:
    - INPUTS: Only check if they exist (fold-level, reused across modes)
    - TARGETS: Check both input and mask (per-mode specific)
    """
    # Check INPUTS exist (fold-level cache)
    if not os.path.exists(input_dir):
        return False
    
    # Check TARGETS exist (per-mode cache)
    if not os.path.exists(mask_dir):
        return False
    
    # Count expected files (patients × labels)
    expected_count = len(dataloader.dataset) * len(lst_label)
    
    # Validate and clean input files
    input_files = []
    for f in os.listdir(input_dir):
        fpath = os.path.join(input_dir, f)
        try:
            torch.load(fpath, weights_only=True)
            input_files.append(f)
        except:
            print(f"Removing corrupted cache file: {fpath}")
            try:
                os.remove(fpath)
            except:
                pass
    
    # Validate and clean mask files
    mask_files = []
    for f in os.listdir(mask_dir):
        fpath = os.path.join(mask_dir, f)
        try:
            torch.load(fpath, weights_only=True)
            mask_files.append(f)
        except:
            print(f"Removing corrupted cache file: {fpath}")
            try:
                os.remove(fpath)
            except:
                pass
    
    # Check if we have all expected files
    # Both inputs and masks should have the same count and match the expected count
    if len(input_files) == expected_count and len(mask_files) == expected_count:
        return True
    
    return False

def pre_render_all_inputs_only(dataloader, agent, lst_label, jawtype,lm_type):
    """
    Pre-render and cache INPUTS ONLY (no landmarks) to a global directory.
    These inputs are reused across all folds and landmark modes.
    This function should be called ONCE before the CV loop.
    
    Args:
        dataloader: PyTorch DataLoader containing ALL patients
        agent: Agent object for rendering
        lst_label: List of label IDs to render
        jawtype: 'L' or 'U' for lower or upper jaw
    """
    # Global input cache (not fold-specific, not lm_typ-specific)
    lm_type_dir = "cervical" if lm_type == "C" else "occlusal"
    input_dir = os.path.join(CACHE_BASE_DIR,lm_type_dir, f'global_inputs_{jawtype}')
    
    # Check if complete global cache exists
    expected_count = len(dataloader.dataset) * len(lst_label)
    
    if os.path.exists(input_dir):
        input_files = [f for f in os.listdir(input_dir) if f.endswith('.pth')]
        if len(input_files) == expected_count:
            print(f"GLOBAL CACHE FOUND | All inputs already cached | Patients: {len(dataloader.dataset)} | Labels: {len(lst_label)}")
            return
    
    # Create directory
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    
    # Move renderers to device
    agent.renderer.to(GV.DEVICE)
    agent.renderer2.to(GV.DEVICE)
    
    print(f"PRE-RENDERING GLOBAL INPUTS (no landmarks) | {len(dataloader.dataset)} patients × {len(lst_label)} labels → {input_dir}")
    
    with torch.no_grad():
        for batch_idx, (S, V, F, RI, CN, LP, MR, SF) in enumerate(tqdm(dataloader)):
            # Minimal GPU transfer
            V_gpu = V.to(GV.DEVICE).float() if not isinstance(V, list) else [v.to(GV.DEVICE).float() for v in V]
            F_gpu = F.to(GV.DEVICE) if not isinstance(F, list) else [f.to(GV.DEVICE) for f in F]
            CN_gpu = CN.to(GV.DEVICE).float()
            
            # Process each patient in the batch individually to avoid CUDA memory issues
            batch_size = len(S)
            for patient_idx in range(batch_size):
                # Extract single patient data
                V_single = V_gpu[patient_idx:patient_idx+1]
                F_single = F_gpu[patient_idx:patient_idx+1]
                CN_single = CN_gpu[patient_idx:patient_idx+1]
                RI_single = [RI[patient_idx]] if isinstance(RI, list) else RI[patient_idx:patient_idx+1]
                
                # Create input mesh (Phong rendering - colored normals)
                mesh_input = Meshes(verts=V_single, faces=F_single, textures=TexturesVertex(verts_features=CN_single))
                
                for label in lst_label:
                    # Position agent for this label (single patient)
                    agent.positions = agent.position_agent(RI_single, V_single, label)
                    
                    # Render INPUTS (no landmarks)
                    inputs_raw = agent.GetView(mesh_input,args.jaw)  # [1, Cam, C, H, W]
                    
                    # Save per patient
                    patient_name = os.path.basename(S[patient_idx]).split('.')[0]
                    input_file = f"input_{patient_name}_{label}.pth"
                    input_path = os.path.join(input_dir, input_file)
                    torch.save(inputs_raw[0].cpu(), input_path)
                
                # Clean up GPU memory
                del mesh_input, V_single, F_single, CN_single
                torch.cuda.empty_cache()
    
    print(f"GLOBAL INPUTS CACHED | {expected_count} files saved")

def pre_render_all_inputs_and_targets(dataloader, agent, lst_label, fold_idx, jawtype, lm_typ='o', cache_type='train'):
    lm_type_dir = "cervical" if lm_typ == "C" else "occlusal"
    mask_dir = os.path.join(CACHE_BASE_DIR,lm_type_dir, f'fold_{fold_idx}_targets_{cache_type}_{jawtype}_{lm_typ}')
    
    # Vérification du cache
    expected_count = len(dataloader.dataset) * len(lst_label)
    if os.path.exists(mask_dir):
        mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.pth')]
        if len(mask_files) >= expected_count:
            print(f"COMPLETE TARGETS CACHE FOUND | Skipping pre-render")
            return
    
    if not os.path.exists(mask_dir): os.makedirs(mask_dir)
    
    agent.renderer.to(GV.DEVICE)
    agent.renderer2.to(GV.DEVICE)
    
    print(f"PRE-RENDERING TARGETS → {mask_dir}")

    with torch.no_grad(): 
            for S, V, F, RI, CN, LP, MR, SF in tqdm(dataloader):
                V_gpu = V.to(GV.DEVICE).float()
                F_gpu = F.to(GV.DEVICE)
                CN_gpu = CN.to(GV.DEVICE).float()
                
                # LP est une liste de tenseurs CPU, on la garde telle quelle pour l'instant
                current_batch_size = V_gpu.shape[0]

                for label in lst_label:
                    # 1. Calculer les positions pour TOUT le batch (B, 5, 3)
                    all_positions = agent.position_agent(RI, V_gpu, label)
                    
                    for i in range(current_batch_size):
                        patient_name = os.path.basename(S[i]).split('.')[0]
                        target_path = os.path.join(mask_dir, f"target_{patient_name}_{label}.pth")
                        if os.path.exists(target_path): continue

                        V_i = V_gpu[i:i+1].detach().clone() # .clone() est vital ici
                        F_i = F_gpu[i:i+1].detach().clone()
                        
                        # On génère le mesh
                        # print(LP)
                        mesh_target = Gen_mesh_patch(S, V_i, F_i, CN_gpu[i:i+1], LP, label, batch_idx=i, jaw=jawtype, lm_typ=lm_typ)
                        
                        # On donne à l'agent SEULEMENT les positions du patient i
                        agent.positions = all_positions[i:i+1].detach().clone()
                        
                        # On lance le rendu pour ce patient SEUL
                        targets_raw = agent.GetView(mesh_target,args.jaw, rend=True) 

                        # Sauvegarde
                        torch.save(targets_raw.squeeze(0).cpu(), target_path)
                        
                        # NETTOYAGE GPU IMMÉDIAT
                        del mesh_target, targets_raw, V_i, F_i

    print(f"TARGETS CACHE COMPLETE")

def main(args):
    print("\n" + "="*80)
    print("               ALIDDM CROSS-VALIDATION TRAINING")
    print("="*80)
    main_start_time = time.time()
    batch_siz = 16 if args.lm_typ == "O" else 4
    
    if args.num_device == '-1' or not torch.cuda.is_available():
        GV.DEVICE = torch.device("cpu")
    else:
        GV.DEVICE = torch.device(f"cuda:{args.num_device}")
    
    jaw = args.jaw  # Store jaw in local variable
    print(f"\n[INIT] Configuration | Device: {GV.DEVICE} | Jaw: {jaw} | Landmark type: {args.lm_typ.upper()}")
    
    lst_label = args.lst_label_u if jaw == "U" else args.lst_label_l
    jaw_suffix = "upper" if jaw == "U" else "lower"

    phong_renderer, mask_renderer = GenPhongRenderer(args.image_size, args.blur_radius, args.faces_per_pixel, GV.DEVICE)
    print(f"[INIT] Renderers initialized | Image size: {args.image_size}x{args.image_size}")

    # === PRE-RENDER GLOBAL INPUTS (ONCE for all folds and modes) ===
    print("\n" + "="*80)
    print("                    PRE-RENDERING GLOBAL INPUTS")
    print("="*80)
    
    # Load ALL UNIQUE patients for global input rendering (all 5 folds combined)
    lm_suffix = args.lm_typ.upper()  # 'O' or 'C'
    all_csvs = []
    for i in range(5):  # Load each fold ONCE
        fold_csv = os.path.join(args.csv_folder, f"data_{jaw_suffix}_fold_{i}_{lm_suffix}.csv")
        all_csvs.append(fold_csv)
    
    df_all = pd.concat([pd.read_csv(csv) for csv in all_csvs], ignore_index=True)
    # Remove duplicates (same patient might appear in multiple folds if not properly split)
    df_all = df_all.drop_duplicates(subset=['surf'], keep='first')
    
    all_data, _ = GenDataSet(df_all, args.dir_patients, FlyByDataset, 'cpu', landmark_type=lm_suffix, jaw=jaw)
    
    all_dataloader = DataLoader(
        all_data,
        batch_size=batch_siz,
        shuffle=False,
        collate_fn=pad_verts_faces,
        num_workers=min(8, os.cpu_count() or 4),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    agent_prerender = Agent(
        renderer=phong_renderer,
        renderer2=mask_renderer,
        radius=args.sphere_radius,
        camera_positions=GV.dic_cam[lm_suffix][jaw],
    )
    
    pre_render_all_inputs_only(all_dataloader, agent_prerender, lst_label, jaw, args.lm_typ)

    # === CROSS-VALIDATION SETUP ===
    n_folds = 5
    fold_results = {}
    
    for fold_idx in range(1):
        print("\n" + "="*80)
        print(f"                    FOLD {fold_idx + 1}/{n_folds}")
        print("="*80)
        
        # Load fold CSVs with lm_typ suffix (_O_ or _C_)
        lm_suffix = args.lm_typ.upper()  # 'O' or 'C'
        fold_val_csv = os.path.join(args.csv_folder, f"data_{jaw_suffix}_fold_{fold_idx}_{lm_suffix}.csv")
        
        # Train = all other folds
        train_csvs = []
        for i in range(n_folds):
            if i != fold_idx:
                train_csv = os.path.join(args.csv_folder, f"data_{jaw_suffix}_fold_{i}_{lm_suffix}.csv")
                train_csvs.append(train_csv)
        
        # Load data
        df_val = pd.read_csv(fold_val_csv)
        df_train_list = [pd.read_csv(csv) for csv in train_csvs]
        df_train = pd.concat(df_train_list, ignore_index=True)
        
        train_data, _ = GenDataSet(df_train, args.dir_patients, FlyByDataset, 'cpu', landmark_type=lm_suffix, jaw=jaw)
        val_data, _ = GenDataSet(df_val, args.dir_patients, FlyByDataset, 'cpu', landmark_type=lm_suffix, jaw=jaw)
        
        print(f"[{fold_idx+1}/5] Dataset loaded | Train: {len(train_data)} | Val: {len(val_data)}")

        # Configure DataLoader with GPU optimizations
        optimal_workers = min(8, os.cpu_count() or 4)
        train_dataloader = DataLoader(
            train_data, 
            batch_size=batch_siz, 
            shuffle=True, 
            collate_fn=pad_verts_faces,
            num_workers=optimal_workers,
            pin_memory=True,
            persistent_workers=(optimal_workers > 0),
            prefetch_factor=2
        )

        val_dataloader = DataLoader(
            val_data, 
            batch_size=batch_siz, 
            shuffle=False,
            collate_fn=pad_verts_faces,
            num_workers=optimal_workers,
            pin_memory=True,
            persistent_workers=(optimal_workers > 0),
            prefetch_factor=2
        )        
        
        agent = Agent(
            renderer=phong_renderer,
            renderer2=mask_renderer,
            radius=args.sphere_radius,
            camera_positions=GV.dic_cam[lm_suffix][jaw],
        )
        print(f"[{fold_idx+1}/5] DataLoaders created | Batch size: {batch_siz}")
        print(f"[{fold_idx+1}/5] Landmark type: {args.lm_typ.upper()} ({'Occlusal+MB+DB' if args.lm_typ.lower()=='o' else 'Cervical Lingual+Buccal'})")
        pre_render_all_inputs_and_targets(train_dataloader, agent, lst_label, fold_idx, jaw, lm_typ=args.lm_typ, cache_type='train')
        pre_render_all_inputs_and_targets(val_dataloader, agent, lst_label, fold_idx, jaw, lm_typ=args.lm_typ, cache_type='val')

        num_classes = 4 if args.lm_typ == 'O' else 3
        num_cameras = len(GV.dic_cam[lm_suffix][jaw])
        channels_per_camera = 4
        total_in_channels = num_cameras * channels_per_camera

        model = Model(in_channels=total_in_channels, out_channels=num_classes)
        print(f"[{fold_idx+1}/5] Model initialized | Cameras: {num_cameras} | Channels/camera: {channels_per_camera} | Total in channels: {total_in_channels} | Out channels: {num_classes}")

        loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        optimizer = Adam(model.parameters(), args.learning_rate)
        dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        
        post_true = AsDiscrete(to_onehot=num_classes)
        post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
        
        print(f"[{fold_idx+1}/5] Loss, Optimizer and Metrics initialized | LR: {args.learning_rate}")
        
        metric_values = list()
        best_metric = -1
        best_metric_epoch = -1
        writer = SummaryWriter(log_dir=f"{args.dir_models}/fold_{fold_idx}")
        
        # Create fold-specific model directory
        fold_dir_models = os.path.join(args.dir_models, f"fold_{fold_idx}")
        if not os.path.exists(fold_dir_models):
            os.makedirs(fold_dir_models)

        # Initialize early stopping for this fold
        early_stopping = EarlyStopping(patience=args.early_stopping_patience, verbose=True)

        print(f"[{fold_idx+1}/5] Ready! | Max epochs: {args.max_epoch} | Val frequency: {args.val_freq} | Early stopping patience: {args.early_stopping_patience}")
        print("="*80 + "\n")
        best_loss = float('inf')

        for epoch in range(args.max_epoch):
            epoch_start_time = time.time()
            
            Training(
                train_dataloader=train_dataloader,
                train_data=train_data,
                agent=agent,
                epoch=epoch,
                nb_epoch=args.max_epoch,
                model=model,
                optimizer=optimizer,
                loss_function=loss_function,
                lst_label=lst_label,
                writer=writer,
                jawtype=jaw,
                lm_typ=args.lm_typ,
                fold_idx=fold_idx,
            )

            if epoch % args.val_freq == 0:        
                best_metric, best_metric_epoch, val_loss,best_loss = Validation(
                    val_dataloader=val_dataloader,
                    epoch=epoch,
                    nb_epoch=args.max_epoch,
                    model=model,
                    agent=agent,
                    lst_label=lst_label,
                    dice_metric=dice_metric,
                    best_metric=best_metric,
                    best_metric_epoch=best_metric_epoch,
                    nb_val=epoch,
                    writer=writer,
                    write_image_interval=1,
                    post_true=post_true,
                    jawtype=jaw,
                    lm_typ=args.lm_typ,
                    post_pred=post_pred,
                    metric_values=metric_values,
                    dir_models=fold_dir_models,
                    loss_function=loss_function,
                    fold_idx=fold_idx,
                    best_loss = best_loss
                )
                
                # Check early stopping condition based on validation loss
                if early_stopping(val_loss, epoch):
                    print(f"\nEARLY STOPPING TRIGGERED | Fold {fold_idx+1} | Epoch {epoch+1}")
                    print(f"Best validation loss: {early_stopping.best_loss:.6f} at epoch {early_stopping.best_epoch}")
                    print(f"No improvement for {args.early_stopping_patience} validations\n")
                    break
            
            epoch_time = time.time() - epoch_start_time
            print(f"→ Fold {fold_idx+1} | Epoch {epoch + 1}/{args.max_epoch} completed in {epoch_time:.2f}s\n")
        
        # Store fold results
        fold_results[fold_idx] = {
            'val_loss': val_loss,
            'best_metric_epoch': best_metric_epoch
        }
        print(f"✓ Fold {fold_idx + 1} completed | Best Val Loss: {val_loss:.4f} at epoch {best_metric_epoch}\n")
    
    # === SUMMARY ===
    total_time = time.time() - main_start_time
    print("="*80)
    print("                  CROSS-VALIDATION SUMMARY")
    print("="*80)
    for fold_idx, results in fold_results.items():
        print(f"Fold {fold_idx + 1}: Best Val Loss = {results['val_loss']:.4f} @ epoch {results['best_metric_epoch']}")
    
    avg_metric = np.mean([r['val_loss'] for r in fold_results.values()])
    print(f"\nAverage Best Val Loss: {avg_metric:.4f}")
    print(f"Total time: {total_time:.2f}s ({total_time/3600:.2f}h)")
    print("="*80 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ALIDDM Training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_param = parser.add_argument_group('input files')
    input_param.add_argument('--dir_project', type=str, help='dataset directory', default='/home/luciacev/Desktop/training ios files/all data')
    input_param.add_argument('--dir_data', type=str, help='Input directory', default='/home/luciacev/Desktop/training ios files/all data')
    input_param.add_argument('--dir_patients', type=str, help='Meshes directory', default='/home/luciacev/Desktop/training ios files/all data')
    input_param.add_argument('--csv_folder', type=str, help='Folder containing CSV folds for cross-validation', default='/home/luciacev/Desktop/training ios files/all data/csv files')

    input_param.add_argument('-j','--jaw', type=str, default="L")
    input_param.add_argument('-lm', '--lm_typ', type=str, default="C", choices=['O', 'C'], help="Landmark type: 'O' for Occlusal (O+MB+DB, 3 landmarks) or 'C' for Cervical (CL+CB, 2 landmarks)")
    input_param.add_argument('-sr', '--sphere_radius', type=float, default=0.25)
    input_param.add_argument('--lst_label_l', type=list, default=["18","19","20","21","22","23","24","25","26","27","28","29","30","31"])
    input_param.add_argument('--lst_label_u', type=list, default=["2","3","4","5","6","7","8","9","10","11","12","13","14","15"])

    input_param.add_argument('--num_device', type=str, default='0')
    input_param.add_argument('--image_size', type=int, default=224)
    input_param.add_argument('--blur_radius', type=int, default=0)
    input_param.add_argument('--faces_per_pixel', type=int, default=1)
    
    input_param.add_argument('-bs', '--batch_size', type=int, default=4)
    input_param.add_argument('-nc', '--num_classes', type=int, default=4)
    input_param.add_argument('-me', '--max_epoch', type=int, default=300)
    input_param.add_argument('-vf', '--val_freq', type=int, default=1)
    input_param.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    input_param.add_argument('-es', '--early_stopping_patience', type=int, default=20,help='Number of epochs with no improvement before early stopping')
    input_param.add_argument('--dir_models', type=str, default='/home/luciacev/Desktop/training ios files/all data/models/Upper')

    args = parser.parse_args()
    main(args)