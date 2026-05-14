from monai.networks.nets import UNet
from Agent_class import *
from ALIDDM_utils import *
from monai.data import decollate_batch
import random
import torch
import time
import os

# Global cache directory for all cache files
CACHE_BASE_DIR = '/media/luciacev/Data/ALI_IOS cache'

def Model(in_channels, out_channels):
    """Create UNet model with specified channels."""
    net = UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2, 2),
        num_res_units=4
    ).to(GV.DEVICE)
    return net

def Training(train_dataloader, train_data, agent, epoch, nb_epoch, model, optimizer, loss_function, lst_label, writer, jawtype, lm_typ='o', fold_idx=0):
    """
    Training loop for one epoch.
    Loads pre-cached inputs and targets from disk.
    """
    print(f'-------- TRAINING EPOCH {epoch + 1}/{nb_epoch} --------')
    epoch_start_time = time.time()
    scaler = torch.amp.GradScaler('cuda')          
    model.train()
    epoch_loss = 0
    lm_type_dir = "cervical" if lm_typ == "C" else "occlusal"
    
    # Cache directories for this fold
    # INPUTS: Global (reused across all modes and folds) - NO fold_idx in path
    # TARGETS: Per-mode (separate for O and C) - includes fold_idx and lm_typ
    input_dir = os.path.join(CACHE_BASE_DIR,lm_type_dir, f'global_inputs_{jawtype}')
    target_dir = os.path.join(CACHE_BASE_DIR,lm_type_dir, f'fold_{fold_idx}_targets_train_{jawtype}_{lm_typ}')

    for batch_idx, (S, V, F, RI, CN, LP, MR, SF) in enumerate(train_dataloader):
        batch_start_time = time.time()
        optimizer.zero_grad()
        
        # Sample labels for this batch
        selected_labels = random.sample(lst_label, k=min(4, len(lst_label)))
        
        batch_cum_loss = 0
        
        for label in lst_label:
            # Load INPUT from cache
            inputs_list = []
            for i in range(len(S)):
                patient_name = os.path.basename(S[i]).split('.')[0]
                input_path = os.path.join(input_dir, f"input_{patient_name}_{label}.pth")
                input_tensor = torch.load(input_path, weights_only=True).to(GV.DEVICE)
                inputs_list.append(input_tensor)
            
            inputs_raw = torch.stack(inputs_list)  # [B, Cam, C, H, W]
            B, Cam, C, H, W = inputs_raw.shape
            inputs = inputs_raw.reshape(B, Cam * C, H, W).float()
            
            # Load TARGET from cache
            targets_list = []
            for i in range(len(S)):
                patient_name = os.path.basename(S[i]).split('.')[0]
                target_path = os.path.join(target_dir, f"target_{patient_name}_{label}.pth")
                target_tensor = torch.load(target_path, weights_only=True).to(GV.DEVICE)
                targets_list.append(target_tensor)
            
            targets_raw = torch.stack(targets_list)  # [B, Cam, C, H, W]
            
            # Prepare target for loss
            # targets_raw shape: [B, Cam, C, H, W] where C is 3 for RGB (occlusal or cervical landmarks)
            # Take first camera view and convert to multi-class format
            B, Cam, C, H, W = targets_raw.shape
            
            # Convert RGB colors to class labels
            # For occlusal: Red (1,0,0) -> class 1, Green (0,1,0) -> class 2, Blue (0,0,1) -> class 3
            # For cervical: Magenta (1,0,1) -> class 1, Cyan (0,1,1) -> class 2
            target_first_view = targets_raw[:, 0, :, :, :]  # [B, C, H, W]
            
            # Create class map: background=0, landmark1=1, landmark2=2, landmark3=3
            y_true = torch.zeros(B, 1, H, W, device=GV.DEVICE, dtype=torch.long)
            
            # Threshold to detect landmarks (if any channel > 0.5)
            landmark_mask = (target_first_view.sum(dim=1, keepdim=True) > 0.5).float()
            
            # For multi-class, we need to extract which class each pixel belongs to
            # Based on color: R, G, B channels or R+B, G+B for cervical
            red_channel = target_first_view[:, 0:1, :, :]
            green_channel = target_first_view[:, 1:2, :, :]
            blue_channel = target_first_view[:, 2:3, :, :]
            
            # Classify pixels by dominant color
            class1_mask = (red_channel > 0.5) & (green_channel < 0.5) & (blue_channel < 0.5)  # Red or Magenta
            class2_mask = (red_channel < 0.5) & (green_channel > 0.5) & (blue_channel < 0.5)  # Green or Cyan
            class3_mask = (red_channel < 0.5) & (green_channel < 0.5) & (blue_channel > 0.5)  # Blue only
            
            y_true[class1_mask] = 1
            y_true[class2_mask] = 2
            y_true[class3_mask] = 3
            
            if y_true.dim() == 4:
                y_true = y_true.squeeze(1)  # Remove channel dimension for DiceCELoss
            
            # Forward pass and backward (with AMP)
            with torch.amp.autocast('cuda', enabled=True):
                outputs = model(inputs)
                loss = loss_function(outputs, y_true)
                loss = loss / len(lst_label)

            scaler.scale(loss).backward()
            batch_cum_loss += loss.item()

        # Update weights (once per batch)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += batch_cum_loss
        batch_time = time.time() - batch_start_time
        print(f"  Batch {batch_idx+1:3d}/{len(train_dataloader)} | Loss: {batch_cum_loss:.4f} | Time: {batch_time:.2f}s")

    avg_loss = epoch_loss / (batch_idx + 1)
    elapsed_time = time.time() - epoch_start_time
    print(f"✓ EPOCH {epoch+1} COMPLETE | Avg Loss: {avg_loss:.4f} | Time: {elapsed_time:.2f}s")
    writer.add_scalar("training_loss", avg_loss, epoch + 1)

def Validation(val_dataloader, epoch, nb_epoch, model, agent, lst_label, dice_metric, best_metric, best_metric_epoch, writer, post_pred, post_true, jawtype, lm_typ='o', metric_values=None, nb_val=0, write_image_interval=1, dir_models=None, loss_function=None, fold_idx=0):
    """
    Validation loop for one epoch.
    Loads pre-cached inputs and targets from disk.
    Saves best model when validation loss is minimized.
    """
    print(f'-------- VALIDATION EPOCH {epoch + 1}/{nb_epoch} --------')
    start_val_time = time.time()
    model.eval()
    final_metric = 0
    val_loss = 0
    step = 0
    best_loss = float('inf')  # Track best loss for this fold
    
    # Cache directories for this fold
    # INPUTS: Global (reused across all modes and folds) - NO fold_idx in path
    # TARGETS: Per-mode (separate for O and C) - includes fold_idx and lm_typ
    input_dir = os.path.join(CACHE_BASE_DIR, f'global_inputs_{jawtype}')
    target_dir = os.path.join(CACHE_BASE_DIR, f'fold_{fold_idx}_targets_val_{jawtype}_{lm_typ}')
    
    with torch.no_grad():
        for batch_idx, (S, V, F, RI, CN, LP, MR, SF) in enumerate(val_dataloader):
            batch_start_time = time.time()

            for label in lst_label:
                # Load INPUT from cache
                inputs_list = []
                for i in range(len(S)):
                    patient_name = os.path.basename(S[i]).split('.')[0]
                    input_path = os.path.join(input_dir, f"input_{patient_name}_{label}.pth")
                    input_tensor = torch.load(input_path, weights_only=True).to(GV.DEVICE)
                    inputs_list.append(input_tensor)
                
                inputs = torch.stack(inputs_list)  # [B, Cam, C, H, W]
                
                # Load TARGET from cache
                targets_list = []
                for i in range(len(S)):
                    patient_name = os.path.basename(S[i]).split('.')[0]
                    target_path = os.path.join(target_dir, f"target_{patient_name}_{label}.pth")
                    target_tensor = torch.load(target_path, weights_only=True).to(GV.DEVICE)
                    targets_list.append(target_tensor)
                
                targets = torch.stack(targets_list)  # [B, Cam, C, H, W]
                
                # Prepare target for loss - same as training
                B_val, Cam, C_val, H, W = targets.shape
                B_in, Cam_in, C_in, H_in, W_in = inputs.shape
                
                # Take first camera view and convert to multi-class format
                target_first_view = targets[:, 0, :, :, :]  # [B, C, H, W]
                
                # Create class map: background=0, landmark1=1, landmark2=2, landmark3=3
                y_true_long = torch.zeros(B_val, 1, H, W, device=GV.DEVICE, dtype=torch.long)
                
                # Extract color channels
                red_channel = target_first_view[:, 0:1, :, :]
                green_channel = target_first_view[:, 1:2, :, :]
                blue_channel = target_first_view[:, 2:3, :, :]
                
                # Classify pixels by dominant color
                class1_mask = (red_channel > 0.5) & (green_channel < 0.5) & (blue_channel < 0.5)  # Red or Magenta
                class2_mask = (red_channel < 0.5) & (green_channel > 0.5) & (blue_channel < 0.5)  # Green or Cyan
                class3_mask = (red_channel < 0.5) & (green_channel < 0.5) & (blue_channel > 0.5)  # Blue only
                
                y_true_long[class1_mask] = 1
                y_true_long[class2_mask] = 2
                y_true_long[class3_mask] = 3
                
                if y_true_long.dim() == 4:
                    y_true_long = y_true_long.squeeze(1)  # Remove channel dimension

                B_multi = inputs.shape[0]
                Cam_multi = inputs.shape[1]
                C_multi = inputs.shape[2]
                H_multi = inputs.shape[3]
                W_multi = inputs.shape[4]
                inputs_multi_view = inputs.reshape(B_multi, Cam_multi * C_multi, H_multi, W_multi)
                
                # Model inference
                with torch.amp.autocast('cuda', enabled=False):
                    outputs_pred = model(inputs_multi_view)
                    
                    if loss_function is not None:
                        try:
                            loss = loss_function(outputs_pred, y_true_long)
                            loss_val = loss.item()
                            if not torch.isnan(loss):
                                val_loss += loss_val
                        except Exception as e:
                            print(f"  ⚠️  Loss computation error: {e}")

                # Metrics computation
                val_pred_outputs = [post_pred(i) for i in decollate_batch(outputs_pred)]
                val_true_outputs = [post_true(i) for i in decollate_batch(y_true_long)]
                
                dice_metric(y_pred=val_pred_outputs, y=val_true_outputs)
                step += 1 
            
            batch_time = time.time() - batch_start_time
            print(f"  Batch {batch_idx+1:3d}/{len(val_dataloader)} | Time: {batch_time:.2f}s")

        final_metric = dice_metric.aggregate().item()
        dice_metric.reset()

        metric = final_metric if step > 0 else 0
        avg_val_loss = val_loss / step if step > 0 else 0
        total_val_time = time.time() - start_val_time
        
        print(f"\n✓ VALIDATION COMPLETE | Epoch {epoch + 1}/{nb_epoch}")
        print(f"  Mean Dice: {metric:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {total_val_time:.2f}s")

        # Save model when validation loss improves (is minimized)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_metric_epoch = epoch + 1
            if not os.path.exists(dir_models): 
                os.makedirs(dir_models)
            torch.save(model.state_dict(), os.path.join(dir_models, "best_metric_model.pth"))
            print(f"★ NEW BEST MODEL SAVED | Val Loss: {best_loss:.6f}")
        
        writer.add_scalar("validation_mean_dice", metric, epoch + 1)
        writer.add_scalar("validation_loss", avg_val_loss, epoch + 1)
        
    return best_metric, best_metric_epoch, avg_val_loss