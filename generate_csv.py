#!/usr/bin/env python3
"""
Generate CSV files for cross-validation with 5 folds (Upper and Lower jaws).

Strategy: 10% external test, 90% split into 5 balanced folds - SEPARATE CSVs for Occlusal (_O_) and Cervical (_C_).

Generated files:
  - data_upper_test_O.csv, data_lower_test_O.csv (10% external test - Occlusal)
  - data_upper_test_C.csv, data_lower_test_C.csv (10% external test - Cervical)
  - data_upper_fold_0_O.csv to data_upper_fold_4_O.csv (5 folds - Occlusal)
  - data_upper_fold_0_C.csv to data_upper_fold_4_C.csv (5 folds - Cervical)
  - data_lower_fold_0_O.csv to data_lower_fold_4_O.csv (5 folds - Occlusal)
  - data_lower_fold_0_C.csv to data_lower_fold_4_C.csv (5 folds - Cervical)
"""
import pandas as pd
import os
import glob
from pathlib import Path

data_dir = "/home/luciacev/Desktop/training ios files/mucogingival"
landmarks_dir = os.path.join(data_dir, "landmarks")
vtk_dir = os.path.join(data_dir, "vtk")

# Verify directories exist
if not os.path.exists(data_dir):
    print(f"❌ ERROR: Data directory not found: {data_dir}")
    exit(1)
if not os.path.exists(vtk_dir):
    print(f"❌ ERROR: VTK directory not found: {vtk_dir}")
    exit(1)
if not os.path.exists(landmarks_dir):
    print(f"❌ ERROR: Landmarks directory not found: {landmarks_dir}")
    exit(1)

print(f"✅ Data directory: {data_dir}")
print(f"✅ VTK directory: {vtk_dir}")
print(f"✅ Landmarks directory: {landmarks_dir}\n")

# Create data list
data_list = []

# Iterate through all VTK files (search recursively in subdirectories)
vtk_files = sorted(glob.glob(os.path.join(vtk_dir, "**/*.vtk"), recursive=True))
print(f"Found {len(vtk_files)} VTK files\n")

if len(vtk_files) == 0:
    print("❌ ERROR: No VTK files found!")
    print(f"   Searched in: {vtk_dir}")
    exit(1)

# Build landmark file mapping: extract base filename without extension or landmark type
# E.g., "A10_T1_L_SegOrReg.vtk" -> base "A10_T1_L_SegOrReg"
# Landmarks: "landmarks/Occlusal/A10_T1_L_SegOrReg_Seg_Lower_O_Pred.json" or
#            "landmarks/Cervical/A10_T1_L_SegOrReg_Seg_Lower_C_Pred.json"

for vtk_file in vtk_files:
    vtk_basename = os.path.basename(vtk_file)
    file_base = os.path.splitext(vtk_basename)[0]  # "A10_T1_L_SegOrReg"
    if '_L' in file_base:
        jaw = 'L'
    elif '_U' in file_base:
        jaw = 'U'
    else:
        continue
    surf_rel = os.path.relpath(vtk_file, data_dir)
    # Find matching landmarks in Occlusal subfolder
    occlusal_jsons = sorted(glob.glob(os.path.join(landmarks_dir, "Occlusal", f"{file_base}*_O_*.json")))
    for landmarks_path in occlusal_jsons:
        landmarks_rel = os.path.relpath(landmarks_path, data_dir)
        data_list.append({
            'jaw': jaw,
            'surf': surf_rel,
            'landmarks': landmarks_rel,
            'landmark_type': 'O',
        })
    
    # Find matching landmarks in Cervical subfolder
    cervical_jsons = sorted(glob.glob(os.path.join(landmarks_dir, "Cervical", f"{file_base}*_C_*.json")))
    for landmarks_path in cervical_jsons:
        landmarks_rel = os.path.relpath(landmarks_path, data_dir)
        data_list.append({
            'jaw': jaw,
            'surf': surf_rel,
            'landmarks': landmarks_rel,
            'landmark_type': 'C',
        })
    mucogingival_jsons = sorted(glob.glob(os.path.join(landmarks_dir, f"{file_base}*_MG.mrk.json")))
    print(mucogingival_jsons,file_base)
    for landmarks_path in mucogingival_jsons:
        print(landmarks_path)
        landmarks_rel = os.path.relpath(landmarks_path, data_dir)
        data_list.append({
            'jaw': jaw,
            'surf': surf_rel,
            'landmarks': landmarks_rel,
            'landmark_type': 'MG',
        })

# Verify we have data
if len(data_list) == 0:
    print("❌ ERROR: No data collected!")
    print("   Check if landmarks files exist in:", landmarks_dir)
    exit(1)

print(f"✅ Found {len(data_list)} landmark entries\n")

# Create DataFrame
df_output = pd.DataFrame(data_list)

# Filter by jaw
for jaw_type, jaw_name in [('U', 'upper'), ('L', 'lower')]:
    df_jaw = df_output[df_output['jaw'] == jaw_type].copy().reset_index(drop=True)
    
    # Separate _C and _O for balanced split
    df_c = df_jaw[df_jaw['landmark_type'] == 'C'].copy().reset_index(drop=True)
    df_o = df_jaw[df_jaw['landmark_type'] == 'O'].copy().reset_index(drop=True)
    df_mg = df_jaw[df_jaw['landmark_type'] == 'MG'].copy().reset_index(drop=True)
    
    print(f"\n{'='*80}")
    print(f"Processing {jaw_name.upper()} jaw | Occlusal: {len(df_o)} | Cervical: {len(df_c)}")
    print(f"{'='*80}")
    
    # ========================
    # OCCLUSAL (_O_) - Separate CSV
    # ========================
    
    # 10% test for Occlusal
    test_size_o = max(1, int(len(df_o) * 0.1))
    df_o_test = df_o.iloc[:test_size_o].copy()
    df_o_trainval = df_o.iloc[test_size_o:].copy().reset_index(drop=True)
    
    # Save test CSV for Occlusal
    test_csv_o = os.path.join(data_dir, f"data_{jaw_name}_test_O.csv")
    df_o_test[['surf', 'landmarks']].to_csv(test_csv_o, index=False)
    print(f"✅ {test_csv_o}: {len(df_o_test)} rows (Occlusal - external test)")
    
    # Create 5 folds for Occlusal
    n_folds = 5
    fold_size_o = len(df_o_trainval) // n_folds
    remaining_o = len(df_o_trainval) % n_folds
    
    for fold_idx in range(n_folds):
        size_o = fold_size_o + (1 if fold_idx < remaining_o else 0)
        start_o = fold_idx * fold_size_o + min(fold_idx, remaining_o)
        end_o = start_o + size_o
        
        df_fold_o = df_o_trainval.iloc[start_o:end_o]
        
        fold_csv_o = os.path.join(data_dir, f"data_{jaw_name}_fold_{fold_idx}_O.csv")
        df_fold_o[['surf', 'landmarks']].to_csv(fold_csv_o, index=False)
        print(f"✅ {fold_csv_o}: {len(df_fold_o)} rows (Occlusal - fold {fold_idx})")
    
    # ========================
    # CERVICAL (_C_) - Separate CSV
    # ========================
    
    # 10% test for Cervical
    test_size_c = max(1, int(len(df_c) * 0.1))
    df_c_test = df_c.iloc[:test_size_c].copy()
    df_c_trainval = df_c.iloc[test_size_c:].copy().reset_index(drop=True)
    
    # Save test CSV for Cervical
    test_csv_c = os.path.join(data_dir, f"data_{jaw_name}_test_C.csv")
    df_c_test[['surf', 'landmarks']].to_csv(test_csv_c, index=False)
    print(f"✅ {test_csv_c}: {len(df_c_test)} rows (Cervical - external test)")
    
    # Create 5 folds for Cervical
    fold_size_c = len(df_c_trainval) // n_folds
    remaining_c = len(df_c_trainval) % n_folds
    
    for fold_idx in range(n_folds):
        size_c = fold_size_c + (1 if fold_idx < remaining_c else 0)
        start_c = fold_idx * fold_size_c + min(fold_idx, remaining_c)
        end_c = start_c + size_c
        
        df_fold_c = df_c_trainval.iloc[start_c:end_c]
        
        fold_csv_c = os.path.join(data_dir, f"data_{jaw_name}_fold_{fold_idx}_C.csv")
        df_fold_c[['surf', 'landmarks']].to_csv(fold_csv_c, index=False)
        print(f"✅ {fold_csv_c}: {len(df_fold_c)} rows (Cervical - fold {fold_idx})")

    # ========================
    # MUCOGINGIVAL (_MG_) - Separate CSV
    # ========================
    test_size_c = max(1, int(len(df_mg) * 0.1))
    df_mg_test = df_mg.iloc[:test_size_c].copy()
    df_mg_trainval = df_mg.iloc[test_size_c:].copy().reset_index(drop=True)
    
    # Save test CSV for Cervical
    test_csv_c = os.path.join(data_dir, f"data_{jaw_name}_test_MG.csv")
    df_mg_test[['surf', 'landmarks']].to_csv(test_csv_c, index=False)
    print(f"✅ {test_csv_c}: {len(df_mg_test)} rows (Cervical - external test)")
    
    # Create 5 folds for Cervical
    fold_size_c = len(df_mg_trainval) // n_folds
    remaining_c = len(df_mg_trainval) % n_folds
    
    for fold_idx in range(n_folds):
        size_c = fold_size_c + (1 if fold_idx < remaining_c else 0)
        start_c = fold_idx * fold_size_c + min(fold_idx, remaining_c)
        end_c = start_c + size_c
        
        df_fold_c = df_mg_trainval.iloc[start_c:end_c]
        
        fold_csv_c = os.path.join(data_dir, f"data_{jaw_name}_fold_{fold_idx}_MG.csv")
        df_fold_c[['surf', 'landmarks']].to_csv(fold_csv_c, index=False)
        print(f"✅ {fold_csv_c}: {len(df_fold_c)} rows (Cervical - fold {fold_idx})")


print("\n" + "="*80)
print("✅ CSV files generation completed!")
print("="*80)
print("\nGenerated files:")
print("  Occlusal (_O_):  data_*_test_O.csv, data_*_fold_*_O.csv")
print("  Cervical (_C_):  data_*_test_C.csv, data_*_fold_*_C.csv")
print("\nUse with main.py:")
print("  python main.py --lm_typ o --csv_folder <csv_folder>")
print("  python main.py --lm_typ c --csv_folder <csv_folder>")
print("="*80 + "\n")
