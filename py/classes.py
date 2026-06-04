from torch.utils.data import Dataset
import numpy as np
import os
import torch
import pickle
import GlobVar as GV
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from utils import(
    ReadSurf,
    ScaleSurf,
    RandomRotation,
    ComputeNormals,
    GetColorArray,
    GetTransform
)
import pandas as pd
import json
from torch.nn.utils.rnn import pad_sequence

from vtk import vtkMatrix4x4
from vtk import vtkMatrix3x3
import vtk

# Global cache directory for VTK geometry
CACHE_BASE_DIR = '/media/luciacev/Data/ALI_IOS cache_3channelsout_cam'

class FlyByDataset(Dataset):
    def __init__(self, df, device, dataset_dir='', rotate=False, cache_dir=None, landmark_type='O', jaw='L'):
        self.df = df
        self.dataset_dir = dataset_dir
        self.rotate = rotate
        self.landmark_type = landmark_type  # 'O' for Occlusal or 'C' for Cervical
        self.jaw = jaw  # 'L' for Lower or 'U' for Upper
        
        # Use default cache directory based on landmark type
        if cache_dir is None:
            # Create separate cache subdirectories for Occlusal and Cervical
            lm_subdir = 'occlusal' if landmark_type == 'O' else 'cervical'
            cache_dir = os.path.join(CACHE_BASE_DIR, lm_subdir, 'geom_render')
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir): os.makedirs(self.cache_dir)

    def __getitem__(self, idx):
        surf_path = self.df.iloc[idx]["surf"]
        patient_id = os.path.basename(surf_path).split('.')[0]
        
        # Define cache path for geometric data
        geom_cache = os.path.join(self.cache_dir, f"geom_{patient_id}.pth")

        if os.path.exists(geom_cache):
            try:
                cached_result = torch.load(geom_cache, weights_only=True)
                # Ensure surf_path is included (handle old cache format)
                if cached_result[0] is None:
                    return (surf_path,) + cached_result[1:]
                return cached_result
            except (RuntimeError, EOFError, pickle.UnpicklingError, OSError, ValueError) as e:
                # Cache file is corrupted, remove it and regenerate
                print(f"\n⚠️  Corrupted cache file detected: {geom_cache}")
                print(f"    Error: {type(e).__name__}: {e}")
                print(f"    Removing and regenerating cache...\n")
                try:
                    os.remove(geom_cache)
                except Exception as remove_error:
                    print(f"    Could not remove cache file: {remove_error}\n")
                # Fall through to regenerate cache below

        # If no cache, perform standard VTK processing
        surf = ReadSurf(os.path.join(self.dataset_dir, surf_path))
        surf, mean_arr, scale_factor = ScaleSurf(surf)
        surf = ComputeNormals(surf) 
        
        verts = torch.from_numpy(vtk_to_numpy(surf.GetPoints().GetData())).float()
        faces = torch.from_numpy(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:, 1:]).long()
        
        # Handle missing Universal_ID gracefully
        try:
            region_id_array = surf.GetPointData().GetScalars("Universal_ID")
            if region_id_array is None:
                # If Universal_ID doesn't exist, create default array
                full_path = os.path.join(self.dataset_dir, surf_path)
                print(f"\n⚠️  WARNING: Universal_ID not found in:")
                print(f"    {full_path}\n")
                region_id = torch.zeros(len(verts), dtype=torch.long)
            else:
                region_id = torch.from_numpy(vtk_to_numpy(region_id_array)).long()
        except Exception as e:
            full_path = os.path.join(self.dataset_dir, surf_path)
            print(f"\n❌ ERROR reading Universal_ID from:")
            print(f"    {full_path}")
            print(f"    Error: {e}\n")
            region_id = torch.zeros(len(verts), dtype=torch.long)
        
        color_normals = torch.from_numpy(vtk_to_numpy(GetColorArray(surf, "Normals"))/255.0).float()
        
        # Landmarks positions - use correct camera based on landmark type and jaw
        cam_pos = GV.dic_cam[self.landmark_type][self.jaw]
        # For now, we'll use the first camera position as the main one
        angle = 0
        vector = cam_pos[0] if isinstance(cam_pos[0], np.ndarray) else np.array(cam_pos[0])
        
        lp = self.get_landmarks_position(idx, mean_arr, scale_factor, angle, vector)

        result = (
            surf_path, verts, faces, region_id, color_normals, 
            torch.from_numpy(lp).float(),
            torch.from_numpy(np.array(mean_arr)).double(),
            torch.from_numpy(np.array(scale_factor)).double()
        )
        
        torch.save(result, geom_cache)
        return result

    def set_env_params(self, params):
        self.params = params

    def __len__(self):
        return len(self.df)
   
    def get_landmarks_position(self, idx, mean_arr, scale_factor, angle, vector):
        """Load landmarks from JSON file, handling cases where not all 5 points per tooth exist"""
        landmark_path = os.path.join(self.dataset_dir, self.df.iloc[idx]["landmarks"])
        
        data = json.load(open(landmark_path))
        markups = data['markups']
        landmarks_lst = markups[0]['controlPoints']
        lst_lm = GV.LANDMARKS[self.jaw]
        landmarks_position = np.zeros([len(lst_lm), 3])
        
        found_count = 0
        for landmark in landmarks_lst:
            label = landmark["label"]
            if label in lst_lm:
                raw_pos = np.array(landmark["position"])
                scaled_pos = Downscale(raw_pos, mean_arr, scale_factor)
                landmarks_position[lst_lm.index(label)] = scaled_pos
                found_count += 1

        landmarks_pos = np.array([np.append(pos, 1) for pos in landmarks_position])
        if angle:
            transform = GetTransform(angle, vector)
            transform_matrix = arrayFromVTKMatrix(transform.GetMatrix())
            landmarks_pos = np.matmul(transform_matrix, landmarks_pos.T).T
            
        return landmarks_pos[:, 0:3]

def pad_verts_faces(batch):
    """
    Collate function for DataLoader to group batch data.
    Ensures all data remains on CPU.
    """
    surf = [s for s, v, f, ri, cn, lp, sc, ma in batch]
    verts = [v for s, v, f, ri, cn, lp, sc, ma in batch]
    faces = [f for s, v, f, ri, cn, lp, sc, ma in batch]
    region_id = [ri for s, v, f, ri, cn, lp, sc, ma in batch]
    color_normals = [cn for s, v, f, ri, cn, lp, sc, ma in batch]
    landmark_position = [lp for s, v, f, ri, cn, lp, sc, ma in batch]
    scale_factor = [sc for s, v, f, ri, cn, lp, sc, ma in batch]
    mean_arr = [ma for s, v, f, ri, cn, lp, sc, ma in batch]

    return (surf, 
            pad_sequence(verts, batch_first=True, padding_value=0.0), 
            pad_sequence(faces, batch_first=True, padding_value=-1),
            region_id, 
            pad_sequence(color_normals, batch_first=True, padding_value=0.), 
            landmark_position, 
            mean_arr, 
            scale_factor)

def arrayFromVTKMatrix(vmatrix):
    if isinstance(vmatrix, vtkMatrix4x4):
        matrixSize = 4
    elif isinstance(vmatrix, vtkMatrix3x3):
        matrixSize = 3
    else:
        raise RuntimeError("Input must be vtk.vtkMatrix3x3 or vtk.vtkMatrix4x4")
    
    narray = np.eye(matrixSize)
    # vmatrix.GetElement(row, col) is the safe way to extract data
    for i in range(matrixSize):
        for j in range(matrixSize):
            narray[i, j] = vmatrix.GetElement(i, j)
    return narray

def Upscale(landmark_pos, mean_arr, scale_factor):
    return (landmark_pos / scale_factor) + mean_arr

def Downscale(pos_center, mean_arr, scale_factor):
    return (pos_center - mean_arr) * scale_factor