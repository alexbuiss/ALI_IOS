import os
import glob
import json
import csv
from scipy.sparse.construct import random
from sklearn.model_selection import train_test_split
import random
import vtk
import torch
import pandas as pd
import GlobVar as GV
from utils import ReadSurf
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import numpy as np
from utils import(
    PolyDataToTensors
) 
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings, MeshRenderer, MeshRasterizer,
    HardPhongShader, PointLights,
)
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex,blending

from torch.utils.data import DataLoader

from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from monai.transforms import (
    ToTensor
)
import monai
import torchvision.transforms as transforms
from shader import *
from post_process import NeighborPoints

def GetLandmarkPosFromLP(lm_pos, target, jaw):
    lst_lm = GV.LANDMARKS[jaw]
    # Vérifie si la target existe dans la liste globale

    if target not in lst_lm:
        print(f"DEBUG: Target {target} non trouvée dans GlobVar!")
        return torch.zeros((len(lm_pos), 3)) # Retourne des zéros si pas trouvé

    lm_coord = torch.empty((0)).cpu()
    for i, lst in enumerate(lm_pos):
        idx = lst_lm.index(target)
        coord = lst[idx].unsqueeze(0)
        # DEBUG: Est-ce que le point extrait est nul ?
        if coord.abs().sum() < 1e-5:
             print(f"DEBUG: Point nul extrait pour {target}, patient {i}")
        lm_coord = torch.cat((lm_coord, coord.cpu()), dim=0)

    return lm_coord

def GenPhongRenderer(image_size,blur_radius,faces_per_pixel,device):
    
    # cameras = FoVOrthographicCameras(znear=0.1,zfar = 10,device=device) # Initialize a ortho camera.

    cameras = FoVPerspectiveCameras(znear=0.01,zfar = 10, fov= 90, device=device) # Initialize a perspective camera.

    raster_settings = RasterizationSettings(        
        image_size=image_size, 
        blur_radius=blur_radius, 
        faces_per_pixel=faces_per_pixel, 
    )

    lights = PointLights(device=device) # light in front of the object. 

    rasterizer = MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        )
    
    b = blending.BlendParams(background_color=(0,0,0))
    phong_renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=HardPhongShader(device=device, cameras=cameras, lights=lights,blend_params=b)
    )
    mask_renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=MaskRenderer(device=device, cameras=cameras, lights=lights,blend_params=b)
    )
    return phong_renderer,mask_renderer
    

def GenDataSet(df, dir_patients, flyBy, device, landmark_type='O', jaw='L'):
    """
    Generate training and validation datasets from a DataFrame.
    
    For cross-validation, pass the entire df (all rows will be used for the dataset).
    For legacy mode, the df should have a 'for' column with 'train' and 'val' values.
    
    Args:
        df: DataFrame with 'surf' and 'landmarks' columns
        dir_patients: Path to the data directory
        flyBy: Dataset class (FlyByDataset)
        device: Device to use ('cpu' or 'cuda')
        landmark_type: 'O' for Occlusal or 'C' for Cervical
        jaw: 'L' for Lower or 'U' for Upper
    
    Returns:
        (dataset, None) - For cross-validation, returns dataset and None
    """
    # Check if this is legacy mode with 'for' column or cross-validation mode
    if 'for' in df.columns:
        # Legacy mode: split by 'for' column
        df_train = df.loc[df['for'] == "train"]
        df_train = df_train.loc[df_train['jaw'] == jaw]
        df_val = df.loc[df['for'] == "val"]
        df_val = df_val.loc[df_val['jaw'] == jaw]
        train_data = flyBy(
            df=df_train,
            device=device,
            dataset_dir=dir_patients,
            rotate=False,
            landmark_type=landmark_type,
            jaw=jaw
        )

        val_data = flyBy(
            df=df_val,
            device=device,
            dataset_dir=dir_patients,
            rotate=False,
            landmark_type=landmark_type,
            jaw=jaw
        )

        return train_data, val_data
    else:
        # Cross-validation mode: use entire df as dataset
        df_data = df.copy()
        data = flyBy(
            df=df_data,
            device=device,
            dataset_dir=dir_patients,
            rotate=False,
            landmark_type=landmark_type,
            jaw=jaw
        )

        return data, None

def generate_sphere_mesh(center,radius,device,color = [1,1,1]):
    sphereSource = vtk.vtkSphereSource()
    sphereSource.SetCenter(center[0],center[1],center[2])
    sphereSource.SetRadius(radius)

    # Make the surface smooth.
    sphereSource.SetPhiResolution(10)
    sphereSource.SetThetaResolution(10)
    sphereSource.Update()
    vtk_landmarks = vtk.vtkAppendPolyData()
    vtk_landmarks.AddInputData(sphereSource.GetOutput())
    vtk_landmarks.Update()

    verts_teeth,faces_teeth = PolyDataToTensors(vtk_landmarks.GetOutput())

    verts_rgb = torch.ones_like(verts_teeth)[None]  # (1, V, 3)
    # verts_rgb[:,0:] *= 0.1
    verts_rgb[:,:, 0] *= color[0]  # red
    verts_rgb[:,:, 1] *= color[1]  # green
    verts_rgb[:,:, 2] *= color[2]  # blue


    # verts_rgb[:,:2] *= 0


    # color_normals = ToTensor(dtype=torch.float32, device=self.device)(vtk_to_numpy(fbf.GetColorArray(surf, "Normals"))/255.0)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))
    mesh = Meshes(
        verts=[verts_teeth], 
        faces=[faces_teeth],
        textures=textures).to(device)
    
    return mesh,verts_teeth,faces_teeth,verts_rgb.squeeze(0)

def SplitCSV_train_Val(csvPath,val_p):
    df = pd.read_csv(csvPath)
    df_test = df.loc[df['for'] == "test"]
    df_train = df.loc[df['for'] == "train"]
    samples = int(len(df_train.index)*val_p)

    for i in range(samples):
        random_num = random.randint(1, 131)
        # print(random_num)
        df_train['for'][random_num] = "val"

    # df.to_csv(csvPath,index=False)

    df_fold = pd.concat([df_train, df_test])
    # print(df_fold['for'])
    df_fold.to_csv(csvPath,index=False)
    

def GenDataSplitCSV(dir_data,csv_path,val_p,test_p):
    patient_dic = {}
    normpath = os.path.normpath("/".join([dir_data, '**', '']))
    for img_fn in sorted(glob.iglob(normpath, recursive=True)):
        if os.path.isfile(img_fn):
            basename = os.path.basename(img_fn).split('.')[0].split("_P")
            jow = basename[0][0]
            patient_ID = "P"+basename[1]
            if patient_ID not in patient_dic.keys():
                patient_dic[patient_ID] = {"L":{},"U":{}}
            
            if ".json" in img_fn:
                patient_dic[patient_ID][jow]["lm"] = img_fn
            elif ".vtk" in img_fn:
                patient_dic[patient_ID][jow]["surf"] = img_fn

    # print(patient_dic)
    test_p = test_p/100
    val_p = val_p/100
    val_p = val_p/(1-test_p)

    patient_dic = list(patient_dic.values())
    random.shuffle(patient_dic)

    # print(patient_dic)
    df_train, df_test = train_test_split(patient_dic, test_size=test_p)
    df_train, df_val = train_test_split(df_train, test_size=val_p)

    data_dic = {
        "train":df_train,
        "val":df_val,
        "test":df_test
        }
    # print(data_dic)
    fieldnames = ['for','jaw','surf', 'landmarks']
    for lab in range(2,32):
        fieldnames.append(str(lab))
    data_list = []
    for type,dic in data_dic.items():
        for patient in dic:
            for jaw,data in patient.items():
                if jaw == "U":
                    rows = {
                        'for':type,
                        'jaw':jaw,
                        'surf':data["surf"].replace(dir_data,"")[1:],
                        'landmarks':data["lm"].replace(dir_data,"")[1:],
                        }
                    # print(data["surf"])
                    read_surf = ReadSurf(data["surf"])
                    ids = ToTensor(dtype=torch.int64, device=GV.DEVICE)(vtk_to_numpy(read_surf.GetPointData().GetScalars("PredictedID")))
                    # print(ids)

                    for label in range(2,32):
                        
                        if label in ids:
                            present = 1
                        else:
                            present = 0
                        
                        rows[str(label)] = present

                    data_list.append(rows)
    
    with open(csv_path, 'w', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_list)
    # return outfile


def PlotDatasetWithLandmark(target,dataLoader):
    for batch, (V, F, CN, LP, MR, SF) in enumerate(dataLoader):
        radius = 0.02
        textures = TexturesVertex(verts_features=CN)
        meshes = Meshes(
            verts=V,   
            faces=F, 
            textures=textures
        )

        lm_pos = torch.empty((0)).to(GV.DEVICE)
        for lst in LP:
            lm_pos = torch.cat((lm_pos,lst[GV.LM_SELECTED_LST.index(target)].unsqueeze(0)),dim=0)

        # print(lm_pos[0])
    PlotMeshAndSpheres(meshes,lm_pos,radius,[0.3,1,0.3])

def PlotMeshAndSpheres(meshes,sphere_pos,radius,col):
    dic = {
    "teeth_mesh": meshes,
    }
    for id,pos in enumerate(sphere_pos):
        # print(pos)
        mesh,verts_teeth,faces_teeth,textures = generate_sphere_mesh(pos,radius,GV.DEVICE,col)
        dic[str(id)] = mesh
    # print(dic)
    plot_fig(dic)


def plot_fig(dic):
    fig = plot_scene({"subplot1": dic},     
        xaxis={"backgroundcolor":"rgb(200, 200, 230)"},
        yaxis={"backgroundcolor":"rgb(230, 200, 200)"},
        zaxis={"backgroundcolor":"rgb(200, 230, 200)"}, 
        axis_args=AxisArgs(showgrid=True))
    fig.show()


def Generate_Mesh(verts,faces,text,lst_landmarks,device):
    verts_rgb = torch.ones_like(text)[None].squeeze(0)  # (1, V, 3)
    verts_rgb[:,:, 0] *= 1  # red
    verts_rgb[:,:, 1] *= 0  # green
    verts_rgb[:,:, 2] *= 0  # blue
    text = verts_rgb

    for landmark in lst_landmarks:
        batch_verts = torch.empty((0)).to(device)
        batch_faces = torch.empty((0)).to(device)
        batch_text = torch.empty((0)).to(device)
        # print(text)
        for position in landmark:
            mesh_l,verts_l,faces_l,text_l = generate_sphere_mesh(position,0.01,device,[0,1,0])
            batch_text = torch.cat((batch_text,text_l.unsqueeze(0).to(device)),dim=0)
            batch_verts = torch.cat((batch_verts,verts_l.unsqueeze(0).to(device)),dim=0)
            batch_faces = torch.cat((batch_faces,faces_l.unsqueeze(0).to(device)),dim=0)
       
        verts,faces,text = merge_meshes(verts,faces,text,batch_verts,batch_faces,batch_text)
    
    textures = TexturesVertex(verts_features=text)
        
    meshes =  Meshes(
        verts=verts,   
        faces=faces, 
        textures=textures
    )
    return meshes


def merge_meshes(verts_1,faces_1,text_1,verts_2,faces_2,text_2):

    verts = torch.cat([verts_2,verts_1], dim=1)
    faces = torch.cat([faces_2,faces_1+verts_2.shape[1]], dim=1)
    text = torch.cat([text_2,text_1], dim=1)

    return verts,faces,text

def Get_lst_landmarks(LP, lst_names_land, jaw):
    lst_landmarks=[]
    print(lst_names_land)
    for landmarks in lst_names_land:
        lm_coords = GetLandmarkPosFromLP(LP, landmarks, jaw)
        lst_landmarks.append(lm_coords)
    return lst_landmarks

def Generate_land_Mesh(lst_landmarks,device):
    verts = torch.empty((0)).to(device)
    faces = torch.empty((0)).to(device)
    text = torch.empty((0)).to(device)
    
    for landmark in lst_landmarks:
        batch_verts = torch.empty((0)).to(device)
        batch_faces = torch.empty((0)).to(device)
        batch_text = torch.empty((0)).to(device)
    
        for position in landmark:
            mesh_l,verts_l,faces_l,text_l = generate_sphere_mesh(position,0.02,device)
            tensor_text = torch.ones_like(verts_l).to(device)*0
            batch_text = torch.cat((batch_text,tensor_text.unsqueeze(0).to(device)),dim=0)
            batch_verts = torch.cat((batch_verts,verts_l.unsqueeze(0).to(device)),dim=0)
            batch_faces = torch.cat((batch_faces,faces_l.unsqueeze(0).to(device)),dim=0)
       
        verts,faces,text = merge_meshes(verts,faces,text,batch_verts,batch_faces,batch_text)
    
    textures = TexturesVertex(verts_features=text)
        
    meshes =  Meshes(
        verts=verts,   
        faces=faces, 
        textures=textures
    )
    return meshes

def Convert_RGB_to_grey(lst_images):
    tens_images = torch.empty((0)).to(GV.DEVICE)
    for image in lst_images:
        # print(image.shape)
        # image = image.to(GV.DEVICE)
        image = image.cpu()
        image = image[:-1,:,:]
        image = transforms.ToPILImage()(image)
        image = transforms.Grayscale(num_output_channels=1)(image)
        image = transforms.ToTensor()(image)
        # print(image[0])
        for row in image[0]:
            for pix in row:
                new_pix = torch.nn.Threshold(pix>0.5, 1)
                pix = new_pix

        tens_images = torch.cat((tens_images,image.unsqueeze(0).to(GV.DEVICE)),dim=0)

    return tens_images

def Gen_patch(V, RED, LP, label, radius, batch_idx, jaw, lm_typ='o'):
    V_coords = V[0].to(GV.DEVICE)
    # Filtre pour ignorer les points de padding (0,0,0) de la dent
    real_surface_mask = torch.linalg.norm(V_coords, dim=1) > 1e-4
    lst_landmarks = Get_lst_landmarks(LP, GV.dic_label[lm_typ][label], jaw)
    # Couleurs pures : R(1,0,0), V(0,1,0), B(0,0,1)
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device=GV.DEVICE)

    colored_landmarks = 0  # Track how many landmarks were actually colored
    for color_index, all_patients_coords in enumerate(lst_landmarks):
        landmark_coord = all_patients_coords[batch_idx].view(1, 3).to(GV.DEVICE)
        print(f"DEBUG Patient {batch_idx} - Label {label} - Coordonnées Target: {landmark_coord.cpu().numpy()}")
        # SI LE LANDMARK EST À L'ORIGINE (0,0,0), ON SKIP TOTALEMENT
        # C'est ce qui supprimera ton point jaune central.
        if torch.norm(landmark_coord) < 0.05:
            print(f"⚠️  Landmark {GV.dic_label[lm_typ][label][color_index]} is missing (all zeros)")
            continue 
            
        distances = torch.cdist(V_coords, landmark_coord, p=2).view(-1)
        # On ne colorie que si c'est proche ET sur la vraie surface
        mask = (distances < radius) & real_surface_mask
        
        if mask.any():
            print("on colorie", len(mask.nonzero()))
            # S'assurer que l'indexation se fait explicitement sur la dimension des sommets (dim 1)
            color_to_apply = colors[color_index % len(colors)]
            RED[0, mask, :] = color_to_apply
            colored_landmarks += 1
        else:
            print(f"⚠️  No vertices colored for landmark {GV.dic_label[lm_typ][label][color_index]}")
    
    if colored_landmarks == 0:
        print(f"⚠️  WARNING: No landmarks were colored for label {label}! Patch will be completely black.")
            
    return RED

def Gen_mesh_patch(surf, V, F, CN, LP, label, batch_idx, jaw, lm_typ='o'):
    # INITIALISATION : On force tout à ZÉRO (Noir total)
    # On ne met pas 0.1 ou 0.05, sinon la silhouette de la dent apparaît.
    verts_rgb = torch.zeros_like(V).to(GV.DEVICE) 
    
    # Appel de Gen_patch pour colorier uniquement les landmarks valides
    patch_region = Gen_patch(V, verts_rgb, LP, label, 0.02, batch_idx, jaw, lm_typ=lm_typ)    
    
    # Sécurité : Si Gen_patch échoue, on renvoie le noir
    if patch_region is None: patch_region = verts_rgb

    textures = TexturesVertex(verts_features=patch_region)
    return Meshes(verts=V, faces=F, textures=textures).to(GV.DEVICE)

def Gen_one_patch(V, RED, radius, coord):
    """
    OPTIMIZED: Vectorized version without loops.
    """
    landmark_coord = coord.unsqueeze(0).to(GV.DEVICE)
    
    # Use broadcasting to avoid loops
    V_flat = V.squeeze(0) if V.dim() == 3 else V
    distance = torch.cdist(landmark_coord, V_flat, p=2).squeeze(0)
    
    # Vectorized mask + atomic assignment
    mask = distance < radius
    RED[mask] = torch.tensor([0.0, 1.0, 0.0], device=GV.DEVICE)
              
    return RED

def Gen_mesh_one_patch(V, F, CN, coord):
    verts_rgb = torch.ones_like(CN)[None].squeeze(0)  # (1, V, 3)
    verts_rgb[:,:, 0] *= 1  # red
    verts_rgb[:,:, 1] *= 0  # green
    verts_rgb[:,:, 2] *= 0  # blue
    patch_region = Gen_one_patch(V, verts_rgb, 0.02, coord)
    textures = TexturesVertex(verts_features=patch_region)
    meshes = Meshes(
        verts=V,   
        faces=F, 
        textures=textures
    ).to(GV.DEVICE)
    
    return meshes

def Gen_Full_Mask_Mesh(S, V, F, RI):
    device = GV.DEVICE
    verts_rgb_list = []
    
    # Prepare V and F on GPU for Meshes constructor
    V_gpu = [v.to(device) for v in V] if isinstance(V, list) else V.to(device)
    F_gpu = [f.to(device) for f in F] if isinstance(F, list) else F.to(device)

    batch_size = len(V) if isinstance(V, list) else V.shape[0]

    for i in range(batch_size):
        ri_patient = RI[i].to(device).float()
        num_verts = V_gpu[i].shape[0]
        
        # Correction de taille si besoin
        if ri_patient.shape[0] != num_verts:
            temp_ri = torch.zeros(num_verts, device=device)
            limit = min(num_verts, ri_patient.shape[0])
            temp_ri[:limit] = ri_patient[:limit]
            ri_patient = temp_ri

        color_patient = torch.zeros((num_verts, 3), device=device)
        color_patient[:, 0] = ri_patient / 255.0
        verts_rgb_list.append(color_patient)
    
    textures = TexturesVertex(verts_features=verts_rgb_list)
    # Le mesh est construit directement avec des composants GPU
    return Meshes(verts=V_gpu, faces=F_gpu, textures=textures)