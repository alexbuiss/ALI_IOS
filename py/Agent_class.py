from operator import itemgetter
import GlobVar as GV
from utils import *
from ALIDDM_utils import *


from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np 
import torchvision.models as models
from pytorch3d.renderer import look_at_rotation
import torch
from monai.transforms import ToTensor
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
import fly_by_features as fbf
import json
from collections import deque
import statistics
import matplotlib.pyplot as plt
import math
import os
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import math
import torch.optim as optim
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from tqdm.std import tqdm
from statistics import mean
import GlobVar as GV

class Agent:
    def __init__(
        self,
        renderer, 
        renderer2,
        radius = 1,
        verbose = True,
        camera_positions = None,
        ):
        super(Agent, self).__init__()
        self.renderer = renderer
        self.renderer2=renderer2
        
        # Use provided camera positions or default
        if camera_positions is None:
            # Fallback: create default camera positions
            camera_positions = np.array([
                [0, 0, 1],
                [0.5, 0., 1.0],
                [-0.5, 0., 1.0],
                [0, 0.5, 1],
                [0, -0.5, 1]
            ], dtype=np.float32)
        else:
            # Ensure it's a numpy array with correct dtype
            camera_positions = np.asarray(camera_positions, dtype=np.float32)
        
        self.camera_points = torch.from_numpy(camera_positions).to(GV.DEVICE)
        self.scale = 0
        self.radius = radius
        self.verbose = verbose


    def position_agent(self, text, vert, label):
        """
        OPTIMISÉE: Calcule la position ET la normale moyenne pour le label donné
        """
        final_pos = torch.empty((0), device=GV.DEVICE)
        
        for mesh_idx in range(len(text)):
            if int(label) in text[mesh_idx]:
                index_pos_land = (text[mesh_idx] == int(label)).nonzero(as_tuple=True)[0]
                if len(index_pos_land) > 0:
                    # Position moyenne
                    lst_pos = vert[mesh_idx][index_pos_land]
                    pos_agent = lst_pos.mean(dim=0)
                    
                else:
                    pos_agent = torch.zeros(3, device=GV.DEVICE)
            else:
                pos_agent = torch.zeros(3, device=GV.DEVICE)
                
            final_pos = torch.cat((final_pos, pos_agent.unsqueeze(0)), dim=0)
        
        self.positions = final_pos
        return final_pos

    
    def GetView(self, meshes, type_lm, rend=False):
        batch_size = len(meshes) 
        device = GV.DEVICE

        # Position du landmark de la dent (batch_size, 1, 3)
        spc = self.positions.view(-1, 1, 3)[:batch_size] 
        
        # 1. CALCUL DU CENTRE DE LA MÂCHOIRE
        verts = meshes.verts_padded() 
        machoire_center = verts.mean(dim=1, keepdim=True) 

        # 2. VECTEUR DE DIRECTION (On calcule en 3D d'abord)
        direction_buccale = spc - machoire_center
        
        # --- DÉTECTION AUTOMATIQUE DE L'AXE VERTICAL ---
        # On regarde si la mâchoire s'étend plutôt en Y ou en Z pour trouver "le haut"
        # Si ton scan a l'axe Y vers le haut :

        # L'axe vertical est Z. On écrase Z pour le calcul horizontal (X, Y)
        direction_buccale[:, :, 2] = 0 
        up_vector = torch.tensor([0.0, 0.0, 1.0], device=device) # Le haut est Z
        hauteur_idx = 2

        # Normalisation stricte de la direction extérieure (longueur = 1.0)
        direction_buccale = direction_buccale / (torch.norm(direction_buccale, dim=-1, keepdim=True) + 1e-6)

        # 3. ROTATION DES 3 CAMÉRAS (Plan horizontal)
        angle = 0.35 
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        dir_face = direction_buccale
        dir_gauche = dir_face.clone()
        dir_droite = dir_face.clone()
        
        # On applique la rotation sur le plan horizontal (X et l'autre axe non vertical)
        plane_idx = 2 if hauteur_idx == 1 else 1
        
        dir_gauche[:, :, 0] = dir_face[:, :, 0] * cos_a - dir_face[:, :, plane_idx] * sin_a
        dir_gauche[:, :, plane_idx] = dir_face[:, :, 0] * sin_a + dir_face[:, :, plane_idx] * cos_a
        
        dir_droite[:, :, 0] = dir_face[:, :, 0] * cos_a - dir_face[:, :, plane_idx] * (-sin_a)
        dir_droite[:, :, plane_idx] = dir_face[:, :, 0] * (-sin_a) + dir_face[:, :, plane_idx] * cos_a

        bonnes_directions = torch.cat([dir_face, dir_gauche, dir_droite], dim=1)
        n_cameras = 3 

        # 4. POSITIONNEMENT DE CÔTÉ (Face à la gencive)
        # On se place à la distance "self.radius"
        current_cam_pos = spc + (bonnes_directions * self.radius)
        
        # COMPORTEMENT DE CÔTÉ : Au lieu de monter très haut, on applique juste une légère 
        # inclinaison sur l'axe vertical pour voir la gencive du bas (légèrement plongeant)
        current_cam_pos[:, :, hauteur_idx] -= (self.radius * 0.15) # Monte de seulement 20% du radius
        
        # La cible (target) est pile sur le landmark
        spc_target = spc.clone()
        offset_gencive = 0.2  # Descend de 4 unités vers la gencive
    
        spc_target[:, :, hauteur_idx] -= offset_gencive
        # On descend la cible de visée d'un poil pour bien voir la gencive sous la dent
        spc_target[:, :, hauteur_idx] -= (self.radius * 0.05) 
        
        R_at = spc_target.expand(-1, n_cameras, -1).reshape(-1, 3)
        R_pos = current_cam_pos.reshape(-1, 3)

        # 5. CONFIGURATION DE LA ROTATION
        R_up = up_vector.view(1, 3).expand(R_pos.shape[0], -1)
        R = look_at_rotation(R_pos, at=R_at, up=R_up, device=device) 
        T = -torch.bmm(R.transpose(1, 2), R_pos.unsqueeze(-1)).squeeze(-1)

        # 6. APPLICATION DU RENDU (Mise à jour du Zoom / FoV)
        batched_meshes = meshes.extend(n_cameras)
        renderer = self.renderer2 if rend else self.renderer
        
        # --- FORCE LE ZOOM ICI (IMPORTANT POUR LES PATCHS) ---
        # On va modifier temporairement les caméras du renderer pour "zoomer" (réduire le FoV)
        if hasattr(renderer, "cameras") and hasattr(renderer.cameras, "fov"):
            renderer.cameras.fov = 20.0 # Un FoV bas (ex: 15.0 ou 20.0) agit comme un zoom puissant sur la dent
            
        images = renderer(meshes_world=batched_meshes, R=R, T=T) 

        if rend:
            y = images[:, :, :, 0:3].permute(0, 3, 1, 2) 
            return y.view(batch_size, n_cameras, 3, images.shape[1], images.shape[2])
        else:
            rgb = images[:, :, :, :-1].permute(0, 3, 1, 2)
            zbuf = renderer.rasterizer(batched_meshes).zbuf.permute(0, 3, 1, 2)
            y = torch.cat([rgb, zbuf], dim=1) 
            return y.view(batch_size, n_cameras, 4, images.shape[1], images.shape[2])
    
    def get_view_rasterize(self, meshes, type_lm):
        """
        OPTIMIZED: Vectorized computation + batch rendering avec principes de GetView
        """
        batch_size = len(meshes)
        device = GV.DEVICE
        
        # Position du landmark de la dent (batch_size, 1, 3) ou (batch_size, 3)
        spc = self.positions.view(batch_size, 1, 3)
        
        # 1. CALCUL DU CENTRE DE LA MÂCHOIRE
        verts = meshes.verts_padded() 
        machoire_center = verts.mean(dim=1, keepdim=True) 

        # 2. VECTEUR DE DIRECTION INITIAL (Vers l'extérieur)
        direction_buccale = spc - machoire_center
        
        # L'axe vertical est Z (index 2), on l'écrase pour le calcul horizontal
        hauteur_idx = 2
        plane_idx = 1 # Axe Y
        
        direction_buccale[:, :, hauteur_idx] = 0 
        up_vector = torch.tensor([0.0, 0.0, 1.0], device=device) # Le haut est Z
        
        # Normalisation de la direction de base
        direction_buccale = direction_buccale / (torch.norm(direction_buccale, dim=-1, keepdim=True) + 1e-6)

        # 3. PRÉPARATION DES CIBLES ET POSITIONS (Pour chaque caméra dans self.camera_points)
        img_lst = []
        tens_pix_to_face = []
        
        # Forçage du FoV / Zoom si nécessaire comme dans GetView
        renderer = self.renderer
        if hasattr(renderer, "cameras") and hasattr(renderer.cameras, "fov"):
            renderer.cameras.fov = 20.0

        for sp in self.camera_points:
            # `sp` est un vecteur directionnel propre à cette caméra spécifique (ex: face, incliné, etc.)
            # On adapte la direction buccale de base avec les spécificités de `sp`
            # Si sp contient déjà des rotations précalculées, on applique l'offset et le rayon :
            sp_i = torch.tensor(sp, device=device).view(1, 1, 3) if not isinstance(sp, torch.Tensor) else sp.view(1, 1, 3)
            
            # Position de base de la caméra
            current_cam_pos = spc + (direction_buccale * self.radius)
            
            # --- APPLICATION DU COMPORTEMENT DE CÔTÉ (Légère plongée) ---
            # On descend un peu la caméra sur l'axe vertical pour créer l'effet plongeant de GetView
            current_cam_pos[:, :, hauteur_idx] -= (self.radius * 0.15)
            
            # --- CONFIGURATION DE LA CIBLE (TARGET) ---
            spc_target = spc.clone()
            offset_gencive = 0.2  # Descend vers la gencive
            
            spc_target[:, :, hauteur_idx] -= offset_gencive
            spc_target[:, :, hauteur_idx] -= (self.radius * 0.05) # Ajustement fin de GetView
            
            # Mise à plat pour look_at_rotation
            cam_pos_flat = current_cam_pos.view(-1, 3)
            target_flat = spc_target.view(-1, 3)
            R_up = up_vector.view(1, 3).expand(cam_pos_flat.shape[0], -1)
            
            # 4. CALCUL DES MATRICES DE VUE R & T
            R = look_at_rotation(cam_pos_flat, at=target_flat, up=R_up, device=device)
            T = -torch.bmm(R.transpose(1, 2), cam_pos_flat[:, :, None])[:, :, 0]
            
            # 5. RENDU ET RASTERIZATION
            images = renderer(meshes_world=meshes, R=R, T=T.to(device))
            images = images.permute(0, 3, 1, 2)
            rgb = images[:, :-1, :, :] # Retire le canal alpha
            
            fragments = renderer.rasterizer(meshes)
            pix_to_face = fragments.pix_to_face
            zbuf = fragments.zbuf.permute(0, 3, 1, 2)
            
            # Concaténation RGB + Z-Buffer (4 canaux)
            y = torch.cat([rgb, zbuf], dim=1)
            
            img_lst.append(y.unsqueeze(1))
            tens_pix_to_face.append(pix_to_face.unsqueeze(1))
        
        # Concaténation finale sur la dimension des caméras (dim=1)
        img_lst = torch.cat(img_lst, dim=1)
        tens_pix_to_face = torch.cat(tens_pix_to_face, dim=1)
        
        return img_lst, tens_pix_to_face


def PlotAgentViews(view):
    for batch in view:
        if batch.shape[0] > 5:
            row = int(math.ceil(batch.shape[0]/5)) 
            f, axarr = plt.subplots(nrows=row,ncols=5)
            c,r = 0,0
            for image in batch:
                image = image.permute(1,2,0)[:,:,:-1]
                axarr[r,c].imshow(image)
                c += 1
                if c == 5:c,r = 0,r+1
        else:
            f, axarr = plt.subplots(nrows=1,ncols=batch.shape[0])
            for i,image in enumerate(batch):
                image = image.permute(1,2,0)[:,:,:-1]
                axarr[i].imshow(image)
        plt.show()

