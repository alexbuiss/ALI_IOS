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
        OPTIMISÉE PHASE 2: Return positions directement pour pré-caching
        """
        final_pos = torch.empty((0)).to(GV.DEVICE)
        
        for mesh in range(len(text)):
            if int(label) in text[mesh]:
                index_pos_land = (text[mesh]==int(label)).nonzero(as_tuple=True)[0]
                if len(index_pos_land) > 0:
                    # Vectorized with indexing
                    lst_pos = vert[mesh][index_pos_land]
                    position_agent = lst_pos.mean(dim=0)
                else:
                    position_agent = torch.zeros(3, device=GV.DEVICE)
                final_pos = torch.cat((final_pos, position_agent.unsqueeze(0).to(GV.DEVICE)), dim=0)
            else:
                final_pos = torch.cat((final_pos, torch.zeros((1, 3), device=GV.DEVICE)), dim=0)
        
        self.positions = final_pos
        return final_pos

    
    def GetView(self, meshes,type_lm,rend=False):
        batch_size = len(meshes) 
        n_cameras = len(self.camera_points)
        device = GV.DEVICE

        spc = self.positions.view(-1, 1, 3)[:batch_size] 
        offset_z = 0.0#-0.2 if type_lm == "L" else 0.2
        # print(offset_z)
        target_offset = torch.tensor([0.0, 0.0, offset_z], device=device).view(1, 1, 3)
        spc_target = spc + target_offset
        
        cam_points_expanded = self.camera_points.unsqueeze(0).expand(batch_size, -1, -1)
        current_cam_pos = spc + (cam_points_expanded * self.radius)
        
        R_at = spc_target.expand(-1, n_cameras, -1).reshape(-1, 3)
        R_pos = current_cam_pos.reshape(-1, 3)

        R = look_at_rotation(R_pos, at=R_at, device=device) 
        T = -torch.bmm(R.transpose(1, 2), R_pos.unsqueeze(-1)).squeeze(-1)

        batched_meshes = meshes.extend(n_cameras)

        renderer = self.renderer2 if rend else self.renderer
        images = renderer(meshes_world=batched_meshes, R=R, T=T) 

        if rend:
            y = images[:, :, :, 0:3].permute(0, 3, 1, 2) 
            return y.view(batch_size, n_cameras, 3, images.shape[1], images.shape[2])
        else:
            rgb = images[:, :, :, :-1].permute(0, 3, 1, 2)
            zbuf = renderer.rasterizer(batched_meshes).zbuf.permute(0, 3, 1, 2)
            y = torch.cat([rgb, zbuf], dim=1) 
            return y.view(batch_size, n_cameras, 4, images.shape[1], images.shape[2])
    
    def get_view_rasterize(self, meshes,type_lm):
        """
        OPTIMIZED: Vectorized computation + batch rendering
        """
        spc = self.positions
        batch_size = spc.shape[0]
        device = GV.DEVICE
        offset_z = 0.0#-0.2 if type_lm == "L" else 0.2
        target_offset = torch.tensor([0.0, 0.0, offset_z], device=device)
        
        if len(spc.shape) == 2:
            spc_target = spc + target_offset.view(1, 3)
        else:
            spc_target = spc + target_offset.view(1, 1, 3)
        
        all_R = []
        all_T = []
        
        for sp in self.camera_points:
            sp_i = sp * self.radius
            current_cam_pos = spc + sp_i
            
            cam_pos_flat = current_cam_pos.view(-1, 3)
            target_flat = spc_target.view(-1, 3)
            
            R = look_at_rotation(cam_pos_flat, at=target_flat, device=device)
            all_R.append(R)
            
            T = -torch.bmm(R.transpose(1, 2), cam_pos_flat[:, :, None])[:, :, 0]
            all_T.append(T)
        
        img_lst = torch.empty((0)).to(device)
        tens_pix_to_face = torch.empty((0)).to(device)
        
        for cam_idx, (R, T) in enumerate(zip(all_R, all_T)):
            renderer = self.renderer
            images = renderer(meshes_world=meshes, R=R, T=T.to(device))
            images = images.permute(0, 3, 1, 2)
            images = images[:, :-1, :, :]
            
            fragments = renderer.rasterizer(meshes)
            pix_to_face = fragments.pix_to_face
            zbuf = fragments.zbuf
            zbuf = zbuf.permute(0, 3, 1, 2)
            
            y = torch.cat([images, zbuf], dim=1)
            img_lst = torch.cat((img_lst, y.unsqueeze(1)), dim=1)
            tens_pix_to_face = torch.cat((tens_pix_to_face, pix_to_face.unsqueeze(1)), dim=1)
        
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

