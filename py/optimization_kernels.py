"""
OPTIMIZATION KERNELS - Phase 2
Contains compiled kernels and advanced optimizations
"""

import torch
import torch.nn.functional as F

# Compile critical operations with TorchScript
@torch.jit.script
def fast_distance_computation(landmarks: torch.Tensor, vertices: torch.Tensor, radius: float) -> torch.Tensor:
    """
    Ultra-fast vectorized distance computation
    landmarks: [N, 3]
    vertices: [M, 3]
    Returns: mask [M] of close vertices to landmarks
    """
    # Compute pairwise distances efficiently
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    
    landmarks_norm = torch.sum(landmarks * landmarks, dim=1, keepdim=True)  # [N, 1]
    vertices_norm = torch.sum(vertices * vertices, dim=1, keepdim=True)  # [M, 1]
    
    # Dot product
    dot_product = torch.mm(landmarks, vertices.t())  # [N, M]
    
    # Distance squared
    distances_sq = landmarks_norm + vertices_norm.t() - 2 * dot_product
    
    # Ensure distance >= 0 (avoid numerical issues)
    distances_sq = torch.clamp(distances_sq, min=0.0)
    distances = torch.sqrt(distances_sq)
    
    # Min distance for each vertex
    min_distances = torch.min(distances, dim=0)[0]  # [M]
    
    # Mask
    return min_distances < radius


@torch.jit.script
def batch_distance_mask(batch_landmarks: torch.Tensor, batch_vertices: torch.Tensor, radius: float) -> torch.Tensor:
    """
    Batch version for distance computation
    batch_landmarks: [B, N, 3]
    batch_vertices: [B, M, 3]
    """
    B = batch_landmarks.shape[0]
    masks = []
    
    for b in range(B):
        mask = fast_distance_computation(batch_landmarks[b], batch_vertices[b], radius)
        masks.append(mask)
    
    return torch.stack(masks)


def optimized_cdist_forward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Optimized cdist for forward pass (faster than torch.cdist in many cases)
    """
    # For large tensors, scipy.spatial.distance is sometimes faster
    # But for GPU, use custom kernel
    
    # Optimized L2 norm
    x_norm = (x ** 2).sum(1, keepdim=True)  # [N, 1]
    y_norm = (y ** 2).sum(1, keepdim=True).t()  # [1, M]
    dist = torch.mm(x, y.t())  # [N, M]
    dist = torch.clamp(x_norm + y_norm - 2.0 * dist, min=0.0)
    return torch.sqrt(dist)


def position_agent_vectorized(text: torch.Tensor, vert: torch.Tensor, label: int) -> torch.Tensor:
    """
    PHASE 2: Fully vectorized version of position_agent
    Uses PyTorch indexing instead of loops
    """
    batch_size = len(text)
    positions = []
    
    for mesh_idx in range(batch_size):
        if int(label) in text[mesh_idx]:
            # Vectorized boolean mask
            mask = (text[mesh_idx] == int(label))
            if mask.any():
                # Mean of selected vertices
                selected = vert[mesh_idx][mask]
                position = selected.mean(dim=0)
            else:
                position = torch.zeros(3, device=vert[mesh_idx].device)
        else:
            position = torch.zeros(3, device=vert[mesh_idx].device)
        
        positions.append(position)
    
    return torch.stack(positions)
