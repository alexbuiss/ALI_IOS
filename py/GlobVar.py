import torch
import os
import numpy as np
from scipy import linalg

global DEVICE 
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mapping = {
            'LL7': 18, 'LL6': 19, 'LL5': 20, 'LL4': 21, 'LL3': 22, 'LL2': 23, 'LL1': 24,
            'LR1': 25, 'LR2': 26, 'LR3': 27, 'LR4': 28, 'LR5': 29, 'LR6': 30, 'LR7': 31,
            'UL7': 15, 'UL6': 14, 'UL5': 13, 'UL4': 12, 'UL3': 11, 'UL2': 10, 'UL1': 9,
            'UR1': 8, 'UR2': 7, 'UR3': 6, 'UR4': 5, 'UR5': 4, 'UR6': 3, 'UR7': 2
        }

LOWER_DENTAL = ['LL7','LL6','LL5','LL4','LL3','LL2','LL1','LR1','LR2','LR3','LR4','LR5','LR6','LR7']
UPPER_DENTAL = ['UL7','UL6','UL5','UL4','UL3','UL2','UL1','UR1','UR2','UR3','UR4','UR5','UR6','UR7']
TYPE_LM = ['O','MB','DB','CL','CB']

LANDMARKS = {
    "L": [tooth + lm for tooth in LOWER_DENTAL for lm in TYPE_LM],
    "U": [tooth + lm for tooth in UPPER_DENTAL for lm in TYPE_LM]
}

LABEL_L = [str(x) for x in range(18, 32)]  # "18" to "31"
LABEL_U = [str(x) for x in range(2, 16)]   # "2" to "15"

dic_label = {
    'O': {
        **{str(15 - i): LANDMARKS["U"][i*5:i*5+3] for i in range(14)},  # teeth 15 to 2
        **{str(18 + i): LANDMARKS["L"][i*5:i*5+3] for i in range(14)}   # teeth 18 to 31
    },
    'C': {
        **{str(15 - i): LANDMARKS["U"][i*5+3:i*5+5] for i in range(14)},
        **{str(18 + i): LANDMARKS["L"][i*5+3:i*5+5] for i in range(14)}
    }
}

MODELS_DICT = {
    'O': {'O': 0, 'MB': 1, 'DB': 2},
    'C': {'CL': 0, 'CB': 1}
}
import numpy as np

def generate_elliptical_360(n_cameras=12, height_angle=0.4):
    """
    Génère N caméras réparties sur une trajectoire elliptique pour compenser
    le fait que la dent soit plus large sur l'axe X que sur l'axe Y.
    """
    cam_points = []
    angles = np.linspace(0, 2 * np.pi, n_cameras, endpoint=False)
    
    # Coeffs basés sur tes Verts bounds (X est ~1.25 fois plus grand que Y)
    scale_x = 1.25
    scale_y = 1.00
    
    for angle in angles:
        # On déforme le cercle en ellipse
        x = np.cos(angle) * scale_x
        y = np.sin(angle) * scale_y
        z = height_angle
        
        cam_points.append([x, y, z])
        
    cam_points = np.array(cam_points)
    # On normalise ensuite pour garder des vecteurs unitaires propres
    return cam_points / np.linalg.norm(cam_points, axis=1, keepdims=True)

global dic_cam

# Camera positions for different landmark types and jaws
dic_cam = {
    'O': {
        'L': np.array([
            [0, 0, 1],
            np.array([0.5, 0., 1.0]) / linalg.norm([0.5, 0.5, 1.0]),
            np.array([-0.5, 0., 1.0]) / linalg.norm([-0.5, -0.5, 1.0]),
            np.array([0, 0.5, 1]) / linalg.norm([1, 0, 1]),
            np.array([0, -0.5, 1]) / linalg.norm([0, 1, 1])
        ]),
        'U': np.array([
            [0, 0, -1],
            np.array([0.5, 0., -1]) / linalg.norm([0.5, 0.5, -1]),
            np.array([-0.5, 0., -1]) / linalg.norm([-0.5, -0.5, -1]),
            np.array([0, 0.5, -1]) / linalg.norm([1, 0, -1]),
            np.array([0, -0.5, -1]) / linalg.norm([0, 1, -1])
        ])
    },
    'C': {
        'U': generate_elliptical_360(n_cameras=12, height_angle=0.4),  # Pour Upper
        'L': generate_elliptical_360(n_cameras=12, height_angle=-0.4) # Pour Lower
    }
    # 'C': {
    #     'L': np.array([
    #         np.array([1, 0, 0]) / linalg.norm([1, 0, 0]),
    #         np.array([-1, 0, 0]) / linalg.norm([-1, 0, 0]),
    #         np.array([1, -1, 0]) / linalg.norm([1, -1, 0]),
    #         np.array([-1, -1, 0]) / linalg.norm([-1, -1, 0]),
    #         np.array([1, 1, 0]) / linalg.norm([1, 1, 0]),
    #         np.array([-1, 1, 0]) / linalg.norm([-1, 1, 0]),
    #         np.array([1, 0, 0.5]) / linalg.norm([1, 0, 0.5]),
    #         np.array([-1, 0, 0.5]) / linalg.norm([-1, 0, 0.5]),
    #         np.array([1, -1, 0.5]) / linalg.norm([1, -1, 0.5]),
    #         np.array([-1, -1, 0.5]) / linalg.norm([-1, -1, 0.5]),
    #         np.array([1, 1, 0.5]) / linalg.norm([1, 1, 0.5]),
    #         np.array([-1, 1, 0.5]) / linalg.norm([-1, 1, 0.5])
    #     ]),
    #     'U': np.array([
    #         np.array([1, 0, 0]) / linalg.norm([1, 0, 0]),
    #         np.array([-1, 0, 0]) / linalg.norm([-1, 0, 0]),
    #         np.array([1, -1, 0]) / linalg.norm([1, -1, 0]),
    #         np.array([-1, -1, 0]) / linalg.norm([-1, -1, 0]),
    #         np.array([1, 1, 0]) / linalg.norm([1, 1, 0]),
    #         np.array([-1, 1, 0]) / linalg.norm([-1, 1, 0]),
    #         np.array([1, 0, -0.5]) / linalg.norm([1, 0, -0.5]),
    #         np.array([-1, 0, -0.5]) / linalg.norm([-1, 0, -0.5]),
    #         np.array([1, -1, -0.5]) / linalg.norm([1, -1, -0.5]),
    #         np.array([-1, -1, -0.5]) / linalg.norm([-1, -1, -0.5]),
    #         np.array([1, 1, -0.5]) / linalg.norm([1, 1, -0.5]),
    #         np.array([-1, 1, -0.5]) / linalg.norm([-1, 1, -0.5])
    #     ])
    # }
#     'C': {
#         'L': np.array([
#             # Niveau z = 0 (8 positions)
#             np.array([1, 0, 0]) / linalg.norm([1, 0, 0]),
#             np.array([-1, 0, 0]) / linalg.norm([-1, 0, 0]),
#             np.array([0, 1, 0]) / linalg.norm([0, 1, 0]),
#             np.array([0, -1, 0]) / linalg.norm([0, -1, 0]),
#             np.array([1, -1, 0]) / linalg.norm([1, -1, 0]),
#             np.array([-1, -1, 0]) / linalg.norm([-1, -1, 0]),
#             np.array([1, 1, 0]) / linalg.norm([1, 1, 0]),
#             np.array([-1, 1, 0]) / linalg.norm([-1, 1, 0]),
#             # Niveau z = 0.5 (8 positions)
#             np.array([1, 0, 0.5]) / linalg.norm([1, 0, 0.5]),
#             np.array([-1, 0, 0.5]) / linalg.norm([-1, 0, 0.5]),
#             np.array([0, 1, 0.5]) / linalg.norm([0, 1, 0.5]),
#             np.array([0, -1, 0.5]) / linalg.norm([0, -1, 0.5]),
#             np.array([1, -1, 0.5]) / linalg.norm([1, -1, 0.5]),
#             np.array([-1, -1, 0.5]) / linalg.norm([-1, -1, 0.5]),
#             np.array([1, 1, 0.5]) / linalg.norm([1, 1, 0.5]),
#             np.array([-1, 1, 0.5]) / linalg.norm([-1, 1, 0.5])
#         ]),
#         'U': np.array([
#             # Niveau z = 0 (8 positions)
#             np.array([1, 0, 0]) / linalg.norm([1, 0, 0]),
#             np.array([-1, 0, 0]) / linalg.norm([-1, 0, 0]),
#             np.array([0, 1, 0]) / linalg.norm([0, 1, 0]),
#             np.array([0, -1, 0]) / linalg.norm([0, -1, 0]),
#             np.array([1, -1, 0]) / linalg.norm([1, -1, 0]),
#             np.array([-1, -1, 0]) / linalg.norm([-1, -1, 0]),
#             np.array([1, 1, 0]) / linalg.norm([1, 1, 0]),
#             np.array([-1, 1, 0]) / linalg.norm([-1, 1, 0]),
#             # Niveau z = -0.5 (8 positions)
#             np.array([1, 0, -0.5]) / linalg.norm([1, 0, -0.5]),
#             np.array([-1, 0, -0.5]) / linalg.norm([-1, 0, -0.5]),
#             np.array([0, 1, -0.5]) / linalg.norm([0, 1, -0.5]),
#             np.array([0, -1, -0.5]) / linalg.norm([0, -1, -0.5]),
#             np.array([1, -1, -0.5]) / linalg.norm([1, -1, -0.5]),
#             np.array([-1, -1, -0.5]) / linalg.norm([-1, -1, -0.5]),
#             np.array([1, 1, -0.5]) / linalg.norm([1, 1, -0.5]),
#             np.array([-1, 1, -0.5]) / linalg.norm([-1, 1, -0.5])
#         ])
#     }
}




