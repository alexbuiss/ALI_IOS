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
TYPE_LM = ['O','MB','DB','CL','CB','MG']

LANDMARKS = {
    "L": [tooth + lm for tooth in LOWER_DENTAL for lm in TYPE_LM],
    "U": [tooth + lm for tooth in UPPER_DENTAL for lm in TYPE_LM]
}

LABEL_L = [str(x) for x in range(18, 32)]  # "18" to "31"
LABEL_U = [str(x) for x in range(2, 16)]   # "2" to "15"

dic_label = {
    'O': {
        **{str(15 - i): LANDMARKS["U"][i*6:i*6+3] for i in range(14)},  # teeth 15 to 2
        **{str(18 + i): LANDMARKS["L"][i*6:i*6+3] for i in range(14)}   # teeth 18 to 31
    },
    'C': {
        **{str(15 - i): LANDMARKS["U"][i*6+3:i*6+5] for i in range(14)},
        **{str(18 + i): LANDMARKS["L"][i*6+3:i*6+5] for i in range(14)}
    },
    'MG': {
        **{str(18 + i): [LANDMARKS["L"][i*6+5]] for i in range(14)}
    }
}

MODELS_DICT = {
    'O': {'O': 0, 'MB': 1, 'DB': 2},
    'C': {'CL': 0, 'CB': 1},
    'MG':{'MG':0}
}

PATH_DICT = {
    'O': "Occlusal",
    'C': "Cervical",
    'MG':"Mucogingival"
}

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
        'L': np.array([
            np.array([1, 0, 0]) / linalg.norm([1, 0, 0]),
            np.array([-1, 0, 0]) / linalg.norm([-1, 0, 0]),
            np.array([1, -1, 0]) / linalg.norm([1, -1, 0]),
            np.array([-1, -1, 0]) / linalg.norm([-1, -1, 0]),
            np.array([1, 1, 0]) / linalg.norm([1, 1, 0]),
            np.array([-1, 1, 0]) / linalg.norm([-1, 1, 0]),
            np.array([1, 0, 0.5]) / linalg.norm([1, 0, 0.5]),
            np.array([-1, 0, 0.5]) / linalg.norm([-1, 0, 0.5]),
            np.array([1, -1, 0.5]) / linalg.norm([1, -1, 0.5]),
            np.array([-1, -1, 0.5]) / linalg.norm([-1, -1, 0.5]),
            np.array([1, 1, 0.5]) / linalg.norm([1, 1, 0.5]),
            np.array([-1, 1, 0.5]) / linalg.norm([-1, 1, 0.5])
        ]),
        'U': np.array([
            np.array([1, 0, 0]) / linalg.norm([1, 0, 0]),
            np.array([-1, 0, 0]) / linalg.norm([-1, 0, 0]),
            np.array([1, -1, 0]) / linalg.norm([1, -1, 0]),
            np.array([-1, -1, 0]) / linalg.norm([-1, -1, 0]),
            np.array([1, 1, 0]) / linalg.norm([1, 1, 0]),
            np.array([-1, 1, 0]) / linalg.norm([-1, 1, 0]),
            np.array([1, 0, -0.5]) / linalg.norm([1, 0, -0.5]),
            np.array([-1, 0, -0.5]) / linalg.norm([-1, 0, -0.5]),
            np.array([1, -1, -0.5]) / linalg.norm([1, -1, -0.5]),
            np.array([-1, -1, -0.5]) / linalg.norm([-1, -1, -0.5]),
            np.array([1, 1, -0.5]) / linalg.norm([1, 1, -0.5]),
            np.array([-1, 1, -0.5]) / linalg.norm([-1, 1, -0.5])
        ])
    },
    'MG': {
        'L': np.array([
            np.array([1, 0, 0]) / linalg.norm([1, 0, 0]),
            np.array([-1, 0, 0]) / linalg.norm([-1, 0, 0]),
            np.array([1, -1, 0]) / linalg.norm([1, -1, 0]),
        ]),
        'U': np.array([
            np.array([1, 0, 0]) / linalg.norm([1, 0, 0]),
            np.array([-1, 0, 0]) / linalg.norm([-1, 0, 0]),
            np.array([1, -1, 0]) / linalg.norm([1, -1, 0]),
            np.array([-1, -1, 0]) / linalg.norm([-1, -1, 0]),
            np.array([1, 1, 0]) / linalg.norm([1, 1, 0]),
            np.array([-1, 1, 0]) / linalg.norm([-1, 1, 0]),
            np.array([1, 0, -0.5]) / linalg.norm([1, 0, -0.5]),
            np.array([-1, 0, -0.5]) / linalg.norm([-1, 0, -0.5]),
            np.array([1, -1, -0.5]) / linalg.norm([1, -1, -0.5]),
            np.array([-1, -1, -0.5]) / linalg.norm([-1, -1, -0.5]),
            np.array([1, 1, -0.5]) / linalg.norm([1, 1, -0.5]),
            np.array([-1, 1, -0.5]) / linalg.norm([-1, 1, -0.5])
        ])
    }
}




