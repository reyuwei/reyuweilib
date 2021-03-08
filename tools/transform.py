import numpy as np
import torch
from scipy.spatial.transform import Rotation

def t2mat(t_vec):
    '''
    np.array
    return 4*4 transformation matrix from 3D translation
    '''
    I = np.eye(4)
    I[0:3, 3] = t_vec
    return I  # 4,4


def axisang2mat(r_axisangle):
    '''
    np.array
    return 4*4 transformation matrix from 3D axis angle rotation
    '''
    rot = Rotation.from_rotvec(r_axisangle)
    I = np.eye(4)
    I[0:3, 0:3] = rot.as_dcm()
    return I  # 4,4

def normalizev(vec):
    """
    Normalize vector (numpy or pytorch)

    Parameters
    ----------
    vec : np.array [N, C]

    Returns
    ----------
    vec: np.array
    """
    if isinstance(vec, np.ndarray):
        return vec / np.linalg.norm(vec, axis=-1)
    else:
        return vec / torch.norm(vec, dim=-1)

def transform_verts(vertices, transmat):
    """
    Apply transmat to vertices

    Parameters
    ----------
    vertices : (np.array, [N, 3]) 3D vertices 
    transmat : (np.array, [4, 4]) transformation matrix
               (np.array, [N, 4, 4]) transformation matrix for each vertex (use einsum)

    Returns
    ----------
    vertices : (np.array, [N, 3]) transformed 3D vertices
    """

    ones = np.ones([vertices.shape[0], 1])
    vertices_hom = np.concatenate([vertices, ones], axis=1)  # [N, 4]
    if len(transmat.shape) == 3 and transmat.shape[0] == vertices.shape[0]:
        vertices_hom = np.einsum("ni,nji->nj", vertices_hom, transmat)
    elif transmat.shape[0] == 4 and transmat.shape[1] == 4:
        vertices_hom = vertices_hom @ transmat.T
    else:
        assert "transmat shape: " +str(transmat.shape) + " is invalid!"
    
    vertices = vertices_hom[:, :3]
    return vertices