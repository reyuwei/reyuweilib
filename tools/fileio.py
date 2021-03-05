import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import torch
import os
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation
import trimesh
import pickle

def savepkl(arr, filename):
    with open(filename, 'wb') as f:
        pickle.dump(arr, f)


def loadpkl(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f, encoding='bytes')

def tensor2str(tens):
    arr = tens.cpu().numpy()
    s = ""
    for a in arr.flatten():
        s += str(a) + " "
    s += "\n"
    return s

def save_mesh(verts, savename, normals=None, faces=None, verts_color=None):
    if faces is None:
        if verts_color is None:
            verts_color = np.zeros([verts.shape[0], 4])
        verts_pts = trimesh.PointCloud(vertices=verts, colors=verts_color)
        verts_pts.export(savename)
        return verts_pts
    else:
        verts_mesh = trimesh.Trimesh(vertices=verts,
                                     faces=faces,
                                     normals=normals,
                                     process=False)
        verts_mesh.export(savename)
        return verts_mesh
