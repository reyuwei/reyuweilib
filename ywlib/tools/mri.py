import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import trimesh
import json
import torch
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from skimage import measure
from umeyama import umeyama
from scipy.interpolate import Rbf, RegularGridInterpolator



def histgram_matching(reference, vol):
    matcher = sitk.HistogramMatchingImageFilter()
    if (reference.GetPixelID() in (sitk.sitkUInt8, sitk.sitkInt8)):
        matcher.SetNumberOfHistogramLevels(128)
    else:
        matcher.SetNumberOfHistogramLevels(1024)
    matcher.SetNumberOfMatchPoints(7)
    matcher.ThresholdAtMeanIntensityOn()
    return matcher.Execute(vol, reference)

def naive_seg(vol):
    """
    Apply segmentation to volume with OtsuThreshold with opening

    Parameters
    ----------
    vol : SimpleITK.Image 

    Returns
    ----------
    mask : SimpltITK.Image 
    threslod : int 
    """
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(0)
    otsu_filter.SetOutsideValue(1)
    mask = otsu_filter.Execute(vol)

    vectorRadius=(50,50,50)
    kernel=sitk.sitkBall
    fg_mask = sitk.BinaryMorphologicalClosing(mask,vectorRadius,kernel)
    return fg_mask

def nda2nii(vol_nda, spacing_new, ori_vol):
    vol_nda = vol_nda.squeeze()
    mri_nii = sitk.GetImageFromArray(vol_nda.transpose(2, 1, 0))
    mri_nii.SetSpacing(spacing_new)
    mri_nii.SetOrigin(ori_vol.GetOrigin())
    mri_nii.SetDirection(ori_vol.GetDirection())
    return mri_nii

def rbf_weights(volume, control_pts, control_pts_weights=None):
    """
    Compute 3D volume weight according to control points with rbf and "thin_plate" as kernel
    if control_pts_weights is None, directly return control points influences

    Parameters
    ----------
    volume : (np.array, [N,3]) volume points with unknown weights 
    control_pts : (np.array, [M,3]) control points
    control_pts_weights : (np.array, [M,K]) control point weights

    Returns
    ----------
    volume_weights : np.array, [N,K]
    """

    if control_pts_weights is None:
        control_pts_weights = np.eye(control_pts.shape[0])

    xyz = volume.reshape(-1, 3)
    chunk = 50000
    rbfi = Rbf(control_pts[:, 0], control_pts[:, 1], control_pts[:, 2], control_pts_weights, function="thin_plate", mode="N-D")
    # rbfi = Rbf(pts[:, 0], pts[:, 1], pts[:, 2], weight, function="multiquadric", mode="N-D")
    weight_volume = np.concatenate([rbfi(xyz[j:j + chunk, 0], xyz[j:j + chunk, 1], xyz[j:j + chunk, 2]) for j in
                                    range(0, xyz.shape[0], chunk)], 0)
    weight_volume[weight_volume < 0] = 0
    weight_volume = weight_volume / np.sum(weight_volume, axis=1).reshape(-1, 1)
    weight_volume = weight_volume.reshape(xyz.shape[0], -1)
    return weight_volume

def mesh2grid(mesh_verts, slices=None, spacing=[0.5, 0.5, 0.5]):
    """
    Create an AABB grid from the mesh_verts. Use slices or spacing to control sampling density.
    !Slices has higher priority over spacing!

    Parameters
    ----------
    mesh_verts : (np.array, [N,3]) 3D vertices
    slices : (int) How many slices per edge
    spacing : (np.array, [3,]) sampling spacing

    Returns
    ----------
    grid vertices : (np.array, [D, H, W, 3])
    meta: spacing/direction/origin of this grid, for creating SimpleITK.Image
    """

    pt = mesh_verts.squeeze()
    xyz = np.array(pt)
    max_xyz = np.max(xyz, axis=0) + spacing[0] * 2
    min_xyz = np.min(xyz, axis=0) - spacing[0] * 2
    D, H, W = max_xyz
    SD, SH, SW = min_xyz
    if slices is not None:
        ds = (D + 1e-5 - SD) / slices
        hs = (H + 1e-5 - SH) / slices
        ws = (W + 1e-5 - SW) / slices
    else:
        ds, hs, ws = spacing
    x_ = np.arange(SD, D + 1e-5, step=ds, dtype=np.float32)
    y_ = np.arange(SH, H + 1e-5, step=hs, dtype=np.float32)
    z_ = np.arange(SW, W + 1e-5, step=ws, dtype=np.float32)
    px, py, pz = np.meshgrid(x_, y_, z_, indexing='ij')
    all_pts = np.stack([px, py, pz], -1)  # [slices, slices, slices, 3]

    meta = {
        "spacing": (ds, hs, ws),
        "direction": np.eye(3),
        "origin": all_pts[0,0,0],
    }

    return all_pts, meta

def mri2grid(vol, _with_rt=False):
    """
    Create a 3D grid from the mri volume.
    if _with_rt: return grid with volume origin and direction; else return grid with only spacing, start at [0,0,0]

    Parameters
    ----------
    vol : SimpleITK.Image 
    _with_rt : bool

    Returns
    ----------
    grid : (np.array, [D, H, W, 3])
    """
    D, H, W = vol.GetSize()
    ds, hs, ws = vol.GetSpacing()
    x_ = np.arange(0, (D - 1) * ds + 1e-5, step=ds, dtype=np.float32)
    y_ = np.arange(0, (H - 1) * hs + 1e-5, step=hs, dtype=np.float32)
    z_ = np.arange(0, (W - 1) * ws + 1e-5, step=ws, dtype=np.float32)
    assert len(x_) == D and len(y_) == H and len(z_) == W
    mri_grid = (x_, y_, z_)

    if _with_rt:
        px, py, pz = np.meshgrid(x_, y_, z_, indexing='ij')
        physical_pts = np.stack([px, py, pz], -1)  # [D, H, W, 3]

        t = np.array(vol.GetOrigin()).reshape(3, 1)
        r = np.array(vol.GetDirection()).reshape(3, 3)
        trans_mat = np.concatenate([r, t], axis=-1)
        verts = physical_pts.reshape(-1, 3)
        ones = np.ones([verts.shape[0], 1]).astype(np.float32)
        verts = np.hstack([verts, ones])
        verts = verts @ trans_mat.T
        physical_pts = verts[:, :3].reshape(physical_pts.shape)

        physical_x_ = physical_pts[:,0,0,0]
        physical_y_ = physical_pts[0,:,0,1]
        physical_z_ = physical_pts[0,0,:,2]
        physical_mesh_grid = (physical_x_, physical_y_, physical_z_)
        return physical_mesh_grid
    else:
        return mri_grid

def mri_marching_cube(vol, mask_nda=None):
    """
    Create mesh from mri volume.

    Parameters
    ----------
    vol : SimpleITK.Image 
    mask : np.array bool

    Returns
    ----------
    mesh : trimesh.mesh
    """
    
    if mask_nda is None:
        mask_nda = naive_seg(vol)
        mask_nda = sitk.GetArrayFromImage(mask_nda).transpose(2,1,0)
    else:
        vol_nda = sitk.GetArrayFromImage(vol).transpose(2,1,0)
        assert mask_nda.shape == vol_nda.shape
        mask_nda = mask_nda

    spacing = np.array(vol.GetSpacing())
    verts, faces, normals, values = measure.marching_cubes(mask_nda, spacing=spacing)
    faces = np.flip(faces, axis=-1)
    mesh = trimesh.Trimesh(vertices=verts,
                           faces=faces,
                           normals=normals,
                           process=True,
                           validate=True)
    t = np.array(vol.GetOrigin()).reshape(3, 1)
    vd = np.array(vol.GetDirection()).reshape(3, 3)
    trans_mat = np.eye(4)
    trans_mat[:3, :3] = vd
    trans_mat[:3, -1:] = t
    mesh.apply_transform(trans_mat)
    return mesh

def create_physical_grid(DHW, spacing, vol=None):
    D, H, W = DHW
    ds, hs, ws = spacing
    x_ = np.arange(0, (D - 1) * ds + 1e-5, step=ds, dtype=np.float32)
    y_ = np.arange(0, (H - 1) * hs + 1e-5, step=hs, dtype=np.float32)
    z_ = np.arange(0, (W - 1) * ws + 1e-5, step=ws, dtype=np.float32)
    assert len(x_) == D and len(y_) == H and len(z_) == W

    trans_mat = np.eye(4)
    if vol:
        t = np.array(vol.GetOrigin()).reshape(3, 1)
        r = np.array(vol.GetDirection()).reshape(3, 3)
        trans_mat[:3, :3] = r
        trans_mat[:3, -1:] = t

    return (x_, y_, z_), trans_mat

def mri_resample(vol, spacing_new, interp="linear"):
    spacing_ori = vol.GetSpacing()
    if len(np.unique(spacing_ori)) != 1 and len(np.unique(spacing_new)) != 1:  # not 1:1:1
        iso_spacing = [np.min(spacing_ori), np.min(spacing_ori), np.min(spacing_ori)]
        vol = mri_resample(vol, iso_spacing, interp="linear")
        spacing_ori = vol.GetSpacing()

    vol_nda = sitk.GetArrayFromImage(vol).transpose(2, 1, 0)
    DHW = np.array(vol.GetSize())
    DHW_new = (spacing_ori * DHW / spacing_new).astype(int)

    grid, index2phy_mat = create_physical_grid(DHW, spacing_ori, vol)
    volume_interp_fn = RegularGridInterpolator(grid, vol_nda, method=interp, bounds_error=False, fill_value=0)

    (x_, y_, z_), _ = create_physical_grid(DHW_new, spacing_new)
    px, py, pz = np.meshgrid(x_, y_, z_, indexing='ij')
    new_sample_pts = np.stack([px, py, pz], -1)  # [D, H, W, 3]

    new_sample_pts_nda = volume_interp_fn(new_sample_pts.reshape(-1, 3))
    new_sample_pts_nda = new_sample_pts_nda.reshape(DHW_new)

    vol_new = nda2nii(new_sample_pts_nda, spacing_new, vol)

    return vol_new