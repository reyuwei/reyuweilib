import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import trimesh
import json
import torch
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from skimage import measure
from scipy.interpolate import Rbf, RegularGridInterpolator

def get_grid(points, resolution):
    eps = 0.2
    input_min = torch.min(points, dim=0)[0].squeeze().numpy()
    input_max = torch.max(points, dim=0)[0].squeeze().numpy()

    bounding_box = input_max - input_min
    shortest_axis = np.argmin(bounding_box)
    if (shortest_axis == 0):
        x = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(x) - np.min(x)
        y = np.arange(input_min[1] - eps, input_max[1] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
    elif (shortest_axis == 1):
        y = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(y) - np.min(y)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
    elif (shortest_axis == 2):
        z = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(z) - np.min(z)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))
        y = np.arange(input_min[1] - eps, input_max[1] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float)
    return {"grid_points": grid_points,
            "shortest_axis_length": length,
            "xyz": [x, y, z],
            "shortest_axis_index": shortest_axis}


def normalize_intensity(img_tensor, normalization="full_volume_mean", norm_values=(0, 1, 1, 0)):
    """
    Accepts an image tensor and normalizes it
    :param normalization: choices = "max", "mean" , type=str
    """
    if normalization == "mean":
        mask = img_tensor.ne(0.0)
        desired = img_tensor[mask]
        mean_val, std_val = desired.mean(), desired.std()
        img_tensor = (img_tensor - mean_val) / std_val
    elif normalization == "max":
        max_val, _ = torch.max(img_tensor)
        img_tensor = img_tensor / max_val
    elif normalization == 'brats':
        # print(norm_values)
        normalized_tensor = (img_tensor.clone() - norm_values[0]) / norm_values[1]
        final_tensor = torch.where(img_tensor == 0., img_tensor, normalized_tensor)
        final_tensor = 100.0 * ((final_tensor.clone() - norm_values[3]) / (norm_values[2] - norm_values[3])) + 10.0
        x = torch.where(img_tensor == 0., img_tensor, final_tensor)
        return x

    elif normalization == 'full_volume_mean':
        img_tensor = (img_tensor.clone() - norm_values[0]) / norm_values[1]

    elif normalization == 'max_min':
        img_tensor = (img_tensor - norm_values[3]) / ((norm_values[2] - norm_values[3]))

    elif normalization == None:
        img_tensor = img_tensor
    return img_tensor

def percentile_clip(img_numpy, min_val=0.1, max_val=99.8):
    """
    Intensity normalization based on percentile
    Clips the range based on the quarile values.
    :param min_val: should be in the range [0,100]
    :param max_val: should be in the range [0,100]
    :return: intesity normalized image
    """
    low = np.percentile(img_numpy, min_val)
    high = np.percentile(img_numpy, max_val)

    img_numpy[img_numpy < low] = low
    img_numpy[img_numpy > high] = high
    return img_numpy


class MRI:
    def __init__(self, filename):
        self.vol = sitk.ReadImage(str(filename))
        self.vol_nda = sitk.GetArrayFromImage(self.vol).transpose(2,1,0)

    def naive_seg(self, radius=3):
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
        mask = otsu_filter.Execute(self.vol)

        vectorRadius=(radius,radius,radius)
        kernel=sitk.sitkBall
        fg_mask = sitk.BinaryMorphologicalClosing(mask,vectorRadius,kernel)
        return fg_mask

    def marching_cube(self, mask_nda=None):
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
        vol = self.vol
        if mask_nda is None:
            mask_nda = self.naive_seg()
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

    def intensity_normalization(self, islabel=False):
        img_np = self.vol_nda
        img_np = np.array(img_np, dtype=np.float32)
        img_tensor = torch.from_numpy(img_np)

        MEAN, STD, MAX, MIN = 0., 1., 1., 0.
        if islabel is False:
            MEAN, STD = img_tensor.mean(), img_tensor.std()
            MAX, MIN = img_tensor.max(), img_tensor.min()
            img_tensor = normalize_intensity(img_tensor, normalization=normalization, norm_values=(MEAN, STD, MAX, MIN))  
        return MRI.nda2nii(img_tensor.numpy(), self.vol.GetSpacing(), self.vol)

    def percentile_clip(self):
        img_np = self.vol_nda
        img_np = np.array(img_np, dtype=np.float32)
        img_np = percentile_clip(img_np)
        return MRI.nda2nii(img_np, self.vol.GetSpacing(), self.vol)

    def get_affine_matrix(self):
        img_nii = self.vol
        # affine = img_nii.affine
        t = np.array(img_nii.GetOrigin()).reshape(3, 1)
        r = np.array(img_nii.GetDirection()).reshape(3, 3)
        s = np.array(img_nii.GetSpacing()).reshape(-1)
        s_id = np.eye(3)
        s_id[0,0] = s[0]
        s_id[1,1] = s[1]
        s_id[2,2] = s[2]
        trans_mat = np.concatenate([r@s_id, t], axis=-1)
        trans_mat_4 = np.eye(4)
        trans_mat_4[:3, :4] = trans_mat
        affine = trans_mat_4
        return affine

    def resample_sitk(self, target_spacing, iter="linear"):
        vol = self.vol
        orig_spacing = self.vol.GetSpacing()
        resample = sitk.ResampleImageFilter()
        if iter=="nn":
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
        elif iter=="linear":
            resample.SetInterpolator(sitk.sitkLinear)
        else:
            assert "No such iter use [linear, nn]!"
        resample.SetOutputDirection(vol.GetDirection())
        resample.SetOutputOrigin(vol.GetOrigin())
        resample.SetOutputSpacing(target_spacing)

        orig_size = np.array(vol.GetSize(), dtype=np.int)
        new_size = orig_size * (orig_spacing / np.array(target_spacing))
        new_size = np.ceil(new_size).astype(np.int)  # Image dimensions are in integers
        new_size = [int(s) for s in new_size]
        resample.SetSize(new_size)
        newimage = resample.Execute(vol)

        return newimage

    def resample(self, spacing_new, interp="linear"):
        vol = self.vol
        spacing_ori = vol.GetSpacing()
        if len(np.unique(spacing_ori)) != 1 and len(np.unique(spacing_new)) != 1:  # not 1:1:1
            iso_spacing = [np.min(spacing_ori), np.min(spacing_ori), np.min(spacing_ori)]
            vol = mri_resample(vol, iso_spacing, interp="linear")
            spacing_ori = vol.GetSpacing()

        vol_nda = sitk.GetArrayFromImage(vol).transpose(2, 1, 0)
        DHW = np.array(vol.GetSize())
        DHW_new = (spacing_ori * DHW / spacing_new).astype(int)

        grid, index2phy_mat = MRI.create_physical_grid(DHW, spacing_ori, vol)
        volume_interp_fn = RegularGridInterpolator(grid, vol_nda, method=interp, bounds_error=False, fill_value=0)

        (x_, y_, z_), _ = MRI.create_physical_grid(DHW_new, spacing_new)
        px, py, pz = np.meshgrid(x_, y_, z_, indexing='ij')
        new_sample_pts = np.stack([px, py, pz], -1)  # [D, H, W, 3]

        new_sample_pts_nda = volume_interp_fn(new_sample_pts.reshape(-1, 3))
        new_sample_pts_nda = new_sample_pts_nda.reshape(DHW_new)

        vol_new = MRI.nda2nii(new_sample_pts_nda, spacing_new, vol)

        return vol_new
        
    def togrid(self, _with_rt=False):
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
        vol = self.vol
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

    def marching_cube_idr(self, mask_nda=None, resolution=128):
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
        vol = self.vol
        if mask_nda is None:
            mask_nda = self.naive_seg()
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
        
        # smooth  keep largest component
        components = mesh.split(only_watertight=False)
        areas = np.array([c.area for c in components], dtype=np.float)
        mesh_low_res = components[areas.argmax()]
        recon_pc = trimesh.sample.sample_surface(mesh_low_res, 10000)[0]
        recon_pc = torch.from_numpy(recon_pc).float()
        
        # Center and align the recon pc
        s_mean = recon_pc.mean(dim=0)
        s_cov = recon_pc - s_mean
        s_cov = torch.mm(s_cov.transpose(0, 1), s_cov)
        vecs = torch.eig(s_cov, True)[1].transpose(0, 1)
        if torch.det(vecs) < 0:
            vecs = torch.mm(torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]]).float(), vecs)
        helper = torch.bmm(vecs.unsqueeze(0).repeat(recon_pc.shape[0], 1, 1),
                           (recon_pc - s_mean).unsqueeze(-1)).squeeze()
        grid_aligned = get_grid(helper.cpu(), resolution)
        grid_points = grid_aligned['grid_points']

        g = []
        for i, pnts in enumerate(torch.split(grid_points, 100000, dim=0)):
            g.append(torch.bmm(vecs.unsqueeze(0).repeat(pnts.shape[0], 1, 1).transpose(1, 2),
                               pnts.unsqueeze(-1)).squeeze() + s_mean)
        grid_points = torch.cat(g, dim=0)
    
        grid, _ = MRI.create_physical_grid(self.vol.GetSize(), self.vol.GetSpacing())
        volume_interp_fn = RegularGridInterpolator(grid, mask_nda, method='linear', bounds_error=False, fill_value=0)

        points = grid_points
        z = []
        for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
            z.append(volume_interp_fn(pnts.reshape(-1, 3)).reshape(-1, 1))
        z = np.concatenate(z, axis=0)
        
        meshexport = None
        if (not (np.min(z) > 0 or np.max(z) < 0)):

            z = z.astype(np.float32)

            verts, faces, normals, values = measure.marching_cubes_lewiner(
                volume=z.reshape(grid_aligned['xyz'][1].shape[0], grid_aligned['xyz'][0].shape[0],
                                 grid_aligned['xyz'][2].shape[0]).transpose([1, 0, 2]),
                level=0.5,
                spacing=(grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                         grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                         grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1]))

            verts = torch.from_numpy(verts).float()
            verts = torch.bmm(vecs.unsqueeze(0).repeat(verts.shape[0], 1, 1).transpose(1, 2),
                       verts.unsqueeze(-1)).squeeze()
            verts = (verts + grid_points[0]).cpu().numpy()

            meshexport = trimesh.Trimesh(verts, faces, normals)
            meshexport.fix_normals()
            
            # largest component
            components = meshexport.split(only_watertight=False)
            areas = np.array([c.area for c in components], dtype=np.float)
            meshexport = components[areas.argmax()]
            
            # smooth
            trimesh.smoothing.filter_humphrey(meshexport, alpha=0.1, beta=0.7, iterations=30)
        
        t = np.array(vol.GetOrigin()).reshape(3, 1)
        vd = np.array(vol.GetDirection()).reshape(3, 3)
        trans_mat = np.eye(4)
        trans_mat[:3, :3] = vd
        trans_mat[:3, -1:] = t
        meshexport.apply_transform(trans_mat)
        mesh_low_res.apply_transform(trans_mat)

        return meshexport

    @staticmethod
    def nda2nii(vol_nda, spacing_new, ori_vol):
        vol_nda = vol_nda.squeeze()
        mri_nii = sitk.GetImageFromArray(vol_nda.transpose(2, 1, 0))
        mri_nii.SetSpacing(spacing_new)
        mri_nii.SetOrigin(ori_vol.GetOrigin())
        mri_nii.SetDirection(ori_vol.GetDirection())
        return mri_nii

    @staticmethod
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

    @staticmethod
    def histgram_matching(reference, vol):
        matcher = sitk.HistogramMatchingImageFilter()
        if (reference.GetPixelID() in (sitk.sitkUInt8, sitk.sitkInt8)):
            matcher.SetNumberOfHistogramLevels(128)
        else:
            matcher.SetNumberOfHistogramLevels(1024)
        matcher.SetNumberOfMatchPoints(7)
        matcher.ThresholdAtMeanIntensityOn()
        return matcher.Execute(vol, reference)


