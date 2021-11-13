import os
from matplotlib.pyplot import hist
import trimesh
import torch
import numpy as np

from pytorch3d.ops.knn import knn_points
import colorsys
from hausdorff import hausdorff_distance

def hausdorff_distance_vismesh(ma:trimesh.Trimesh, mb:trimesh.Trimesh, min=0, max=10*1e-3):
    '''
    ma: trimesh
    mb: trimesh
    min: for visualization, clip cham_ab, color blue
    max: for visualization, clip cham_ab, color blue

    return: 
    bidirectional hausdorff distance, 
    colored mesh ma, 
    chamfer distance from ma to mb, 
    chamfer distance from mb to ma
    '''
    va = torch.from_numpy(ma.vertices).float().unsqueeze(0)
    vb = torch.from_numpy(mb.vertices).float().unsqueeze(0)
    x_nn = knn_points(va, vb, K=1)
    cham_x = x_nn.dists[..., 0].squeeze()  # (N, P1)
    cham_x = cham_x**0.5

    y_nn = knn_points(vb, va, K=1)
    cham_y = y_nn.dists[..., 0].squeeze()  # (N, P1)
    cham_y = cham_y**0.5
    
    hd_xy = cham_x.max()
    hd_yx = cham_y.max()
    hd = torch.max(hd_xy, hd_yx)

    # color ma by cham_x
    cham_x_clip = cham_x.clone()
    cham_x_clip[cham_x_clip>max] = max
    cham_x_clip[cham_x_clip<min] = min

    ratio = (cham_x_clip - min) / (max-min)
    max_h = 0 #red
    min_h = 0.667 #blue
    color_h = min_h - ratio * min_h

    rgb = np.stack([colorsys.hsv_to_rgb(i, 1, 1) for i in color_h])
    a = np.ones([rgb.shape[0], 1])
    rgba = np.hstack([rgb, a])
    rgba *= 255.

    ma.visual.vertex_colors = rgba
    return hd, ma, cham_x.numpy(), cham_y.numpy()

def histogram_vis(data: np.ndarray, bins_c=50, lab='label'):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5, 3))
    counts, bins = np.histogram(data, bins=bins_c)
    plt.hist(bins[:-1], bins_c, weights=counts/data.shape[0], label=lab, histtype='bar')
    print(bins)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    rootp = r"F:\OneDrive\Projects_ongoing\13_HANDMRI_Muscle\3_CODE_learn_muscle_skinning\debug_folder\compare_mano"
    a = trimesh.load(os.path.join(rootp, "37.obj"), process=False)
    b = trimesh.load(os.path.join(rootp, "1636647278_306936\\3\\mano_test_37.obj"), process=False)
    c = trimesh.load(os.path.join(rootp, "piano\\1636647559_4610462\\3\\skin_0.obj"), process=False)



    d1 = hausdorff_distance(b.vertices, a.vertices, distance='euclidean')
    d2 = hausdorff_distance(c.vertices, a.vertices, distance='euclidean')
    print(d1, d2)


    d1, cb, dba, dab= hausdorff_distance_vismesh(b, a)
    d2, cc, dca, dac= hausdorff_distance_vismesh(c, a)
    print(d1, d2)

    histogram_vis(dba)
    histogram_vis(dab)
    histogram_vis(dca)
    histogram_vis(dac)
