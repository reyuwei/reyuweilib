import numpy as np
import trimesh
import colorsys


def visual_weights(bone_obj, bone_pts_weights):

    '''
    visulize_weight 
    
    Parameters:
    bone_obj: trimesh with N vertices / np.array [N, 3]
    bone_pts_weights: np.array [N,k]

    Return:
    colored bone obj (trimesh.mesh or trimesh.pointcloud)
    '''

    label_list = np.argmax(bone_pts_weights, axis=1)
    bone_num = bone_pts_weights.shape[-1]
    if isinstance(bone_obj, np.array):
        rgb = np.stack([colorsys.hsv_to_rgb(i * 1.0 / bone_num, 0.8, 0.8) for i in label_list])
        a = np.ones([rgb.shape[0], 1])
        rgba = np.hstack([rgb, a])
        bone_pts = trimesh.PointCloud(vertices=bone_obj.reshape(-1, 3), colors=rgba)
        return bone_pts
    else:
        if isinstance(bone_obj.visual, trimesh.visual.TextureVisuals):
            bone_obj.visual = bone_obj.visual.to_color()
            bone_obj.visual.vertex_colors = np.zeros([len(bone_obj.vertices), 4])
        for i in range(len(label_list)):
            (r, g, b) = colorsys.hsv_to_rgb(label_list[i] * 1.0 / bone_num, 0.8, 0.8)
            bone_obj.visual.vertex_colors[i] = (r * 255, g * 255, b * 255, 255)
        return bone_obj