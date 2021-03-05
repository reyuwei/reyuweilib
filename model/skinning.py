import os
import numpy as np
from scipy.interpolate import Rbf
import os 
from pathlib import Path
from scipy.spatial.transform import Rotation
import trimesh
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def RBF_weights(volume, control_pts, weight=None):
    # volume : D* H* W, 3
    # pts: N, 3
    # weight: N, M
    if weight is None:
        weight = np.eye(control_pts.shape[0])

    xyz = volume.reshape(-1, 3)
    chunk = 50000
    rbfi = Rbf(control_pts[:, 0], control_pts[:, 1], control_pts[:, 2], weight, function="thin_plate", mode="N-D")
    # rbfi = Rbf(pts[:, 0], pts[:, 1], pts[:, 2], weight, function="multiquadric", mode="N-D")
    weight_volume = np.concatenate([rbfi(xyz[j:j + chunk, 0], xyz[j:j + chunk, 1], xyz[j:j + chunk, 2]) for j in
                                    range(0, xyz.shape[0], chunk)], 0)
    weight_volume[weight_volume < 0] = 0
    weight_volume = weight_volume / np.sum(weight_volume, axis=1).reshape(-1, 1)
    weight_volume = weight_volume.reshape(xyz.shape[0], -1)
    return weight_volume

def joint_to_bone_mesh(joints, sample=10, use_cylinder=False):
    body_ske_namelist = {
    'head': [17,15,15,0,0,16,16,18],
    'RightArm': [2,3],
    'RightForeArm': [3,4],
    'LeftArm':[5,6],
    'LeftForeArm':[6,7],
    'Spine':[1,8,2,1,1,5,9,8,8,12,5,9,2,12],
    'RightUpLeg':[9,10],
    'RightLeg':[10,11],
    'RightFoot':[23, 11, 11, 24],
    'LeftUpLeg':[12, 13],
    'LeftLeg':[13, 14],
    'LeftFoot':[21, 14, 14, 20],
    }

    keys = list(body_ske_namelist.keys())
    bone_num = len(keys)
    
    bone_ske = []
    bone_ske_weight = []
    t = np.linspace(0.15, 0.85, sample).reshape(-1, 1)

    for ki in range(bone_num):
        bone_name = keys[ki]
        bone_link = np.array(body_ske_namelist[bone_name])
        if len(bone_link) % 2 == 0:
            bone_link = bone_link.reshape(-1, 2)
            for link in bone_link:
                i, joint_parent = link
                one_bone_line = joints[i] + t * (joints[joint_parent] - joints[i])
                
                if use_cylinder:
                    bone_length = np.linalg.norm(joints[joint_parent] - joints[i])
                    cylinder = trimesh.primitives.Cylinder(height=bone_length * 9. / 10., radius=bone_length / 20.0, sections=8)
                    rot_target = (joints[joint_parent] - joints[i]) / bone_length
                    rot_from = cylinder.direction
                    rot_mat = np.eye(4)
                    rot_mat[:3, :3] = Rotation.align_vectors(rot_target[None, ...], rot_from[None, ...])[0].as_matrix()
                    cylinder.apply_transform(rot_mat)
                    cylinder.apply_translation((joints[i] + joints[joint_parent]) / 2)
                    bone_ske.extend(np.vstack([cylinder.vertices, np.array(one_bone_line)]))
                    weights = np.zeros([t.shape[0] + cylinder.vertices.shape[0], bone_num])
                else:
                    bone_ske.extend(one_bone_line)
                    weights = np.zeros([t.shape[0], bone_num])
                
                weights[:, ki] = 1.0
                bone_ske_weight.extend(weights)
        else:
            bone_ske.extend(joints[bone_link[0]].reshape(1, 3))
            weights = np.zeros([1, bone_num])
            weights[:, ki] = 1.0
            bone_ske_weight.extend(weights)

    bone_ske = np.stack(bone_ske).reshape(-1, 3)
    bone_ske_weight = np.stack(bone_ske_weight).reshape(-1, bone_num)
    return bone_ske, bone_ske_weight, keys, bone_num

joints = np.loadtxt("girl_1_rotated\\openpose\\skeleton_body\\skeleton.txt")
ply = trimesh.load("girl_1_rotated\\cr\\mesh.ply", process=False)
savepath = "girl_1_rotated\\cr\\rbf\\"
os.makedirs(savepath, exist_ok=True)
ply_verts = ply.vertices

bone_ske, bone_ske_weight, bone_names, bone_num = joint_to_bone_mesh(joints, sample=15, use_cylinder=True)
ply_verts_weight = RBF_weights(ply_verts, bone_ske, bone_ske_weight)
np.savetxt(savepath + "mesh_weight_rbf.txt", ply_verts_weight)

with open(savepath + "bone_list.txt", "w") as f:
    for s in bone_names:
        f.write(s+"\n")

# visualize
import colorsys
labelid = np.argmax(bone_ske_weight, axis=1)
rgb = np.stack([colorsys.hsv_to_rgb(i * 1.0 / bone_num, 0.8, 0.8) for i in labelid])
np.savetxt(savepath + "bone_weight_rbf_vis.obj", np.hstack([bone_ske, rgb]),
                   fmt="v %.5f %.5f %.5f %.5f %.5f %.5f ")
labelid = np.argmax(ply_verts_weight, axis=1)
rgb = np.stack([colorsys.hsv_to_rgb(i * 1.0 / bone_num, 0.8, 0.8) for i in labelid])
np.savetxt(savepath + "mesh_weight_rbf_vis.obj", np.hstack([ply_verts, rgb]),
                   fmt="v %.5f %.5f %.5f %.5f %.5f %.5f ")