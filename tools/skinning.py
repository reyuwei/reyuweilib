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

def body_joint_to_bone_mesh(joints, sample=10, use_cylinder=False):
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

        if bone_name == "Spine":
            # add a sphere
            # center_idx = np.unique(np.array(body_ske_namelist[bone_name]))
            # center = np.mean(joints[center_idx], axis=0)
            torso_length = np.linalg.norm(joints[8] - joints[1])
            t = np.linspace(0.3, 0.7, 2).reshape(-1, 1)
            spine_line = joints[8] + t * (joints[1] - joints[8])
            for center in spine_line:
                sphere = trimesh.primitives.Sphere(center=center, radius=torso_length / 4.0)

                bone_ske.extend(sphere.vertices)
                weights = np.zeros([sphere.vertices.shape[0], bone_num])

                weights[:, ki] = 1.0
                bone_ske_weight.extend(weights)

        if len(bone_link) % 2 == 0:
            bone_link = bone_link.reshape(-1, 2)
            for link in bone_link:
                i, joint_parent = link
                one_bone_line = joints[i] + t * (joints[joint_parent] - joints[i])
                
                if use_cylinder:
                    bone_length = np.linalg.norm(joints[joint_parent] - joints[i])
                    cylinder = trimesh.primitives.Cylinder(height=bone_length * 9. / 10., radius=bone_length / 30.0, sections=8)
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


import torch
def hand_joint_to_bone_mesh(hand_joints, sample=10, use_cylinder=False):
    joint_count = hand_joints.shape[0]
    bone_ske = []
    bone_ske_weight = []

    if joint_count == 21:  # mano
        bone_count = 16
        BONE_PARENT_LABEL_DICT = {
            0: [-1, -1],
            1: [0, 0],
            2: [1, 13],
            3: [2, 14],
            4: [3, 15],
            5: [0, 0],
            6: [5, 1],
            7: [6, 2],
            8: [7, 3],
            9: [0, 0],
            10: [9, 4],
            11: [10, 5],
            12: [11, 6],
            13: [0, 0],
            14: [13, 10],
            15: [14, 11],
            16: [15, 12],
            17: [0, 0],
            18: [17, 7],
            19: [18, 8],
            20: [19, 9],
        }
        weight_bone_list = [
            # 'Hand', 'HandIndex1', 'HandIndex2', 'HandIndex3',
            # 'HandMiddle1', 'HandMiddle2', 'HandMiddle3',
            # 'HandPinky1', 'HandPinky2', 'HandPinky3',
            # 'HandRing1', 'HandRing2', 'HandRing3',
            # 'HandThumb1', 'HandThumb2', 'HandThumb3',
        ]
    elif joint_count == 25:
        bone_count = 25
        BONE_PARENT_LABEL_DICT = {
            0: [-1, -1],
            1: [0, 0],
            2: [1, 1],
            3: [2, 2],
            4: [3, 3],
            5: [0, 0],
            6: [5, 5],
            7: [6, 6],
            8: [7, 7],
            9: [8, 8],
            10: [0, 0],
            11: [10, 10],
            12: [11, 11],
            13: [12, 12],
            14: [13, 13],
            15: [0, 0],
            16: [15, 15],
            17: [16, 16],
            18: [17, 17],
            19: [18, 18],
            20: [0, 0],
            21: [20, 20],
            22: [21, 21],
            23: [22, 22],
            24: [23, 23],
        }

        weight_bone_list = [
            # 'Hand_wrist',
            # 'HandThumb1', 'HandThumb2', 'HandThumb3', '_end',
            # 'HandIndex1', 'HandIndex2', 'HandIndex3','HandIndex4', '_end',
            # 'HandMiddle1', 'HandMiddle2', 'HandMiddle3','HandMiddle4','_end',
            # 'HandRing1', 'HandRing2', 'HandRing3', 'HandRing4','_end',
            # 'HandPinky1', 'HandPinky2', 'HandPinky3','HandPinky4','_end',
        ]

    if isinstance(hand_joints, torch.Tensor):
        hand_joints = hand_joints.numpy()
    for i in range(1, hand_joints.shape[0]):
        t = np.linspace(0.15, 0.85, sample).reshape(-1, 1)

        joint_parent, joint_weight_id = BONE_PARENT_LABEL_DICT[i]
        one_bone_line = hand_joints[i] + t * (hand_joints[joint_parent] - hand_joints[i])

        # add cylinder
        if use_cylinder:
            bone_length = np.linalg.norm(hand_joints[joint_parent] - hand_joints[i])
            cylinder = trimesh.primitives.Cylinder(height=bone_length * 9. / 10., radius=bone_length / 20.0, sections=8)
            rot_target = (hand_joints[joint_parent] - hand_joints[i]) / bone_length
            rot_from = cylinder.direction
            rot_mat = np.eye(4)
            rot_mat[:3, :3] = Rotation.align_vectors(rot_target[None, ...], rot_from[None, ...])[0].as_matrix()
            cylinder.apply_transform(rot_mat)
            cylinder.apply_translation((hand_joints[i] + hand_joints[joint_parent]) / 2)
            bone_ske.append(np.vstack([cylinder.vertices, np.array(one_bone_line)]))
            weights = np.zeros([t.shape[0] + cylinder.vertices.shape[0], bone_count])
        else:
            bone_ske.append(one_bone_line)
            weights = np.zeros([t.shape[0], bone_count])

        weights[:, joint_weight_id] = 1.0
        bone_ske_weight.append(weights)

    bone_ske = np.stack(bone_ske).reshape(-1, 3)
    bone_ske_weight = np.stack(bone_ske_weight).reshape(-1, bone_count)
    return bone_ske, bone_ske_weight


if __name__ == "__main__":
    joints = np.loadtxt("C:\\Users\\liyuwei\\Desktop\\real\\calib_1029\\1_rotated\\openpose\\skeleton_body\\skeleton_fix.txt")
    ply = trimesh.load("C:\\Users\\liyuwei\\Desktop\\real\\calib_1029\\1_rotated\\mesh.obj", process=False)
    ply.export("C:\\Users\\liyuwei\\Desktop\\real\\calib_1029\\1_rotated\\mesh_trimesh.obj")
    ply.export("C:\\Users\\liyuwei\\Desktop\\real\\calib_1029\\1_rotated\\mesh_trimesh.ply")
    savepath = "C:\\Users\\liyuwei\\Desktop\\real\\calib_1029\\1_rotated\\rbf\\"


    # joints = np.loadtxt("C:\\Users\\liyuwei\\Desktop\\real\\girl_1_agi\\openpose\\skeleton_body\\skeleton.txt")
    # ply = trimesh.load("C:\\Users\\liyuwei\\Desktop\\real\\girl_1_agi\\mesh.obj", process=False)
    # ply.export("C:\\Users\\liyuwei\\Desktop\\real\\girl_1_agi\\mesh_trimesh.obj")
    # ply.export("C:\\Users\\liyuwei\\Desktop\\real\\girl_1_agi\\mesh_trimesh.ply")
    # savepath = "C:\\Users\\liyuwei\\Desktop\\real\\girl_1_agi\\rbf\\"

    os.makedirs(savepath, exist_ok=True)
    ply_verts = ply.vertices
    print(ply_verts.shape)

    bone_ske, bone_ske_weight, bone_names, bone_num = body_joint_to_bone_mesh(joints, sample=15, use_cylinder=True)
    ply_verts_weight = RBF_weights(ply_verts, bone_ske, bone_ske_weight)
    np.savetxt(savepath + "mesh_weight_rbf.txt", ply_verts_weight)

    np.savetxt(savepath + "mesh_weight_labellist.txt", np.argmax(ply_verts_weight, axis=1), fmt="%d")

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
