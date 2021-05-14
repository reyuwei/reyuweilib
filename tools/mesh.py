import numpy as np
import trimesh
import torch
from scipy.spatial.transform import Rotation


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


def joints2mesh_body(joints, sample=10, use_cylinder=False):
    '''
    convert openpose body joints to a skeleton mesh, connected with 3D points

    Return
    bone_ske: skeleton vertices  [25, 3]
    bone_ske_weight: skeleton vertices weights [25, M]
    keys: bone name list [M]
    bone_num: bone number = M
    '''
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


def joints2mesh_hand(hand_joints, sample=10, use_cylinder=False):
    '''
    convert hand joints to a skeleton mesh, connected with 3D points

    Return
    bone_ske: skeleton vertices  [N, 3]
    bone_ske_weight: skeleton vertices weights [N, M]
    weight_bone_list: bone name list [M]
    bone_num: bone number = M
    '''
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
            'Hand', 'HandIndex1', 'HandIndex2', 'HandIndex3',
            'HandMiddle1', 'HandMiddle2', 'HandMiddle3',
            'HandPinky1', 'HandPinky2', 'HandPinky3',
            'HandRing1', 'HandRing2', 'HandRing3',
            'HandThumb1', 'HandThumb2', 'HandThumb3',
        ]
    elif joint_count == 25: # piano
        bone_count = 20
        BONE_PARENT_LABEL_DICT = {
            0: [-1, -1],
            1: [0, 0],
            2: [1, 1],
            3: [2, 2],
            4: [3, 3],
            5: [0, 0],
            6: [5, 4],
            7: [6, 5],
            8: [7, 6],
            9: [8, 7],
            10: [0, 0],
            11: [10, 8],
            12: [11, 9],
            13: [12, 10],
            14: [13, 11],
            15: [0, 0],
            16: [15, 12],
            17: [16, 13],
            18: [17, 14],
            19: [18, 15],
            20: [0, 0],
            21: [20, 16],
            22: [21, 17],
            23: [22, 18],
            24: [23, 19],
        }
        weight_bone_list = [
            'Hand_wrist',
            'HandThumb1', 'HandThumb2', 'HandThumb3'
            'HandIndex1', 'HandIndex2', 'HandIndex3','HandIndex4',
            'HandMiddle1', 'HandMiddle2', 'HandMiddle3','HandMiddle4',
            'HandRing1', 'HandRing2', 'HandRing3', 'HandRing4',
            'HandPinky1', 'HandPinky2', 'HandPinky3','HandPinky4',
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
    return bone_ske, bone_ske_weight, weight_bone_list, bone_count