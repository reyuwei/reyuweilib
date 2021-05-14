import os
import numpy as np
import xml.dom.minidom
import re
import xml.etree.cElementTree as ET
import os
from pathlib import *  
import py3drendutils


class Camera:
    def __init__(self, K, Rt, name="", w=None, h=None):
        self.K = np.array(K.copy())
        self.Rt = np.array(Rt.copy())
        self.name = name
        if w is not None:
            self.width = w
        if h is not None:
            self.height = h
    
    @classmethod
    def fromkrt(cls, K, R, t, w, h, name=""):
        K = K.reshape(3,3)
        R = R.reshape(3,3)
        t = t.reshape(3,1)
        Rt = np.concatenate([R, t], axis=1)
        return cls(K, Rt, name, w, h)


    @classmethod
    def fromfile(cls, krtfile, name=""):
        mat = np.loadtxt(krtfile)
        return cls(mat[0:3, 0:3], mat[3:, :], name)    
    
    @property
    def projection(self):
        return self.K.dot(self.extrinsics)
    
    @property
    def R(self):
        return self.Rt[0:3, 0:3]
    
    @property
    def t(self):
        return self.Rt[0:3, -1]

    @property
    def extrinsics(self):
        return self.Rt


    def project3dpoint(self, point):
        point = np.array(point)
        assert(point.shape[0] == 3)
        projection = np.matmul(self.projection, np.hstack([point, 1]).reshape(4, 1))
        projection = projection / projection[-1]
        return np.array([int(projection[0]),int(projection[1])])

    def project3dpoint_tocamera(self, points):
        points= np.array(points) # n, 3
        points = np.concatenate([points, np.ones([points.shape[0], 1])], axis=-1)
        projection = np.matmul(self.extrinsics, points.T).T
        return projection

def readCRKRT(cr_camera_projection):
    print("Loading camera projection matrix from " + str(cr_camera_projection))    
    files = os.listdir(str(cr_camera_projection))
    files.sort()
    cameras = []
    for f in files:
        if "_KRT.txt" in f:
            cam_proj_file = cr_camera_projection / f
            print(f[f.index('cam'):f.index('cam')+5])
            cameras.append(Camera.fromfile(cam_proj_file, name=f[f.index('cam'):f.index('cam')+5]))
    return cameras
  
def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i    

def addheadline(cam_file):
    with open(cam_file, "r") as f:
        inf = f.readlines()
    with open(cam_file, "w") as f:
        f.write("<?xml version=\"1.0\"?>\n")
        f.writelines(inf)

def write_cam(K, R, T, cam_in, cam_ex):
    from xml.etree.ElementTree import Element,ElementTree
    K_str = ''.join(str(e)+" " for e in K)
    T_str = ''.join(str(e)+" " for e in T)
    R_str = ''.join(str(e)+" " for e in R)

    content = [
    {
        'name':'M',
        'type':'in',
        'rows':'3',
        'cols':'3',
        'dt':'d',
        'data':K_str
    },
    {
        'name':'D',
        'type':'in',
        'rows':'5',
        'cols':'1',
        'dt':'d',
        'data':'0 0 0 0 0'
    },
    {
        'name':'R',
        'type':'ex',
        'rows':'3',
        'cols':'3',
        'dt':'d',
        'data':R_str
    },
    {
        'name':'T',
        'type':'ex',
        'rows':'3',
        'cols':'1',
        'dt':'d',
        'data':T_str
    }
    ]
    
    root_in = Element('opencv_storage')
    tree_in = ElementTree(root_in)
    
    root_ex = Element('opencv_storage')
    tree_ex = ElementTree(root_ex)
    
    for cont in content:
        for k,v in cont.items():
            if k == 'name':
                child0 = Element(v)
                child0.set("type_id","opencv-matrix")
                continue
            if v == "in":
                root_in.append(child0)
            elif v == 'ex':
                root_ex.append(child0)
            else:
                child00 = Element(k)
                child00.text = v
                child0.append(child00)
    
    indent(root_in,0)
    tree_in.write(cam_in, 'UTF-8')#, xml_declaration=True)
    
    indent(root_ex,0)
    tree_ex.write(cam_ex, 'UTF-8')#, xml_declaration=True)
    
    addheadline(cam_in)
    addheadline(cam_ex)

def camloadfromCalibFolder(root_path):
    print("Loading camera from " + str(root_path))    
    files= os.listdir(root_path)
    files.sort()
    cameras = []
    with open(root_path / "Idmap.txt") as f:
        ids = f.readlines()
    ws = []
    hs = []
    for line in ids[1:]:
        w = int(line.split(" ")[2])
        h = int(line.split(" ")[3].replace("\n",""))
        ws.append(w)
        hs.append(h)

    i = 0
    for cam_file in files:
        path = root_path / cam_file
        if not os.path.isdir(path):
            continue
        if not path.exists():
            continue
        extrinsics = path / 'extrinsics.xml'
        intrinsic = path / 'intrinsic.xml'
        w = ws[i]
        h = hs[i]
        i = i+1

        # load extrinsic
        tree = ET.ElementTree(file=extrinsics)
        root = tree.getroot()
        # get R
        R_node = root[0] 
        text = R_node[3].text
        R_data = re.split('[\s]\s*',text)
        R_data.remove('')
        R = list(map(eval, R_data))
        R = np.array(R)
        R = R.reshape(3,3)
        R_mat = np.matrix(R)
        #print(R)

        # get T
        T_node = root[1]
        text = T_node[3].text
        T_data = re.split('[\s]\s*',text)
        T_data.remove('')
        T = list(map(eval, T_data))
        T = np.array(T)
        T = T.reshape(3,1)
        T_mat = np.matrix(T)
        #print(T)

        # load intrinsic
        tree = ET.ElementTree(file=intrinsic)
        root = tree.getroot()
        # get K
        date = root[0]
        if date.tag == "date":
            M_node = 2
        else:
            M_node = 0

        K_node = root[M_node] 
        #K_node = root[2] 
        text = K_node[3].text
        K_data = re.split('[\s]\s*',text)
        K_data.remove('')
        K = list(map(eval, K_data))
        K = np.array(K)
        K = K.reshape(3,3)
        K_mat = np.matrix(K)
        #print(K)

        cam_name = cam_file
        # print("load camera", cam_name)
        cameras.append(Camera.fromkrt(K, R, T, w, h, cam_name))
    return cameras

import colorsys

def visual_weights(bone_obj, label_list):

    '''
    visulize_weight 
    
    Parameters:
    bone_obj: trimesh with N vertices / np.array [N, 3]
    label_list: np.array [N]

    Return:
    colored bone obj (trimesh.mesh or trimesh.pointcloud)
    '''
    if isinstance(bone_obj, np.ndarray):
        # rgb = np.stack([colorsys.hsv_to_rgb(i * 1.0 / bone_num, 0.8, 0.8) for i in label_list])
        rgb = np.stack([(i+1)*10,(i+1)*10,(i+1)*10] for i in label_list)
        a = np.ones([rgb.shape[0], 1])
        rgba = np.hstack([rgb, a])
        bone_pts = trimesh.PointCloud(vertices=bone_obj.reshape(-1, 3), colors=rgba)
        return bone_pts
    else:
        if isinstance(bone_obj.visual, trimesh.visual.TextureVisuals):
            bone_obj.visual = bone_obj.visual.to_color()
            bone_obj.visual.vertex_colors = np.zeros([len(bone_obj.vertices), 4])
        
        color = (label_list + 1)*10
        vertex_colors = np.vstack([color, color, color, np.ones_like(color)*255]).T
        face_colors = vertex_colors[bone_obj.faces].max(axis=1)
        bone_obj.visual.face_colors = face_colors
        bone_obj.visual.vertex_colors = vertex_colors
        return bone_obj, face_colors

import sys

if len(sys.argv) < 6:
    print("Arguments: src_model, src_label(one column), src_mask_folder, camera_file_folder, output_folder, [downsample_ratio=10](optional), [batch_size=2](optional, to speed up)")

if __name__ == "__main__":

# python label_projection.py C:\Users\liyuwei\Desktop\real\calib_1029\1_rotated\mesh_trimesh.ply C:\Users\liyuwei\Desktop\real\calib_1029\1_rotated\rbf\mesh_weight_labellist.txt C:\Users\liyuwei\Desktop\real\calib_1029\1_rotated\mask_project\ C:\Users\liyuwei\Desktop\real\calib_1029\1_rotated\calib C:\Users\liyuwei\Desktop\real\calib_1029\1_rotated\project_smask_batch_speedtest 10 2

    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv))

    ply_path = sys.argv[1]
    label_path = sys.argv[2]
    binary_mask_img_path = sys.argv[3]
    calib_folder_path = Path(sys.argv[4])
    save_mask_img_path = Path(sys.argv[5])
    if len(sys.argv)>6:
        down_scale = float(sys.argv[6])
    else:
        down_scale=10.0
    if len(sys.argv) > 7:
        batch_size = int(sys.argv[7])
    else:
        batch_size=2

    # camerapath = Path("C:\\Users\\liyuwei\\Desktop\\real\\calib_1029\\1_rotated\\")
    cameras = camloadfromCalibFolder(calib_folder_path)

    # Data structures and functions for rendering
    import torch
    # Setup
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # Util function for loading meshes
    import trimesh
    import cv2
    # mesh = trimesh.load(str(camerapath / "mesh_simp_trimesh.ply"), process=False)
    mesh = trimesh.load(ply_path, process=False)
    os.makedirs(save_mask_img_path, exist_ok=True)
    
    imgs = [os.path.join(binary_mask_img_path, x) for x in os.listdir(binary_mask_img_path) if ".png" in x and "cam" in x]
    if len(imgs) == 0:
        imgs = [os.path.join(binary_mask_img_path, x) for x in os.listdir(binary_mask_img_path) if ".jpg" in x and "cam" in x]
    imgs.sort()
    assert len(imgs) == len(cameras)

    ply_label = np.loadtxt(label_path)

    assert(len(ply_label) == mesh.vertices.shape[0])

    _, face_colors = visual_weights(mesh, ply_label)
    face_colors = torch.from_numpy(face_colors / 255.0).float().unsqueeze(0).to(device)
    # mesh_color.show()
    pts = mesh.vertices
    faces = torch.from_numpy(mesh.faces).to(device)

    batch_verts = torch.tensor(()).to(device)
    batch_cameras = torch.tensor(()).to(device)
    batch_image_sizes = torch.tensor(()).to(device)
    batch_camera_ids = []
    img_names = []
    for img_rgb_f in imgs:
        # print(img_rgb_f)
        cam_id = int(Path(img_rgb_f).stem[9:11])
        c = cameras[cam_id-1]
        assert int(cam_id) == int(c.name[3:5])
        batch_camera_ids.append(cam_id-1)
        img_names.append(img_rgb_f)

        verts = c.project3dpoint_tocamera(pts)
        verts = torch.from_numpy(verts).float().to(device)

        K = torch.eye(4).to(device)
        K[:3, :3] = torch.from_numpy(c.K).float().to(device)
        K[0,0] /= down_scale
        K[1,1] /= down_scale
        K[0,2] /= down_scale
        K[1,2] /= down_scale
        K = K.unsqueeze(0)

        batch_verts = torch.cat([batch_verts, verts.unsqueeze(0)], axis=0)
        batch_cameras = torch.cat([batch_cameras, K], axis=0)
        image_sizes = [c.width / down_scale, c.height/down_scale]
        batch_image_sizes = torch.cat([batch_image_sizes, torch.tensor(image_sizes).unsqueeze(0).to(device)], axis=0)


    batch_faces = faces.unsqueeze(0).repeat(batch_verts.shape[0], 1, 1)
    face_colors = face_colors.repeat(batch_verts.shape[0], 1, 1)
    save_image_smasks = []
    for bi in range(0, batch_verts.shape[0], batch_size):
        images = py3drendutils.batch_render(batch_verts[bi:bi+batch_size],
                                        batch_faces[bi:bi+batch_size],
                                        K=batch_cameras[bi:bi+batch_size],
                                        image_sizes=batch_image_sizes[bi:bi+batch_size],
                                        mode="facecolor",
                                        shading="hard",
                                        face_colors=face_colors[bi:bi+batch_size])
        image_smasks = images[:, ..., 0:1].cpu().numpy()*255.0
        # save_image_smasks.append(save_image_smasks)
        # save_image_smasks = np.stack(save_image_smasks)

        for bii in range(image_smasks.shape[0]):
            camid = batch_camera_ids[bi+bii]
            image_smask = image_smasks[bii]
            image_smask_back = cv2.resize(image_smask, (cameras[camid].width, cameras[camid].height), interpolation = cv2.INTER_NEAREST)
            image_smask_back = np.round(image_smask_back)
            savepath = str(save_mask_img_path / (Path(img_names[bi+bii]).stem + ".png"))
            print(savepath, image_smask_back.shape)
            cv2.imwrite(savepath, image_smask_back)


    # for img_rgb_f in imgs:
    #     # print(img_rgb_f)
    #     cam_id = int(Path(img_rgb_f).stem[9:11])
    #     c = cameras[cam_id-1]
    #     assert int(cam_id) == int(c.name[3:5])

    #     verts = c.project3dpoint_tocamera(pts)
    #     verts = torch.from_numpy(verts).float().to(device)

    #     K = torch.eye(4).to(device)
    #     K[:3, :3] = torch.from_numpy(c.K).float().to(device)
    #     K[0,0] /= down_scale
    #     K[1,1] /= down_scale
    #     K[0,2] /= down_scale
    #     K[1,2] /= down_scale
    #     K = K.unsqueeze(0)
        
    #     image_sizes = [[c.width / down_scale, c.height/down_scale]]
    #     images = py3drendutils.batch_render(verts.unsqueeze(0),
    #                                     faces.unsqueeze(0),
    #                                     K=K,
    #                                     image_sizes=image_sizes,
    #                                     mode="facecolor",
    #                                     shading="hard",
    #                                     face_colors=face_colors)
    #     image_smask = images[0, ..., 0:1].cpu().numpy()*255.0
    #     image_smask_back = cv2.resize(image_smask, (c.width, c.height), interpolation = cv2.INTER_NEAREST)
    #     image_smask_back = np.round(image_smask_back)
    #     savepath = str(save_mask_img_path / (Path(img_rgb_f).stem + ".png"))
    #     print(savepath, image_smask_back.shape)
    #     cv2.imwrite(savepath, image_smask_back)