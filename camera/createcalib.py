import os
import numpy as np
import xml.dom.minidom
import re
import xml.etree.cElementTree as ET
import os
from pathlib import *  

class Camera:
    def __init__(self, K, Rt, name=""):
        self.K = np.array(K.copy())
        self.Rt = np.array(Rt.copy())
        self.name = name
    
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

def readCRKRT(cr_camera_projection):
    print("Loading camera projection matrix from " + str(cr_camera_projection))    
    files = os.listdir(str(cr_camera_projection))
    files.sort()
    cameras = []
    for f in files:
        if "_KRT.txt" in f:
            cam_proj_file = cr_camera_projection / f
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


camerapath = Path("girl_1_rotated\\")
cameras = readCRKRT(camerapath)

savepath = Path("girl_1_rotated\\calib_folder")
savepath.mkdir(exist_ok=True)
w = 6000
h = 4000
idmapf_str = []
cam_count = 0
for cam in cameras:
    cam_count = cam_count+1 
    camfolder = savepath / cam.name
    camfolder.mkdir(exist_ok=True)
    print(camfolder)
    cam_in = camfolder / "intrinsic.xml"
    cam_ex = camfolder / "extrinsics.xml"
    write_cam(cam.K.reshape(-1).tolist(), cam.R.reshape(-1).tolist(), cam.t.reshape(-1).tolist(), cam_in, cam_ex)
    
    idmapf_str_cam = str(cam_count) + " " + "\"" + cam.name+"\"" + " " + str(w) + " " + str(h)+"\n"
    idmapf_str.append(idmapf_str_cam)
    
f = open(str(savepath / "Idmap.txt"),'w')
f.write(str(cam_count) + " " + str(cam_count) + " \"Resolution\"\n")
f.writelines(idmapf_str)
f.close()

import trimesh
import cv2
mesh = trimesh.load("girl_1_rotated\\cr\\mesh.ply")
img_rgb = cv2.imread("girl_1_rotated\\image.cam01_000000.png")
pts = mesh.vertices
for c in cameras[:1]:
    for p in pts[::1000]:
        p2d = c.project3dpoint(p)
        cv2.circle(img_rgb, (int(p2d[0]), int(p2d[1])), 50, (0,255,0))
    # plt.figure()
    # plt.imshow(img_rgb)
    # plt.show()
cv2.imwrite("project.png",img_rgb)