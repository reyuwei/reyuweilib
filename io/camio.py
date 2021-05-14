import os
import re
import numpy as np
from pathlib import Path
import xml.dom.minidom
import xml.etree.cElementTree as ET
# from base.cam import Camera

class Camera:
    def __init__(self, K, Rt, w, h, name=""):
        self.K = np.array(K.copy())
        self.Rt = np.array(Rt.copy())
        self.name = name
        self.width = w
        self.height = h
    
    @classmethod
    def fromkrt(cls, K, R, t, w, h, name=""):
        K = K.reshape(3,3)
        R = R.reshape(3,3)
        t = t.reshape(3,1)
        Rt = np.concatenate([R, t], axis=1)
        return cls(K, Rt, w, h, name)

    @classmethod
    def fromfile(cls, krtfile, w, h, name=""):
        mat = np.loadtxt(krtfile)
        return cls(mat[0:3, 0:3], mat[3:, :], w, h, name)    
    
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

    def project3dpoint_on_image(self, vertices, image):
        for p in vertices:
            p2d = self.project3dpoint(p)
            cv2.circle(image, (int(p2d[0]), int(p2d[1])), 50, (255,0,0), thickness=-1)
        return image
 
def _indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            _indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i    

def _addheadline(cam_file):
    with open(cam_file, "r") as f:
        inf = f.readlines()
    with open(cam_file, "w") as f:
        f.write("<?xml version=\"1.0\"?>\n")
        f.writelines(inf)

def _write_cam(K, R, T, cam_in, cam_ex):
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
    
    _indent(root_in,0)
    tree_in.write(cam_in, 'UTF-8')#, xml_declaration=True)
    
    _indent(root_ex,0)
    tree_ex.write(cam_ex, 'UTF-8')#, xml_declaration=True)
    
    _addheadline(cam_in)
    _addheadline(cam_ex)

def camwrite2CalibFolder(cameras, savefolder):
    # savepath = tofolder / "calib"
    savefolder.mkdir(exist_ok=True)
    idmapf_str = []
    cam_count = 0
    for cam in cameras:
        w = cam.width
        h = cam.height
        cam_count = cam_count+1 
        camfolder = savefolder / cam.name
        camfolder.mkdir(exist_ok=True)
        print(camfolder)
        cam_in = camfolder / "intrinsic.xml"
        cam_ex = camfolder / "extrinsics.xml"
        _write_cam(cam.K.reshape(-1).tolist(), cam.R.reshape(-1).tolist(), cam.t.reshape(-1).tolist(), cam_in, cam_ex)
        
        idmapf_str_cam = str(cam_count) + " " + "\"" + cam.name+"\"" + " " + str(w) + " " + str(h)+"\n"
        idmapf_str.append(idmapf_str_cam)
        
    f = open(str(savefolder / "Idmap.txt"),'w')
    f.write(str(cam_count) + " " + str(cam_count) + " \"Resolution\"\n")
    f.writelines(idmapf_str)
    f.close()


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
        print("name", cam_name)
        cameras.append(Camera.fromkrt(K, R, T, w, h, cam_name))
    return cameras


def camloadfromCRKRT(cr_camera_projection, w, h):
    print("Loading camera projection matrix from " + str(cr_camera_projection))
    files = os.listdir(str(cr_camera_projection))
    files.sort()
    cameras = []
    for f in files:
        if "_KRT.txt" in f:
            cam_proj_file = cr_camera_projection / f
            print("name", f[f.index('cam'):f.index('cam')+5])
            cameras.append(Camera.fromfile(cam_proj_file, w, h, name=f[f.index('cam'):f.index('cam')+5]))
    return cameras