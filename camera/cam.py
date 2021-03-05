import os
import cv2
import numpy as np

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
        Rt = np.stack([R, t], axis=1)
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