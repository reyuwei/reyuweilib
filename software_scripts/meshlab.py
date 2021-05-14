from pathlib import Path
import os
import sys
sys.path.append("software_scripts\\meshlab")
meshlabserver = "D:\\MeshLab\\meshlabserver.exe"

def meshlab_resave(modelname, outputname):
    modelname = Path(modelname)
    cmd = meshlabserver + " -i " + str(modelname) + " -o " + str(outputname) +" -m vc vn"
    print(cmd)
    os.system(cmd)

def meshlab_crop(modelname):
    modelname = Path(modelname)
    outputname = str(modelname.parent) + "\\" + str(modelname.stem) + "_crop.ply"
    scriptname = " -s meshlab\\meshlab_cropsinglehand.mlx"
    cmd = meshlabserver + " -i " + str(modelname) + " -o " + outputname +" -m vc vn"  + scriptname
    print(cmd)
    os.system(cmd)
    return outputname

def meshlab_simp(modelname):
    modelname = Path(modelname)
    outputname = str(modelname.parent) + "\\" + str(modelname.stem) + "_simp.xyz"
    scriptname = " -s meshlab\\meshlab_simple.mlx"
    cmd = meshlabserver + " -i " + str(modelname) + " -o " + outputname  +" -m vc vn" + scriptname
    print(cmd)
    os.system(cmd)
    return outputname
