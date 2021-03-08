import os
import sys
import subprocess
from pathlib import Path


rootpath = Path("F:\\DATA\\handdome_21\\STATIC_0128\\")
crbox = rootpath / "reconstructionRegion.rcbox"

for name in os.listdir(rootpath):
    if os.path.isdir(rootpath / name):
        if name != "xmp" and name != "calib":
            folder = rootpath / name
            frames = os.listdir(folder)
            for f in frames[:4]:
                input_folder = folder / f
                os.makedirs(input_folder / "cr", exist_ok=True)
                save_bdproj = input_folder / "cr" / "bd.rcproj"
                save_model = input_folder / "cr" / (f + ".obj")
                save_proj = input_folder / "cr" / (f + ".rcproj")

                cmd = "process_one.bat " + str(input_folder) + " " + str(crbox) + " " + str(save_bdproj) + " " + str(save_model) + " " + str(save_proj)
                print(cmd)
                subprocess.run(cmd)



rootpath = Path("F:\\DATA\\handdome_21\\STATIC_0130\\")
crbox = rootpath / "reconstructionRegion.rcbox"

for name in os.listdir(rootpath):
    if os.path.isdir(rootpath / name):
        if name != "xmp" and name != "calib":
            folder = rootpath / name
            frames = os.listdir(folder)
            for f in frames[:4]:
                input_folder = folder / f
                os.makedirs(input_folder / "cr", exist_ok=True)
                save_bdproj = input_folder / "cr" / "bd.rcproj"
                save_model = input_folder / "cr" / (f + ".obj")
                save_proj = input_folder / "cr" / (f + ".rcproj")

                cmd = "process_one.bat " + str(input_folder) + " " + str(crbox) + " " + str(save_bdproj) + " " + str(save_model) + " " + str(save_proj)
                print(cmd)
                subprocess.run(cmd)

