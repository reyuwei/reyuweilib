import json
import os
import numpy as np
from pathlib import Path
import subprocess
import shutil
import sys

r3dsserver = "F:\\R3DS\\Wrap 3.4\\Wrap3Cmd.exe"

def batch_wrap3d(floating_obj_path, fixed_obj_path, save_obj_path):
    # create custom wrap project file
    if not isinstance(floating_obj_path, list):
        floating_obj_path = [floating_obj_path]
        fixed_obj_path = [fixed_obj_path]

    frame_num = len(floating_obj_path)
    template_wrap = Path("wrap3d\\wrapping_template.wrap")
    wrap_json = json.load(open(template_wrap))

    wrap_json['nodes']['template_bone']['params']['fileNames']['value'] = floating_obj_path
    wrap_json['nodes']['capture_bone']['params']['fileNames']['value'] = fixed_obj_path
    wrap_json['nodes']['SaveGeom']['params']['fileName']['value'] = save_obj_path + "\\##.obj"

    # save tmp wrap project
    save_wrap_proj = str(np.random.randint(9999)) +  "tmp.wrap"
    with open(save_wrap_proj, 'w') as outfile:
        json.dump(wrap_json, outfile)

    # wrap cmd
    cmd = r3dsserver + " compute " + str(save_wrap_proj) + " -s 1 -e " + str(frame_num)
    subprocess.run(cmd)
    
    # delete tmp wrap project
    os.remove(save_wrap_proj)