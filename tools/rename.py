import os
import re 
from PIL import Image
import PIL.ExifTags


def get_orientation(original_im):
    exif = {
        PIL.ExifTags.TAGS[k]: v
        for k, v in original_im._getexif().items()
        if k in PIL.ExifTags.TAGS
    }
    return exif['Orientation']

# folder = 'C:\\Users\\liyuwei\\Desktop\\real\\calib_1029\\1\\'
# new_folder = 'C:\\Users\\liyuwei\\Desktop\\real\\calib_1029\\1_rotated'

folder = "C:\\Users\\liyuwei\\Desktop\\real\\girl_1\\"
new_folder = "C:\\Users\\liyuwei\\Desktop\\real\\girl_1_rotated\\"

if not os.path.exists(new_folder):
    os.mkdir(new_folder)

img_files = os.listdir(folder)
for f in img_files:
    if ('.jpg' in f) or ('.png' in f):
        orig_f = folder + '/' + f
        # rename 
        # cam_id = re.findall(r"\d+\.?\d*",f)
        cam_id = re.findall(r"\d*",f)
        cam_id = int(cam_id[0])
        new_name = "image.cam%02d_000000.png" % cam_id
        des_f = new_folder + '/' + new_name 

        # rotate 
        original_im = Image.open(orig_f)
        orientation = get_orientation(original_im)
        
        rotated_im = None
        if orientation == 3:
            rotated_im=original_im.rotate(180, expand=True)
        elif orientation == 6:
            rotated_im=original_im.rotate(270, expand=True)
        elif orientation == 8:
            rotated_im=original_im.rotate(90, expand=True)
        else:
            print(orientation)
        
        if rotated_im is not None:
            rotated_im.save(des_f)