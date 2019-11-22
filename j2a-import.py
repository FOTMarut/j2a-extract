from __future__ import print_function
import os
import sys
import glob
import struct
import re
import zlib
from PIL import Image
from j2a import J2A
import misc

if sys.version_info[0] <= 2:
    input = raw_input

#leaf frame order:
#0 1 2 1 0 3 4 5 6 7 8 9 3 4 5 6 7 8 9 0 1 2 1 0 10 11 12 13 14 15 16

frame_filename_pat = re.compile(r"^(\d+)(t?),(-?\d+),(-?\d+),(-?\d+),(-?\d+),(-?\d+),(-?\d+).png$")
def get_numeric_subdirs(folder = "."):
    dirlist = sorted((dirname for dirname in os.listdir(folder) if dirname.isdigit()), key = lambda x : int(x))
    return [dirname for dirname in dirlist if os.path.isdir(dirname)]


def main():
    dirname = sys.argv[1] if (len(sys.argv) >= 2) else \
        input("Please type the folder you wish to import:\n")
    dirname = os.path.abspath(dirname)
    if not os.path.basename(dirname).endswith("-j2a"):
        print("Folder name is improperly formatted (must be named ***-j2a)!", file=sys.stderr)
        return 1
    outfilename = sys.argv[2] if (len(sys.argv) >= 3) else dirname.replace("-j2a", ".j2a")
    os.chdir(dirname)
    setdirlist = get_numeric_subdirs()
    if not setdirlist:
        print("No sets were found in that folder. No .j2a file will be compiled.", file=sys.stderr)
        return 1
    anims = J2A(outfilename, empty_set = "crop")
    anims.sets = [J2A.Set() for _ in range(int(setdirlist[-1]) + 1)]
    for set_dir in setdirlist:
        os.chdir(set_dir)
        cur_set = anims.sets[int(set_dir)]
        animdirlist = get_numeric_subdirs()
        num_animations = 1 + int(animdirlist[-1]) if animdirlist else 0
        cur_set.animations = [J2A.Animation() for _ in range(num_animations)]
        for anim_dir in animdirlist:
            os.chdir(anim_dir)
            cur_anim = cur_set.animations[int(anim_dir)]
            framelist = glob.glob('*.png')
            frameinfo_list = list(map(frame_filename_pat.match, framelist))
            for frame_filename, frameinfo in zip(framelist, frameinfo_list):
                if not frameinfo:
                    print("Warning: found file %s/%s/%s not matching frame naming format" %
                        (set_dir, anim_dir, frame_filename), file=sys.stderr)
            frameinfo_list = sorted(filter(bool, frameinfo_list), key = lambda x : int(x.group(1)))
            fpsfile = [filename[4:] for filename in glob.glob('fps.*') if filename[4:].isdigit()]
            if len(fpsfile) == 1:
                cur_anim.fps = int(fpsfile[0])
            elif len(fpsfile) > 1:
                print("Warning: found multiple fps files in folder %s/%s, ignoring" % (set_dir, anim_dir))

            for frame_num, frameinfo in enumerate(frameinfo_list):
                frame_filename = frameinfo.group()
                groups = list(frameinfo.groups())
                groups[2:] = list(map(int, groups[2:]))
                assert int(groups[0]) == frame_num, \
                    "unexected frame %s/%s/%s, might be a duplicate or a frame is missing" % (set_dir, anim_dir, frame_filename)
                image = Image.open(frame_filename)
                assert image.mode == "P", "image file %s is not paletted" % os.path.abspath(frame_filename)
                frame = J2A.Frame(
                    pixmap = image,
                    origin = tuple(groups[2:4]),
                    coldspot = tuple(groups[4:6]),
                    gunspot = tuple(groups[6:8]),
                    tagged = bool(groups[1])
                )
                frame.autogenerate_mask()
                cur_anim.frames.append(frame)
            os.chdir("..")
        cur_set.samplesbaseindex = 0
        cur_set.pack(anims.config)
        os.chdir("..")
    anims.write()
    return 0


if __name__ == "__main__":
    sys.exit(main())
