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

#http://stackoverflow.com/questions/2669059/how-to-sort-alpha-numeric-set-in-python
convert = lambda text: int(text) if text.isdigit() else text
alphanum_key = lambda key: [ convert(c) for c in re.split('(\d+)', key) ]


def main():
    dirname = sys.argv[1] if (len(sys.argv) >= 2) else \
        input("Please type the folder you wish to import:\n")
    dirname = os.path.normpath(dirname)
    outfilename = sys.argv[2] if (len(sys.argv) >= 3) else \
            dirname.replace("-j2a", ".j2a")
    if not os.path.isdir(dirname):
        print("Folder does not exist!")
        return 1
    if not re.search(r"-j2a$", os.path.basename(dirname)):
        print("Folder name is improperly formatted (must be named ***-j2a)!")
        return 1
    setdirlist = [os.path.join(dirname, name) for name in sorted(os.listdir(dirname)) if os.path.isdir(os.path.join(dirname, name))]
    setdirlist.sort(key=alphanum_key)
    if not setdirlist:
        print("No sets were found in that folder. No .j2a file will be compiled.", file=sys.stderr)
        return 1
    anims = J2A(outfilename)
    for setdir in setdirlist:
        cur_set = J2A.Set()
        animdirlist = sorted([ fullpath
            for name in sorted(os.listdir(setdir))
            for fullpath in ( os.path.join(setdir, name), )
            if name.isdigit() and os.path.isdir(fullpath)
        ], key = alphanum_key)
        for animdir in animdirlist:
            cur_anim = J2A.Animation()
            framelist = sorted(glob.glob( os.path.join(animdir, '*.png') ), key=alphanum_key)
            fpsfile = glob.glob( os.path.join(animdir, 'fps.*') )
            if len(fpsfile) == 1:
                cur_anim.fps = int(os.path.splitext(fpsfile[0])[1][1:])

            for frame_filename in framelist:
                frameinfo = re.match(r"^\d+(t?),(-?\d+),(-?\d+),(-?\d+),(-?\d+),(-?\d+),(-?\d+).png$", os.path.basename(frame_filename)).groups()
                frameinfo = [bool(frameinfo[0])] + [int(val) for val in frameinfo[1:]]
                image = Image.open(frame_filename)
                assert(image.mode == "P")
                frame = J2A.Frame(
                    pixmap = image,
                    origin = tuple(frameinfo[1:3]),
                    coldspot = tuple(frameinfo[3:5]),
                    gunspot = tuple(frameinfo[5:7]),
                    tagged = bool(frameinfo[0])
                )
                frame.autogenerate_mask()
                cur_anim.frames.append(frame)
            cur_set.animations.append(cur_anim)
        cur_set.samplesbaseindex = 0
        cur_set.pack(anims.config)
        anims.sets.append(cur_set)
    anims.write()
    return 0


if __name__ == "__main__":
    sys.exit(main())
