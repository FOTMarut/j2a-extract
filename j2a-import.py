from __future__ import print_function
import os
import sys
import logging
from logging import error, warning, info
import argparse
import glob
import re
from PIL import Image
import j2a
from j2a import J2A, FrameConverter

if sys.version_info[0] <= 2:
    input = raw_input

#leaf frame order:
#0 1 2 1 0 3 4 5 6 7 8 9 3 4 5 6 7 8 9 0 1 2 1 0 10 11 12 13 14 15 16

frame_filename_pat = re.compile(r"^(\d+)(t?),(-?\d+),(-?\d+),(-?\d+),(-?\d+),(-?\d+),(-?\d+).png$")
def get_numeric_subdirs(folder = "."):
    dirlist = sorted((dirname for dirname in os.listdir(folder) if dirname.isdigit()), key = lambda x : int(x))
    return [dirname for dirname in dirlist if os.path.isdir(dirname)]


def legacy_importer(dirname, outfilename = None, palette = "Diamondus_2.pal"):
    dirname = os.path.abspath(dirname)
    if outfilename is None:
        if not os.path.basename(dirname).endswith("-j2a"):
            error("Folder name is improperly formatted (must be named ***-j2a)!")
            return 1
        outfilename = dirname.replace("-j2a", ".j2a")
    outfilename = os.path.abspath(outfilename)
    fconv = FrameConverter(palette_file = palette)
    os.chdir(dirname)
    setdirlist = get_numeric_subdirs()
    if not setdirlist:
        error("No sets were found in that folder. No .j2a file will be compiled.")
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
                    warning("found file %s not matching frame naming format", os.path.join(set_dir, anim_dir, frame_filename))
            frameinfo_list = sorted(filter(bool, frameinfo_list), key = lambda x : int(x.group(1)))
            fpsfile = [filename[4:] for filename in glob.glob('fps.*') if filename[4:].isdigit()]
            if len(fpsfile) == 1:
                cur_anim.fps = int(fpsfile[0])
            elif len(fpsfile) > 1:
                warning("found multiple fps files in folder %s, ignoring", os.path.join(set_dir, anim_dir))

            try:
                for frame_num, frameinfo in enumerate(frameinfo_list):
                    frame_filename = frameinfo.group()
                    groups = list(frameinfo.groups())
                    groups[2:] = list(map(int, groups[2:]))
                    if int(groups[0]) != frame_num:
                        error("unexpected frame %s, might be a duplicate or a frame is missing" % os.path.join(set_dir, anim_dir, frame_filename))
                        return 1
                    image = Image.open(frame_filename)
                    frame = fconv.from_image(
                        image,
                        origin = tuple(groups[2:4]),
                        coldspot = tuple(groups[4:6]),
                        gunspot = tuple(groups[6:8]),
                        tagged = bool(groups[1])
                    )
                    frame.autogenerate_mask()
                    cur_anim.frames.append(frame)
            except ValueError as e:
                e.args = ("%s: %s" % (e.args[0], os.path.join(dirname, set_dir, anim_dir, frame_filename)), ) + e.args[1:]
                raise

            os.chdir("..")
        cur_set.samplesbaseindex = 0
        cur_set.pack(anims.config)
        os.chdir("..")
    anims.write()
    return 0

def main():
    parser = argparse.ArgumentParser()
    parser.set_defaults(
        source_folder = None,
        anims_file = None,
        palette = os.path.join(os.path.dirname(sys.argv[0]), "Diamondus_2.pal"),
        log_level = logging.WARNING
    )
    parser.add_argument("source_folder", nargs="?", help="path to the folder to import")
    parser.add_argument("anims_file", nargs="?", help="path to the .j2a file to produce (parent folder must exist)")
    parser.add_argument("--palette", help="palette file to use for import")
    parser.add_argument("--verbose", action="store_const", const=logging.INFO, dest="log_level",
        help="produce verbose console output (overrides --quiet)")
    parser.add_argument("--quiet", action="store_const", const=logging.ERROR, dest="log_level",
        help="suppress warning messages (overrides --verbose)")
    args = parser.parse_args()

    logging.basicConfig(format="%(levelname)s:j2a-import: %(message)s", level = args.log_level)
    j2a.logger.setLevel(args.log_level)

    source_dir = args.source_folder if not args.source_folder is None else \
        input("Please type the folder you wish to import (current folder: %s):\n" % os.getcwd())
    return legacy_importer(source_dir, args.anims_file, args.palette)

if __name__ == "__main__":
    sys.exit(main())
