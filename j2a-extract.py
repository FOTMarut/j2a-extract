from __future__ import print_function
import os
import sys
import logging
from logging import error, warning, info
import argparse
import j2a
from j2a import J2A, FrameConverter

if sys.version_info[0] <= 2:
    input = raw_input

def legacy_extractor(animsfilename, outputdir = None, palette_file = "Diamondus_2.pal"):
    if outputdir is None:
        outputdir = os.path.join(os.path.dirname(animsfilename), os.path.basename(animsfilename).replace('.', '-'))
    j2a = J2A(animsfilename).read()
    info("Extracting to: %s", outputdir)
    renderer = FrameConverter(palette_file = palette_file)
    for setnum, s in enumerate(j2a.sets):
        s = j2a.sets[setnum]
        setdir = os.path.join(outputdir, str(setnum))
        if not os.path.exists(setdir):
            os.makedirs(setdir)
        for animnum, anim in enumerate(s.animations):
            animdir = os.path.join(setdir, str(animnum))
            if not os.path.exists(animdir):
                os.makedirs(animdir)
            fps_filename = os.path.join(animdir, "fps.%d" % anim.fps)
            open(fps_filename, "a").close() # Touch fps file, leave it empty
            for framenum, frame in enumerate(anim.frames):
                frameid = str(framenum)
                if frame.tagged:
                    frameid += "t"
                imgfilename = os.path.join(animdir, "{0:s},{1:d},{2:d},{3:d},{4:d},{5:d},{6:d}.png".format(
                    frameid,
                    *frame.origin +
                     frame.coldspot +
                     frame.gunspot
                ))
                renderer.to_image(frame).save(imgfilename)
        info("Finished extracting set %d (%d animations)", setnum, animnum + 1)
    return 0

def main():
    parser = argparse.ArgumentParser()
    parser.set_defaults(
        anims_file = None,
        destination_folder = None,
        palette = os.path.join(os.path.dirname(sys.argv[0]), "Diamondus_2.pal"),
        log_level = logging.WARNING
    )
    parser.add_argument("anims_file", nargs="?", help="path to the .j2a file to extract")
    parser.add_argument("destination_folder", nargs="?", help="where to extract data (parent folder must exist)")
    parser.add_argument("--palette", help="palette file to use for extraction")
    parser.add_argument("--verbose", action="store_const", const=logging.INFO, dest="log_level",
        help="produce verbose console output (overrides --quiet)")
    parser.add_argument("--quiet", action="store_const", const=logging.ERROR, dest="log_level",
        help="suppress warning messages (overrides --verbose)")
    args = parser.parse_args()

    logging.basicConfig(format="%(levelname)s:j2a-export: %(message)s", level = args.log_level)
    j2a.logger.setLevel(args.log_level)

    animsfilename = args.anims_file if not args.anims_file is None else \
        input("Please type the path to the .j2a file you wish to extract (current folder: %s):\n" % os.getcwd())
    return legacy_extractor(animsfilename, args.destination_folder, args.palette)

if __name__ == "__main__":
    sys.exit(main())
