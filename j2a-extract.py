from __future__ import print_function
import os
import sys
import struct
from j2a import J2A

if sys.version_info[0] <= 2:
    input = raw_input

def main():
    animsfilename = sys.argv[1] if (len(sys.argv) >= 2) else \
        input("Please type the animsfilename of the .j2a file you wish to extract:\n")
    outputdir = sys.argv[2] if (len(sys.argv) >= 3) else \
        os.path.join(os.path.dirname(animsfilename), os.path.basename(animsfilename).replace('.', '-'))
    j2a = J2A(animsfilename).read()
    for setnum, s in enumerate(j2a.sets):
        s = j2a.sets[setnum]
        setdir = os.path.join(outputdir, str(setnum))
        if not os.path.exists(setdir):
             os.makedirs(setdir)
        for animnum, anim in enumerate(s.animations):
            dirname = os.path.join(setdir, str(animnum))
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            fps_filename = os.path.join(dirname, "fps.%d" % anim.fps)
            open(fps_filename, "a").close() # Touch fps file, leave it empty
            for framenum, frame in enumerate(anim.frames):
                frameid = str(framenum)
                if frame.tagged:
                    frameid += "t"
                imgfilename = os.path.join(dirname, "{0:s},{1:d},{2:d},{3:d},{4:d},{5:d},{6:d}.png".format(
                    frameid,
                    *frame.origin +
                     frame.coldspot +
                     frame.gunspot
                ))
                j2a.render_paletted_pixelmap(frame).save(imgfilename)
        print("Finished extracting set %d (%d animations)" % (setnum, animnum + 1))

if __name__ == "__main__":
    main()
