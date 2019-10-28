from __future__ import print_function
import os
import sys
import struct
from j2a import J2A #http://www.jazz2online.com/junk/j2a.py

if sys.version_info[0] <= 2:
    input = raw_input

def main():
    animsfilename = sys.argv[1] if (len(sys.argv) >= 2) else \
        input("Please type the animsfilename of the .j2a file you wish to extract:\n")
    j2a = J2A(animsfilename).read()
    outputdir = os.path.join(os.path.dirname(animsfilename), os.path.basename(animsfilename).replace('.', '-'))
    for setnum,s in enumerate(j2a.sets):
        s = j2a.sets[setnum]
        setdir = os.path.join(outputdir, str(setnum))
        if not os.path.exists(setdir):
             os.makedirs(setdir)
        for animnum,anim in enumerate(s.animations):
            dirname = os.path.join(setdir, str(animnum))
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            fps_filename = os.path.join(dirname, "fps.%d" % anim.fps)
            open(fps_filename, "a").close() # Touch fps file, leave it empty
            for framenum,frame in enumerate(anim.frames):
                thisframeinfo = frame.header
                imageoffset = thisframeinfo["imageoffset"]
                frameid = str(framenum)
                if (struct.unpack_from("<H", frame.data, imageoffset)[0] >= 32768):
                    frameid += "t"
                imgfilename = os.path.join(dirname, "%s,%d,%d,%d,%d,%d,%d.png" % (
                    frameid,
                    thisframeinfo["hotspotx"],
                    thisframeinfo["hotspoty"],
                    thisframeinfo["coldspotx"],
                    thisframeinfo["coldspoty"],
                    thisframeinfo["gunspotx"],
                    thisframeinfo["gunspoty"])
                )
                j2a.render_paletted_pixelmap(j2a.make_pixelmap(frame.data, imageoffset)) \
                    .save(imgfilename)
        print("Finished extracting set %d (%d animations)" % (setnum, animnum + 1))

if __name__ == "__main__":
    main()
