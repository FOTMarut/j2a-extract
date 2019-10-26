from __future__ import print_function
import os
import sys
import struct
import misc
from j2a import J2A #http://www.jazz2online.com/junk/j2a.py

if sys.version_info[0] <= 2:
    input = raw_input

def main():
    animsfilename = sys.argv[1] if (len(sys.argv) >= 2) else \
        input("Please type the animsfilename of the .j2a file you wish to extract:\n")
    j2a = J2A(animsfilename).read()
    outputdir = os.path.join(os.path.dirname(animsfilename), os.path.basename(animsfilename).replace('.', '-'))
    for setnum in range(j2a.header["setcount"]):
        s = j2a.sets[setnum]
        animinfo = s.get_substream(1)
        frameinfo = s.get_substream(2)
        imagedata = s.get_substream(3)
#         print("# set {:3}: ulengths {:3} {:5} {:8}".format(
#             setnum, *map(len, (animinfo, frameinfo, imagedata))
#         ))
        setdir = os.path.join(outputdir, str(setnum))
        if not os.path.exists(setdir):
             os.makedirs(setdir)
        for animnum in range(s.header["animcount"]):
            thisaniminfo = misc.named_unpack(j2a._animinfostruct, animinfo[:8])
            animinfo = animinfo[8:]
            dirname = os.path.join(setdir, str(animnum))
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            fps_filename = os.path.join(dirname, "fps.%d" % thisaniminfo["fps"])
            open(fps_filename, "a").close() # Touch fps file, leave it empty
            for framenum in range(thisaniminfo["framecount"]):
                thisframeinfo = misc.named_unpack(j2a._frameinfostruct, frameinfo[:24])
                frameinfo = frameinfo[24:]
                raw = imagedata[thisframeinfo["imageoffset"]:]
                frameid = str(framenum)
                if (struct.unpack_from("<H", raw)[0] >= 32768):
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
                j2a.render_paletted_pixelmap(j2a.make_pixelmap(raw)).save(imgfilename)
        print("Finished extracting set %d (%d animations)" % (setnum, animnum + 1))

if __name__ == "__main__":
    main()
