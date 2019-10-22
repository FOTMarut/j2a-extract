import os
import struct
import misc
from j2a import J2A #http://www.jazz2online.com/junk/j2a.py

def main():
    filename = raw_input("Please type the filename of the .j2a file you wish to extract:\n")
    j2a = J2A(filename)
    j2a.read_header()
    for setnum in range(j2a.header["setcount"]):
        if j2a.setoffsets[setnum] == 0: #the shareware demo (or at least the TSF one) removes some of the animsets to save on filesize, but leaves the order of animations intact, causing gaping holes with offests of zero in the .j2a file
            continue
        j2a.load_set(setnum)
        thissetinfo = j2a.setdata[setnum]
        animinfo = j2a.get_substream(1)
        frameinfo = j2a.get_substream(2)
        imagedata = j2a.get_substream(3)
        animnum = -1
        setdir = os.path.join(os.path.dirname(filename), os.path.basename(filename).replace('.', '-'), str(setnum))
        if not os.path.exists(setdir):
             os.makedirs(setdir)
        for animnum in range(thissetinfo["animcount"]):
            thisaniminfo = misc.named_unpack(j2a._animinfostruct, animinfo[:8])
            animinfo = animinfo[8:]
            dirname = os.path.join(setdir, str(animnum))
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            fps = open(os.path.join(dirname, "fps.%d" % thisaniminfo["fps"]), "w")
            fps.close()
            for framenum in range(thisaniminfo["framecount"]):
                thisframeinfo = misc.named_unpack(j2a._frameinfostruct, frameinfo[:24])
                frameinfo = frameinfo[24:]
                raw = imagedata[thisframeinfo["imageoffset"]:]
                frameid = str(framenum)
                if (struct.unpack("H", raw[:2])[0] >= 32768): frameid += "t"
                j2a.render_paletted_pixelmap(j2a.make_pixelmap(raw)).save(os.path.join(dirname, "%s,%d,%d,%d,%d,%d,%d.png" % (frameid, thisframeinfo["hotspotx"], thisframeinfo["hotspoty"], thisframeinfo["coldspotx"], thisframeinfo["coldspoty"], thisframeinfo["gunspotx"], thisframeinfo["gunspoty"])))
        print "Finished extracting set %d (%d animations)" % (setnum, animnum + 1)

if __name__ == "__main__":
    main()
