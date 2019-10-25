
from __future__ import print_function
from j2a import J2A
import os
import sys
from types import FunctionType

def show_frame(set_num, anim_num, frame_num):
    global anims_path
    try:
        import matplotlib.pyplot as plt
        def show_img(img):
            plt.imshow(img)
            plt.axis('off')
            plt.show()
    except ImportError:
        def show_img(img):
            img.show()

    anims = J2A(anims_path)
    frame = anims.get_frame(set_num, anim_num, frame_num)
    if frame:
        show_img(frame[1])

def _read_hdr():
    global anims_path
    anims = J2A(anims_path)
    anims.read()
    return anims

def print_j2a_stats():
    anims = _read_hdr()
    print("Jazz Jackrabbit 2 animations file")
    for k in ("magic", "headersize", "version", "unknown", "filesize", "crc32", "setcount"):
        print("\t{}: {}".format(k, anims.header[k]))
    for i,s in enumerate(anims.sets):
        print("\tSet {}:".format(i))
        setinfo = s.header
        for k in ("animcount", "samplecount", "framecount", "priorsamplecount", "c1", "u1", "c2", "u2", "c3", "u3", "c4", "u4"):
            print("\t\t{}: {}".format(k, setinfo[k]))

def stress_test(initial_set_num = 0):
    import misc
    anims = _read_hdr()
    for setnum in range(initial_set_num, anims.header["setcount"]):
        # The shareware demo (or at least the TSF one) removes some of the animsets to save on filesize,
        # but leaves the order of animations intact, causing gaping holes with offsets of zero in the .j2a file
        if anims.setoffsets[setnum] == 0:
            continue
        s = anims.sets[setnum]
        animinfo = s.get_substream(1)
        frameinfo = s.get_substream(2)
        imagedata = s.get_substream(3)
        for animnum in range(s.header["animcount"]):
            thisaniminfo = misc.named_unpack(anims._animinfostruct, animinfo[:8])
            animinfo = animinfo[8:]
            for framenum in range(thisaniminfo["framecount"]):
                thisframeinfo = misc.named_unpack(anims._frameinfostruct, frameinfo[:24])
                frameinfo = frameinfo[24:]
                raw = imagedata[thisframeinfo["imageoffset"]:]
#                 if (struct.unpack_from("<H", raw)[0] >= 32768):
#                 print("Set {}, anim {}, frame {}".format(setnum, animnum, framenum))
                anims.make_pixelmap(raw)

#############################################################################################################

if __name__ == "__main__":
    fmap = {k: v for k,v in globals().items() if isinstance(v, FunctionType) and not k.startswith('_')}

    assert(int(True) is 1)
    isint = lambda x : x[int(x[:1] in '+-'):].isdigit()

    anims_path = None
    fargs = []
    for arg in sys.argv[2:]:
        if arg.endswith('.j2a'):
            anims_path = arg
        else:
            if isint(arg): # Don't use integers for file names
                arg = int(arg)
            fargs.append(arg)
    anims_path = anims_path or os.path.join(os.path.dirname(sys.argv[0]), "Anims.j2a")

    print("Calling {} with arguments: {}".format(sys.argv[1], fargs))
    fmap[sys.argv[1]](*fargs)
