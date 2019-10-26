
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

def show_anim(set_num, anim_num):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import ArtistAnimation
    import misc

    anims = _read_hdr()
    s = anims.sets[set_num]
    if anim_num >= s.header["animcount"]:
        raise KeyError("Animation number %d is out of bounds for set %d (must be between 0 and %d)"
            % (anim_num, set_num, s.header["animcount"]-1))
    animinfo = s.get_substream(1)
    frameinfo = s.get_substream(2)
    frameoffset = 0
    for i in range(0, anim_num):
        ainfo = misc.named_unpack(J2A._animinfostruct, animinfo[i*8:(i*8)+8])
        frameoffset += ainfo["framecount"]
    ainfo = misc.named_unpack(J2A._animinfostruct, animinfo[anim_num*8:(anim_num*8)+8])
    framecount, fps = ainfo["framecount"], ainfo["fps"]
    frameoffset *= 24
    frameinfo_l = [
        misc.named_unpack(J2A._frameinfostruct, frameinfo[frameoffset:frameoffset+24])
        for frameoffset in range(frameoffset, frameoffset + 24*framecount, 24)
    ]
    imageoffsets = [finfo["imageoffset"] for finfo in frameinfo_l]
    imagedata = s.get_substream(3)
    images = [anims.render_pixelmap(anims.make_pixelmap(imagedata[offset:])) for offset in imageoffsets]

    borders = np.array([[finfo["hotspotx"], finfo["width"], finfo["hotspoty"], finfo["height"]] for finfo in frameinfo_l])
    borders[:,1] += borders[:,0]
    borders[:,3] += borders[:,2]
    extremes = ((borders[:,0].min(), borders[:,1].max()), (borders[:,2].min(), borders[:,3].max()))

    fig, ax = plt.subplots()
    artists = [[plt.imshow(image, animated=True, extent=borders[i])] for i,image in enumerate(images)]
    ani = ArtistAnimation(fig, artists, interval=1000.0/fps, blit=True)
    ax.axis("off")
    ax.set_xlim(extremes[0])
    ax.set_ylim(extremes[1])
    plt.show()

def _read_hdr():
    global anims_path
    return J2A(anims_path).read()

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
