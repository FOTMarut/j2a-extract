
from __future__ import print_function
from j2a import J2A
import os
import sys
from types import FunctionType

def show_frame(set_num, anim_num, frame_num):
    try:
        import matplotlib.pyplot as plt
        def show_img(img):
            plt.imshow(img)
            plt.axis('off')
            plt.show()
    except ImportError:
        def show_img(img):
            img.show()

    anims = _read_hdr()
    frame = anims.get_frame(set_num, anim_num, frame_num)
    if frame:
        show_img(frame[1])

def show_anim(set_num, anim_num):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import ArtistAnimation

    anims = _read_hdr()
    s = anims.sets[set_num]
    anim = s.animations[anim_num]

    frameinfo_l = [frame.header for frame in anim.frames]
    images = [
        anims.render_pixelmap(anims.make_pixelmap(frame.data, frame.header["imageoffset"]))
        for frame in anim.frames
    ]
    fps = anim.fps

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
    for s in anims.sets:
        for anim in s.animations:
            for frame in anim.frames:
                anims.make_pixelmap(frame.data, frame.header["imageoffset"])

def profile_stress_test(arg):
    '''
    Run stress_test() repeatedly until condition specified by `arg` is satisfied.
    `arg` must be a string of one of the following types:
     - "#x", where # is an integer; this calls the function # times
     - "#s", where # is an integer; this calls the function until # seconds are elapsed
    This function is useful for profiling from the command line with a command such as:
    > python -m cProfile run_test.py profile_stress_test <arg>
    Optionally you can add `-o <file>` after "cProfile" to save the results to a file.
    '''
    from time import time
    startingtime = time()
    if arg[-1] == "x":
        for i in range(1, int(arg[:-1])+1):
            stress_test()
        curtime = time()
    elif arg[-1] == "s":
        i = 0
        seconds = float(arg[:-1])
        curtime = startingtime
        while curtime - startingtime <= seconds:
            stress_test()
            curtime = time()
            i += 1
    else:
        raise KeyError
    print("Running for {:.3} s, {} iterations".format(curtime-startingtime, i))

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
