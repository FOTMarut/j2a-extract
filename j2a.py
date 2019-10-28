# python j2a parser
# by stijn, edited by violet clm
# thanks to neobeo/j2nsm for the file format specs
# see http://www.jazz2online.com

from __future__ import print_function
import itertools
import struct
import os
import sys
import zlib
import array
#needs python image library, http://www.pythonware.com/library/pil/
from PIL import Image, ImageDraw

import misc

# From the official Python docs
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class J2A:
    _Header = misc.NamedStruct("4s|signature/L|magic/L|headersize/h|version/h|unknown/L|filesize/L|crc32/L|setcount")
    _defaultpalette = "Diamondus_2.pal"

    class J2AParseError(Exception):
        pass

    class Set(object):
        _Header = misc.NamedStruct("4s|signature/B|animcount/B|samplecount/h|framecount/l|priorsamplecount/l|c1/l|u1/l|c2/l|u2/l|c3/l|u3/l|c4/l|u4")

        def __init__(self, *pargs, **kwargs):
            if pargs:
                setheader, self._chunks = pargs
                self._samplecount = setheader["samplecount"]
            else:
                self._anims = []
                self._samplecount = 0
                self._chunks = None

        @staticmethod
        def read(f, crc):
            chunk = f.read(J2A.Set._Header.size)
            crc = zlib.crc32(chunk, crc)
            setheader = J2A.Set._Header.unpack(chunk)
            assert(
                (setheader["signature"], setheader["u1"], setheader["u2"]) ==
                (b'ANIM', 8*setheader["animcount"], 24*setheader["framecount"])
            )
            chunks = [(f.read(setheader["c" + k]), setheader["u" + k]) for k in "1234"]
            for chunk in chunks:
                crc = zlib.crc32(chunk[0], crc)
            return (J2A.Set(setheader, chunks), crc, setheader["priorsamplecount"])

        def unpack(self):
            if self._chunks:
                animinfo, frameinfo, imagedata, self._sampledata = \
                    (zlib.decompress(c, zlib.MAX_WBITS, usize) for c,usize in self._chunks)
                animinfo  = list(J2A.Animation._Header.iter_unpack(animinfo))
                frameinfo = list(J2A.Frame._Header.iter_unpack(frameinfo))
                imagedata = array.array("B", imagedata)
                self._anims = []
                for anim in animinfo:
                    framecount = anim["framecount"]
                    self._anims.append(J2A.Animation.read(anim, frameinfo[:framecount], imagedata))
                    frameinfo = frameinfo[framecount:]
                self._chunks = None
            return self

        def pack(self):
            raise NotImplementedError #TODO

        @property
        def animations(self):
            if self._chunks:
                self.unpack()
            return self._anims

        @property
        def samples(self):
            raise NotImplementedError #TODO


    class Animation:
        _Header = misc.NamedStruct("H|framecount/H|fps/l|reserved")

        def __init__(self, frames, fps):
            self.frames = frames
            self.fps = fps

        @staticmethod
        def read(animinfo, frameinfo_l, imagedata):
            return J2A.Animation(
                [J2A.Frame.read(frameinfo, imagedata) for frameinfo in frameinfo_l],
                animinfo["fps"]
            )


    class Frame:
        _Header = misc.NamedStruct("H|width/H|height/h|coldspotx/h|coldspoty/h|hotspotx/h|hotspoty/h|gunspotx/h|gunspoty/L|imageoffset/L|maskoffset")

        def __init__(self, frameinfo, imagedata):
            self.header = frameinfo
            self.data = imagedata

        @staticmethod
        def read(frameinfo, imagedata):
            return J2A.Frame(frameinfo, imagedata)


    def __init__(self, filename):
        ''' initializes class, sets file name '''
        self.palette = None
        self.sets = []
        self.set_filename(filename)

    def set_filename(self, filename):
        self.filename = filename

    @staticmethod
    def _seek(f, newpos):
        delta = newpos - f.tell()
        if delta > 0:
            print("Warning: skipping over %d bytes" % delta)
            b = f.read(delta)
            assert(len(b) == delta)
        elif delta < 0:
            raise J2AParseError("File is not a valid J2A file (overlapping sets)")

    def read(self):
        ''' reads whole J2A file, parses ALIB and ANIM headers and collects all sets '''
        with open(self.filename, "rb") as j2afile:
            # TODO: maybe add a separate check for ALIB version?
            try:
                alibheader = self._Header.unpack(j2afile.read(self._Header.size))
                setcount = alibheader["setcount"]
                assert(
                    (alibheader["signature"], alibheader["magic"], alibheader["headersize"], alibheader["version"]) ==
                    (b'ALIB', 0x00BEBA00, self._Header.size + 4*setcount, 0x0200)
                )
                if alibheader["unknown"] != 0x1808:
                    print("Warning: minor difference found in ALIB header. Ignoring...", file=sys.stderr)
                alibheader.pop("signature")
                raw = j2afile.read(4*setcount)
                setoffsets = struct.unpack('<%iL' % setcount, raw)
                crc = zlib.crc32(raw)
                assert(setoffsets[0] == alibheader["headersize"])
                prevsamplecount = ps_miscounts = 0
                self.sets = []
                for offset in setoffsets:
                    # The shareware demo removes some of the animsets to save on filesize, but leaves the
                    # order of animations intact, causing gaping holes with offsets of zero in the .j2a file
                    if offset == 0:
                        self.sets.append(J2A.Set(prevsamplecount=prevsamplecount))
                    else:
                        J2A._seek(j2afile, offset)
                        s, crc, reported_psc = J2A.Set.read(j2afile, crc)
                        if prevsamplecount != reported_psc:
                            ps_miscounts += 1
                        prevsamplecount = s._samplecount + reported_psc
                        self.sets.append(s)
                if ps_miscounts:
                    print("Warning: %d sample miscounts detected (this is expected for the shareware demo)" % ps_miscounts)
                if crc & 0xffffffff != alibheader["crc32"]:
                    print("Warning: CRC32 mismatch in J2A file %s. Ignoring..." % self.filename, file=sys.stderr)
                raw = j2afile.read()
                if raw:
                    print("Warning: extra %d bytes found at the end of J2A file %s. Ignoring..." % (len(raw), self.filename), file=sys.stderr)
            except (AssertionError, struct.error):
                raise J2A.J2AParseError("File %s is not a valid J2A file" % self.filename)

        return self

    def unpack(self):
        for s in self.sets:
            s.unpack()

    def get_palette(self, given = None):
        if not self.palette:
            palfile = open(self._defaultpalette).readlines() if not given else given
            pal = list()
            for i in range(3, 259):
                color = palfile[i].rstrip("\n").split(' ')
                pal.append((int(color[0]), int(color[1]), int(color[2])))
            self.palette = pal
            self.palettesequence = [band for color in pal for band in color]

        return self.palette

    @staticmethod
    def make_pixelmap(raw, offset=0):
        width, height = struct.unpack_from("<HH", raw, offset)
        width &= 0x7FFF #unset msb
        #prepare pixmap
        pixmap = [[0]*width for _ in range(height)]
        #fill it with data! (image format parser)
        length = len(raw)
        x = y = 0
        i = offset + 4
        # This loop fails silently if decoding would cause OOB exceptions
        while i < length:
            byte = raw[i]
            if byte > 128:
                byte -= 128
                l = min(byte, width - x)
                pixmap[y][x:x+l] = raw[i+1:i+1+l]
                x += byte
                i += byte
            elif byte < 128:
                x += byte
            else:
                x = 0
                y += 1
                if y >= height:
                    break
            i += 1
        return pixmap

    def render_pixelmap(self, pixelmap):
        width, height = (len(pixelmap[0]), len(pixelmap))
        img = Image.new("RGBA", (width, height))
        im = img.load()
        pal = self.get_palette()

        for x, row in enumerate(pixelmap):
            for y, index in enumerate(row):
                if index > 1:
                    im[y, x] = pal[index]

        return img

    def render_paletted_pixelmap(self, pixelmap):
        width, height = (len(pixelmap[0]), len(pixelmap))
        img = Image.new("P", (width, height))
        img.putdata([pixel for col in pixelmap for pixel in col])
        self.get_palette()
        img.putpalette(self.palettesequence)
        return img

    def get_frame(self, set_num, anim_num, frame_num):
        ''' gets image info and image corresponding to a specific set, animation and frame number '''
        s = self.sets[set_num]
        anim = s.animations[anim_num]
        frame = anim.frames[frame_num]

        pixelmap = self.make_pixelmap(frame.data, frame.header["imageoffset"])
        return [frame.header, self.render_pixelmap(pixelmap)]

    def render_frame(self, *coordinates):
        self.get_frame(*coordinates)[1].save("preview.png", "PNG")

def main():
    from sys import argv
    filename = argv[1] if (len(argv) >= 2) else "C:\Games\Jazz2\Anims.j2a"
    try:
        j2a = J2A(filename)
    except IOError:
        print("File %s could not be read!" % filename, file=sys.stderr)
        return 1

    j2a.render_frame(9, 0, 6)

if __name__ == "__main__":
    sys.exit(main())
