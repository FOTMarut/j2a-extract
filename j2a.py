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
    _headerstruct = "s4|signature/L|magic/L|headersize/h|version/h|unknown/L|filesize/L|crc32/L|setcount"
    _animinfostruct = "H|framecount/H|fps/l|reserved"
    _frameinfostruct = "H|width/H|height/h|coldspotx/h|coldspoty/h|hotspotx/h|hotspoty/h|gunspotx/h|gunspoty/L|imageoffset/L|maskoffset"
    _ALIBheadersize = 28
    _defaultpalette = "Diamondus_2.pal"

    class J2AParseError(Exception):
        pass

    class Set(object):
        _animheaderstruct = "s4|signature/B|animcount/B|samplecount/h|framecount/l|priorsamplecount/l|c1/l|u1/l|c2/l|u2/l|c3/l|u3/l|c4/l|u4"
        _ANIMheadersize = 44

        def __init__(self, *pargs, **kwargs):
            if pargs:
                self.header, self.chunks = pargs
            else:
                self.header = {(a+b):0 for a in "cu" for b in "1234"}
                self.header.update(
                    animcount=0,
                    samplecount=0,
                    framecount=0,
                    priorsamplecount=kwargs["prevsamplecount"]
                )
                self.chunks = [zlib.compress(b'')] * 4

        @staticmethod
        def read(f, crc):
            chunk = f.read(J2A.Set._ANIMheadersize)
            crc = zlib.crc32(chunk, crc)
            setheader = misc.named_unpack(J2A.Set._animheaderstruct, chunk)
            assert(
                (setheader["signature"], setheader["u1"], setheader["u2"]) ==
                (b'ANIM', 8*setheader["animcount"], 24*setheader["framecount"])
            )
            setheader.pop("signature")
            chunks = [f.read(setheader["c" + k]) for k in "1234"]
            for chunk in chunks:
                crc = zlib.crc32(chunk, crc)
            return (J2A.Set(setheader, chunks), crc)

        def get_substream(self, streamnum):
            return zlib.decompress(self.chunks[streamnum-1])

    def __init__(self, filename):
        ''' initializes class, sets file name '''
        self.header = self.palette = None
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
        if not self.header:
            with open(self.filename, "rb") as j2afile:
                # TODO: maybe add a separate check for ALIB version?
                # TODO: check for previous sample count consistency
                try:
                    self.header = misc.named_unpack(self._headerstruct, j2afile.read(self._ALIBheadersize))
                    setcount = self.header["setcount"]
                    assert(
                        (self.header["signature"], self.header["magic"], self.header["headersize"], self.header["version"]) ==
                        (b'ALIB', 0x00BEBA00, self._ALIBheadersize + 4*setcount, 0x0200)
                    )
                    if self.header["unknown"] != 0x1808:
                        print("Warning: minor difference found in ALIB header. Ignoring...", file=sys.stderr)
                    self.header.pop("signature")
                    raw = j2afile.read(4*setcount)
                    setoffsets = struct.unpack('<%iL' % setcount, raw)
                    crc = zlib.crc32(raw)
                    assert(setoffsets[0] == self.header["headersize"])
                    prevsamplecount = ps_miscounts = 0
                    self.sets = []
                    for offset in setoffsets:
                        # The shareware demo removes some of the animsets to save on filesize, but leaves the
                        # order of animations intact, causing gaping holes with offsets of zero in the .j2a file
                        if offset == 0:
                            self.sets.append(J2A.Set(prevsamplecount=prevsamplecount))
                        else:
                            J2A._seek(j2afile, offset)
                            s, crc = J2A.Set.read(j2afile, crc)
                            if prevsamplecount != s.header["priorsamplecount"]:
                                ps_miscounts += 1
                            prevsamplecount = s.header["samplecount"] + s.header["priorsamplecount"]
                            self.sets.append(s)
                    if ps_miscounts:
                        print("Warning: %d sample miscounts detected (this is expected for the shareware demo)" % ps_miscounts)
                    if crc & 0xffffffff != self.header["crc32"]:
                        print("Warning: CRC32 mismatch in J2A file %s. Ignoring..." % self.filename, file=sys.stderr)
                    raw = j2afile.read()
                    if raw:
                        print("Warning: extra %d bytes found at the end of J2A file %s. Ignoring..." % (len(raw), self.filename), file=sys.stderr)
                except (AssertionError, struct.error):
                    raise J2A.J2AParseError("File %s is not a valid J2A file" % self.filename)

        return self.header

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

    def make_pixelmap(self, raw):
        width, height = struct.unpack_from("<HH", raw)
        width &= 0x7FFF #unset msb
        raw = raw[4:]
        #prepare pixmap
        pixmap = [[0]*width for _ in range(height)]
        #fill it with data! (image format parser)
        length = len(raw)
        x = y = i = 0
        # This loop fails silently if decoding would cause OOB exceptions
        while i < length:
            byte = struct.unpack_from("<B", raw, i)[0]
#             print("Byte: {:3}, {}".format(byte, ('<' if byte < 128 else '=' if byte == 128 else '>'))) # TODO: delme
            if byte > 128:
                byte -= 128
                l = min(byte, width - x)
                pixmap[y][x:x+l] = struct.unpack_from("<%iB" % l, raw, i+1)
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
        if not self.header:
            self.read()

        s = self.sets[set_num]
        if anim_num >= s.header["animcount"]:
            raise KeyError("Animation number %d is out of bounds for set %d (must be between 0 and %d)"
                % (anim_num, set_num, s.header["animcount"]-1))
        animinfo = s.get_substream(1)
        frameinfo = s.get_substream(2)
        frameoffset = frame_num
        for i in range(0, anim_num):
            try:
                info = misc.named_unpack(self._animinfostruct, animinfo[i*8:(i*8)+8])
            except:
                print("couldn't load frame at coordinates %s" % repr((set_num, anim_num, frame_num)))
                return
            frameoffset += info["framecount"]
        info = misc.named_unpack(self._frameinfostruct, frameinfo[frameoffset*24:(frameoffset*24)+24])
        dataoffset = info["imageoffset"]
        imagedata = s.get_substream(3)

        pixelmap = self.make_pixelmap(imagedata[dataoffset:])
        return [info, self.render_pixelmap(pixelmap)]

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
