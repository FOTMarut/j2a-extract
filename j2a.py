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
    _animheaderstruct = "s4|signature/B|animcount/B|samplecount/h|framecount/l|priorsamplecount/l|c1/l|u1/l|c2/l|u2/l|c3/l|u3/l|c4/l|u4"
    _animinfostruct = "H|framecount/H|fps/l|reserved"
    _frameinfostruct = "H|width/H|height/h|coldspotx/h|coldspoty/h|hotspotx/h|hotspoty/h|gunspotx/h|gunspoty/L|imageoffset/L|maskoffset"
    _ALIBheadersize = 28
    _ANIMheadersize = 44

    class J2AParseError(Exception):
        pass

    def __init__(self, filename):
        ''' initializes class, sets file name '''
        self.header = self.setdata = self.setoffsets = self.palette = None
        self.currentset = -1
        self.set_filename(filename)

    def set_filename(self, filename):
        self.filename = filename

    def get_substream(self, streamnum, setnum=None):
        if setnum is None:
            setnum = self.currentset

        data = self.setdata[setnum]
        suboffset = sum(data["c" + str(i)] for i in range(1, streamnum))

        chunk = self.setchunks[setnum][suboffset:suboffset+data["c" + str(streamnum)]]
        return zlib.decompressobj().decompress(chunk, data["u" + str(streamnum)])

    def read(self):
        ''' reads whole J2A file, parses ALIB and ANIM headers and collects all sets '''
        if not self.header:
            with open(self.filename, "rb") as j2afile:
                # TODO: maybe add a separate check for ALIB version?
                try:
                    self.header = misc.named_unpack(self._headerstruct, j2afile.read(self._ALIBheadersize))
                    setcount = self.header["setcount"]
                    assert(
                        (self.header["signature"], self.header["magic"], self.header["headersize"], self.header["version"]) ==
                        (b'ALIB', 0x00BEBA00, self._ALIBheadersize + 4*setcount, 0x0200)
                    )
                    if self.header["unknown"] != 0x1808:
                        print("Warning: minor difference found in ALIB header. Ignoring...", file=sys.stderr)
                    raw = j2afile.read(4*setcount)
                    self.setoffsets = struct.unpack('<%iL' % setcount, raw)
                    crc = zlib.crc32(raw)
                    assert(self.setoffsets[0] == self.header["headersize"])
                    self.setdata = []
                    self.setchunks = []
                    alloffsets = self.setoffsets + (self.header["filesize"],)
                    setsizes = (b-a for a,b in pairwise(alloffsets))
                    for size in setsizes:
                        assert(size >= self._ANIMheadersize)
                        raw = j2afile.read(size)
                        crc = zlib.crc32(raw, crc)
                        assert(len(raw) == size)
                        setheader = misc.named_unpack(self._animheaderstruct, raw[:self._ANIMheadersize])
                        assert(
                            (setheader["signature"], setheader["u1"], setheader["u2"]) ==
                            (b'ANIM', 8*setheader["animcount"], 24*setheader["framecount"])
                        )
                        assert(self._ANIMheadersize + setheader["c1"] + setheader["c2"] + setheader["c3"] + setheader["c4"] == size)
                        self.setdata.append(setheader)
                        self.setchunks.append(raw[self._ANIMheadersize:])
                    if crc & 0xffffffff != self.header["crc32"]:
                        print("Warning: CRC32 mismatch in J2A file %s. Ignoring..." % self.filename, file=sys.stderr)
                    raw = j2afile.read()
                    if raw:
                        print("Warning: extra %i bytes found at the end of J2A file %s. Ignoring..." % (len(raw), self.filename), file=sys.stderr)
                except (AssertionError, struct.error):
                    raise J2A.J2AParseError("Error: file %s is not a valid J2A file" % self.filename)

        return self.header

    def load_set(self, setnum):
        if not self.header:
            self.read()

        if -1 < setnum < self.header["setcount"]:
            self.currentset = setnum
        else:
            print("set %s doesn't exist!" % setnum, file=sys.stderr)
            sys.exit(1)

    def get_palette(self, given = None):
        if not self.palette:
            palfile = open("Diamondus_2.pal").readlines() if not given else given
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

        self.load_set(set_num)
        data = self.setdata[set_num]
        animinfo = self.get_substream(1)
        frameinfo = self.get_substream(2)
        frameoffset = frame_num
        for i in range(0, anim_num):
            try:
                info = misc.named_unpack(self._animinfostruct, animinfo[i*8:(i*8)+8])
            except:
                print("couldnt load frame at coordinates %s" % repr((set_num, anim_num, frame_num)))
                return
            frameoffset += info["framecount"]
        info = misc.named_unpack(self._frameinfostruct, frameinfo[frameoffset*24:(frameoffset*24)+24])
        dataoffset = info["imageoffset"]
        imagedata = self.get_substream(3)

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
