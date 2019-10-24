# python j2a parser
# by stijn, edited by violet clm
# thanks to neobeo/j2nsm for the file format specs
# see http://www.jazz2online.com

from __future__ import print_function
import struct
import os
import zlib
#needs python image library, http://www.pythonware.com/library/pil/
from PIL import Image, ImageDraw

import misc

class J2A:
    _headerstruct = "s4|signature/L|magic/L|headersize/h|version/h|unknown/L|filesize/L|crc32/L|setcount"
    _animheaderstruct = "s4|signature/B|animcount/B|samplecount/h|framecount/l|priorsamplecount/l|c1/l|u1/l|c2/l|u2/l|c3/l|u3/l|c4/l|u4"
    _animinfostruct = "H|framecount/H|fps/l|reserved"
    _frameinfostruct = "H|width/H|height/h|coldspotx/h|coldspoty/h|hotspotx/h|hotspoty/h|gunspotx/h|gunspoty/L|imageoffset/L|maskoffset"
    _headersize = 28
    header = setdata = setoffsets = palette = None
    currentset = -1

    def __init__(self, filename):
        ''' loads file contents into memory '''
        try:
            j2afile = open(filename, "rb")
        except:
            print("file %s could not be read!" % filename, file=os.sys.stderr)
            os.sys.exit(1)
            
        self.filesize = os.path.getsize(filename)
        self.j2afile = j2afile.read(self.filesize)
        
    def get_substream(self, streamnum):
        offset = self.setoffsets[self.currentset]
        data = self.setdata[self.currentset]
        suboffset = offset + 44
        for i in range(1, streamnum):
            suboffset += data["c" + str(i)]

        chunk = self.j2afile[suboffset:suboffset+data["c" + str(streamnum)]]
        return zlib.decompressobj().decompress(chunk, data["u" + str(streamnum)])
    
    def read_header(self):
        ''' reads and parses header, setdata and setoffsets '''
        if not self.header:
            self.header = misc.named_unpack(self._headerstruct, self.j2afile[:self._headersize])
            setcount = self.header["setcount"]
            self.setoffsets = struct.unpack_from('<%iL' % setcount, self.j2afile, self._headersize)
            self.setdata = tuple(misc.named_unpack(self._animheaderstruct, self.j2afile[offset:offset+44]) for offset in self.setoffsets)
            
        return self.header
        
    def load_set(self, setnum):
        if not self.header:
            self.read_header()
        
        if -1 < setnum < self.header["setcount"]:
            self.currentset = setnum
        else:
            print("set %s doesn't exist!" % setnum, file=os.sys.stderr)
            os.sys.exit(1)
        
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
            self.read_header()
            
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
    j2a = J2A(filename)
    j2a.render_frame(9, 0, 6)

if __name__ == "__main__":
    main()
