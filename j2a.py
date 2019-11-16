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
from PIL import Image

import misc

if sys.version_info[0] < 3:
    zip = itertools.izip

# From the official Python docs
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

# From the official Python docs
def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEF', 3) --> ABC DEF
    args = [iter(iterable)] * n
    return zip(*args)

# From the official Python docs
def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(itertools.islice(iterable, n))

def raising_function(string, exception):
    raise exception(string)


# TODO: add __slots__ definitions
# TODO: use bytearray instead of array.array
# TODO: add raise from clauses
class J2A:
    _Header = misc.NamedStruct("4s|signature/L|magic/L|headersize/h|version/h|unknown/L|filesize/L|crc32/L|setcount")
    _defaultconfig = {"palette": "Diamondus_2.pal", "compress_method": 9, "null_image": "error", "null_mask": "ignore",
        "fake_size_and_crc": None, "empty_set": None}
    _error_action = {
        "ignore":  (lambda s, c: None),
        "warning": (lambda s, c: print("Warning:", s, file=sys.stderr)),
        "error":   raising_function
    }

    class J2AParsingError(Exception):
        pass

    class J2APackingError(Exception):
        pass

    class Set(object):
        _Header = misc.NamedStruct("4s|signature/B|animcount/B|samplecount/h|framecount/l|priorsamplecount/l|c1/l|u1/l|c2/l|u2/l|c3/l|u3/l|c4/l|u4")

        def __init__(self, *pargs, **kwargs):
            if pargs:
                setheader, self._chunks = pargs
                self._samplecount = setheader["samplecount"]
                self.samplesbaseindex = setheader["priorsamplecount"]
            else:
                self._anims = []
                self._samples = []
                self._chunks = None
                if "samplesbaseindex" in kwargs:
                    self.samplesbaseindex = kwargs["samplesbaseindex"]

        def is_empty(self):
            if self._chunks:
                return 0 == self._chunks[0][1] + self._chunks[3][1]
            else:
                return 0 == len(self._anims) + len(self._samples)

        @staticmethod
        def read(f, crc):
            chunk = f.read(J2A.Set._Header.size)
            crc = zlib.crc32(chunk, crc)
            setheader = J2A.Set._Header.unpack(chunk)
            assert(
                (setheader["signature"], setheader["u1"], setheader["u2"]) ==
                (b'ANIM', J2A.Animation._Header.size * setheader["animcount"], J2A.Frame._Header.size * setheader["framecount"])
            )
            chunks = [(f.read(setheader["c" + k]), setheader["u" + k]) for k in "1234"]
            for chunk in chunks:
                crc = zlib.crc32(chunk[0], crc)
            return (J2A.Set(setheader, chunks), crc, setheader["priorsamplecount"])

        def unpack(self):
            if self._chunks:
                animinfo, frameinfo, imagedata, sampledata = \
                    (zlib.decompress(c, zlib.MAX_WBITS, usize) for c,usize in self._chunks)

                animinfo  = (J2A.Animation._Header.iter_unpack(animinfo))
                frameinfo = list(J2A.Frame._Header.iter_unpack(frameinfo))

                offsets = sorted((info[key], 2*i+j) for i, info in enumerate(frameinfo) for j, key in enumerate(("imageoffset", "maskoffset")))
                offsets.append((len(imagedata), 2*len(frameinfo)))
                if take(1, (o for o,i in offsets if o != -1 and o & 3)):
                    print("Warning: unaligned offset found while unpacking", file=sys.stderr)
                data = sorted((i1, None) if o1 == -1 else (i1, imagedata[o1:o2]) for (o1, i1), (o2, i2) in pairwise(offsets))
                frames = (J2A.Frame.read(info, img, mask) for ((i1, img), (i2, mask)), info in zip(grouper(data, 2), frameinfo))
                self._anims = [J2A.Animation(frames=take(info["framecount"], frames), fps=info["fps"]) for info in animinfo]

                self._samples = []
                offset, length = 0, len(sampledata)
                while offset < length:
                    size = struct.unpack_from("<L", sampledata, offset)[0]
                    self._samples.append(sampledata[offset:offset+size])
                    offset += size
                if len(self._samples) != self._samplecount:
                    print("Warning: internal sample miscount (expected: %d, got: %d)" % (self._samplecount, len(self._samples)), file=sys.stderr)

                self._chunks = None
                del self._samplecount
            return self

        @staticmethod
        def _compress(animinfo, frameinfo, imagedata, sampledata, config):
            compress_method = config["compress_method"]

            if isinstance(compress_method, int):
                return [(zlib.compress(c, compress_method), len(c)) for c in (animinfo, frameinfo, imagedata, sampledata)]

            def compress_ext(raw, *pargs):
                c_obj = zlib.compressobj(*pargs)
                return c_obj.compress(raw) + c_obj.flush()

            if isinstance(compress_method, tuple):
                return [(compress_ext(c, *compress_method), len(c)) for c in (animinfo, frameinfo, imagedata, sampledata)]
            elif compress_method == "fastest_model":
                return [(compress_ext(c, *method), len(c)) for c, method in (
                    (animinfo,   (9, zlib.DEFLATED, zlib.MAX_WBITS, 9)),
                    (frameinfo,  (9, zlib.DEFLATED, zlib.MAX_WBITS, 9)),
                    (imagedata,  (9, zlib.DEFLATED, zlib.MAX_WBITS, 5)),
                    (sampledata, (9, zlib.DEFLATED, zlib.MAX_WBITS, 6)),
                )]
            else:
                raise ValueError("Invalid compress_method specified")

        def serialize(self, config):
            if config["empty_set"] == "crop" and self.is_empty():
                return b''

            self.pack(config)

            try:
                samplesbaseindex = self.samplesbaseindex
            except AttributeError:
                if self._samplecount == 0:
                    samplesbaseindex = 0
                else:
                    raise J2APackingError("'samplesbaseindex' member must be set when packing a set with samples")

            setheader = {
                "signature": b'ANIM',
                "animcount": self._chunks[0][1] // J2A.Animation._Header.size,
                "samplecount": self._samplecount,
                "framecount": self._chunks[1][1] // J2A.Frame._Header.size,
                "priorsamplecount": samplesbaseindex
            }
            for k, (chunk, usize) in zip("1234", self._chunks):
                setheader["c" + k] = len(chunk)
                setheader["u" + k] = usize

            return b''.join( [J2A.Set._Header.pack(**setheader)] + [c[0] for c in self._chunks] )

        def pack(self, config):
            if not self._chunks:
                animinfo = J2A.Animation._Header.iter_pack(
                    {"framecount": len(a.frames), "fps": a.fps, "reserved": 0} for a in self._anims
                )
                l_frameinfo = []
                img_data, mask_data = b'', b''
                null_pixmaps = null_masks = 0
                for an,a in enumerate(self._anims):
                    for fn,f in enumerate(a.frames):
                        f.encode_image()
                        no_pixmap, no_mask = (f._rle_encoded_pixmap is None, f.mask is None)
                        l_frameinfo.append(f._get_header(
                            -1 if no_pixmap else len(img_data),
                            -1 if no_mask   else len(mask_data)
                        ))
                        width, height = f.shape
                        img_data += struct.pack("<HH", (width | 0x8000 if f.tagged else width), height)

                        if no_pixmap:
                            null_pixmaps += 1
                        else: # Image and mask data should be aligned to a 4-byte boundary
                            img_data += f._rle_encoded_pixmap + b'\x00' * (-len(f._rle_encoded_pixmap) & 3)

                        if no_mask:
                            null_masks += 1
                        else:
                            mask_data += f.mask + b'\x00' * (-len(f.mask) & 3)

                if null_pixmaps > 0:
                    J2A._error_action[config["null_image"]]("found %d frames with null images" % null_pixmaps, J2A.J2APackingError)
                if null_masks > 0:
                    J2A._error_action[config["null_mask" ]]("found %d frames with null masks"  % null_masks  , J2A.J2APackingError)

                img_length = len(img_data)
                for frame_info in l_frameinfo:
                    maskoffset = frame_info["maskoffset"]
                    frame_info["maskoffset"] = maskoffset + img_length if maskoffset != -1 else -1
                frameinfo = J2A.Frame._Header.iter_pack(l_frameinfo)
                sampledata = b''.join(self._samples)
                self._samplecount = len(self._samples)

                self._chunks = J2A.Set._compress(animinfo, frameinfo, img_data + mask_data, sampledata, config)
                del self._anims, self._samples
            return self

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

        def __init__(self, frames=None, fps=10):
            if frames is None:
                frames = []
            self.frames, self.fps = frames, fps


    class Frame:
        _Header = misc.NamedStruct("H|width/H|height/h|coldspotx/h|coldspoty/h|hotspotx/h|hotspoty/h|gunspotx/h|gunspoty/l|imageoffset/l|maskoffset")

        def __init__(self, shape=None, origin=None, coldspot=None, gunspot=None, pixmap=None, mask=None, rle_encoded_pixmap=None, tagged=False):
            assert((pixmap is None) ^ (rle_encoded_pixmap is None))
            self.shape, self.origin, self.coldspot, self.gunspot, self.mask, self.tagged = shape, origin, coldspot, gunspot, mask, tagged
            if not rle_encoded_pixmap is None:
                assert(not shape is None)
                self._rle_encoded_pixmap = rle_encoded_pixmap
            elif isinstance(pixmap, Image.Image):
                assert(shape is None or shape == pixmap.size)
                self.shape = pixmap.size
                width, height = pixmap.size
                self._pixmap = [bytearray(row) for row in grouper(pixmap.tobytes(), width)]
                assert(len(self._pixmap) == height)
            else:
                self._pixmap = pixmap

        def _get_header(self, img_offset, mask_offset):
            return dict((k, v) for k, v in zip(
                J2A.Frame._Header._names,
                self.shape + self.coldspot + self.origin + self.gunspot + (img_offset, mask_offset)
            ))

        @staticmethod
        def read(frameinfo, imagedata, maskdata):
            width, height = struct.unpack_from("<HH", imagedata)
            tagged = bool(width & 0x8000)
            width &= 0x7FFF
            assert(width == frameinfo["width"] and height == frameinfo["height"])
            return J2A.Frame(
                shape = (width, height),
                origin = (frameinfo["hotspotx"], frameinfo["hotspoty"]),
                coldspot = (frameinfo["coldspotx"], frameinfo["coldspoty"]),
                gunspot = (frameinfo["gunspotx"], frameinfo["gunspoty"]),
                rle_encoded_pixmap = imagedata[4:],
                mask = maskdata,
                tagged = tagged
            )

        # TODO: need to stress test these two methods
        def decode_image(self):
            if not hasattr(self, "_pixmap"):
                width, height = self.shape
                raw = array.array("B", self._rle_encoded_pixmap)
                #prepare pixmap
                pixmap = [[0]*width for _ in range(height)]
                #fill it with data! (image format parser)
                length = len(raw)
                x = y = i = 0
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
                self._pixmap = pixmap
                del self._rle_encoded_pixmap
            return self

        def encode_image(self):
            if not hasattr(self, "_rle_encoded_pixmap"):
                encoded = array.array("B")
                for row in self._pixmap:
                    while True:
                        row = bytes(row)
                        length = len(row)
                        row = row.lstrip(b'\x00')
                        if not row:
                            break
                        length -= len(row)
                        while length:
                            m = min(length, 0x7f)
                            encoded.append(m)
                            length -= m
                        length = row.find(b'\x00')
                        if length == -1:
                            length = len(row)
                        while length:
                            m = min(length, 0x7f)
                            encoded.append(m ^ 0x80)
                            encoded += array.array("B", row[:m])
                            row = row[m:]
                            length -= m
                    encoded.append(0x80)
                self._rle_encoded_pixmap = bytes(encoded)
                del self._pixmap
            return self

        def autogenerate_mask(self):
            self.decode_image()
            mask = bytearray(b'\x00') * ((self.shape[0] * self.shape[1] + 7) // 8)
            pix_iter = itertools.chain(*self._pixmap)
            for i in range(len(mask)):
                mask[i] = sum(bool(pix) << j for j, pix in enumerate(take(8, pix_iter)))
            self.mask = bytes(mask)
            return self

    def __init__(self, filename=None, **kwargs):
        ''' initializes class, sets file name '''
        self.palette = None
        self.sets = []
        self.set_filename(filename)
        self.config = J2A._defaultconfig.copy()
        self.config.update(kwargs)

    def set_filename(self, filename):
        self.filename = filename
        return self

    @staticmethod
    def _seek(f, newpos):
        delta = newpos - f.tell()
        if delta > 0:
            print("Warning: skipping over %d bytes" % delta, file=sys.stderr)
            b = f.read(delta)
            assert(len(b) == delta)
        elif delta < 0:
            raise J2AParsingError("File is not a valid J2A file (overlapping sets)")

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
                raw = j2afile.read(4*setcount)
                setoffsets = struct.unpack('<%dL' % setcount, raw)
                crc = zlib.crc32(raw)
                assert(setoffsets[0] == alibheader["headersize"])
                prevsamplecount = ps_miscounts = 0
                self.sets = []
                for offset in setoffsets:
                    # The shareware demo removes some of the animsets to save on filesize, but leaves the
                    # order of animations intact, causing gaping holes with offsets of zero in the .j2a file
                    if offset == 0:
                        self.sets.append(J2A.Set(samplesbaseindex=prevsamplecount))
                    else:
                        J2A._seek(j2afile, offset)
                        s, crc, reported_psc = J2A.Set.read(j2afile, crc)
                        if prevsamplecount != reported_psc:
                            ps_miscounts += 1
                        prevsamplecount = s._samplecount + reported_psc
                        self.sets.append(s)
                if ps_miscounts:  # TODO: remove this check entirely, it's bogus
                    print("Warning: %d sample miscounts detected (this is expected for the shareware demo)" % ps_miscounts, file=sys.stderr)
                raw = j2afile.read()
                if raw:
                    print("Warning: extra %d bytes found at the end of J2A file %s. Ignoring..." % (len(raw), self.filename), file=sys.stderr)
                    crc = zlib.crc32(raw, crc)
                if crc & 0xffffffff != alibheader["crc32"]:
                    print("Warning: CRC32 mismatch in J2A file %s. Ignoring..." % self.filename, file=sys.stderr)
            except (AssertionError, struct.error):
                raise J2A.J2AParsingError("File %s is not a valid J2A file" % self.filename)

        return self

    def unpack(self):
        for s in self.sets:
            s.unpack()
        return self

    def write(self, filename=None):
        if filename is None:
            filename = self.filename
            if filename is None:
                raise J2APackingError("no filename specified")
        self.pack()
        setcount = len(self.sets)
        set_data = [s.serialize(self.config) for s in self.sets]
        set_offsets = []
        cur_offset = headersize = J2A._Header.size + 4 * setcount
        for sdata in set_data:
            set_length = len(sdata)
            set_offsets.append(cur_offset if set_length > 0 else 0)
            cur_offset += set_length
        set_offsets_raw = struct.pack("<%dL" % setcount, *set_offsets)
        crc = zlib.crc32(set_offsets_raw)
        for sdata in set_data:
            crc = zlib.crc32(sdata, crc)
        crc &= 0xffffffff

        extra_data = b''
        if not self.config["fake_size_and_crc"] is None:
            target_filesize, target_crc = self.config["fake_size_and_crc"]
            target_crc &= 0xffffffff
            if (cur_offset, crc) != (target_filesize, target_crc):
                if target_filesize - cur_offset < 4:
                    raise J2APackingError("Can't fake filesize and CRC32, obtained size is too large")
                extra_data = b'\x00' * (target_filesize - cur_offset - 4)
                crc = zlib.crc32(extra_data, crc)
                salt = misc.fake_crc(target_crc)
                salt = struct.pack("<L", crc ^ salt)
                extra_data += salt
                crc = zlib.crc32(salt, crc)
                cur_offset += len(extra_data)
                assert(crc == target_crc and cur_offset == target_filesize)

        with open(filename, "wb") as f:
            f.write(J2A._Header.pack(
                signature=b'ALIB',
                magic=0x00BEBA00,
                headersize=headersize,
                version=0x200,
                unknown=0x1808,
                filesize=cur_offset,
                crc32=crc,
                setcount=setcount
            ))
            f.write(set_offsets_raw)
            for sdata in set_data:
                f.write(sdata)
            if extra_data:
                f.write(extra_data)
        return self

    def pack(self):
        for s in self.sets:
            s.pack(self.config)
        return self

    def get_palette(self, given = None):
        if not self.palette:
            palfile = open(self.config["palette"]).readlines() if not given else given
            pal = list()
            for i in range(3, 259):
                color = palfile[i].rstrip("\n").split(' ')
                pal.append((int(color[0]), int(color[1]), int(color[2])))
            self.palette = pal
            self.palettesequence = [band for color in pal for band in color]

        return self.palette

    def render_pixelmap(self, frame):
        img = Image.new("RGBA", frame.shape)
        im = img.load()
        pal = self.get_palette()

        for x, row in enumerate(frame.decode_image()._pixmap):
            for y, index in enumerate(row):
                if index > 1:
                    im[y, x] = pal[index]

        return img

    def render_paletted_pixelmap(self, frame):
        pixelmap = frame.decode_image()._pixmap
        img = Image.new("P", frame.shape)
        img.putdata([pixel for col in pixelmap for pixel in col])
        self.get_palette()
        img.putpalette(self.palettesequence)
        return img

    def get_frame(self, set_num, anim_num, frame_num):
        ''' gets image info and image corresponding to a specific set, animation and frame number '''
        s = self.sets[set_num]
        anim = s.animations[anim_num]
        frame = anim.frames[frame_num]

        return [frame, self.render_pixelmap(frame)]

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
