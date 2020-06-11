# python j2a parser
# by stijn, edited by violet clm
# thanks to neobeo/j2nsm for the file format specs
# see http://www.jazz2online.com

from __future__ import print_function
import itertools
import struct
import os
import sys
import logging
import zlib
import array
#needs python image library, http://www.pythonware.com/library/pil/
from PIL import Image

import misc

if sys.version_info[0] <= 2:
    zip = itertools.izip
    as_compressible = lambda x : bytes(x)
else:
    as_compressible = lambda x : x

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
if not getattr(logger, "handlers", False):
    handler   = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    del handler, formatter
error, warning, info = logger.error, logger.warning, logger.info

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


class J2A:
    __slots__ = ["sets", "filename", "config"]
    _Header = misc.NamedStruct("4s|signature/L|magic/L|headersize/h|version/h|unknown/L|filesize/L|crc32/L|setcount")
    _defaultconfig = {"compress_method": 9, "null_image": "error", "null_mask": "ignore", "fake_size_and_crc": None,
        "empty_set": None}
    _error_action = {
        "ignore":  (lambda s, c: None),
        "warning": warning,
        "error":   raising_function
    }

    class ParsingError(Exception):
        pass

    class PackingError(Exception):
        pass

    class Set(object):
        __slots__ = ["_chunks", "_samplecount", "samplesbaseindex", "_anims", "_samples"]
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
            assert (setheader["signature"], setheader["u1"], setheader["u2"]) == \
                   (b'ANIM', J2A.Animation._Header.size * setheader["animcount"], J2A.Frame._Header.size * setheader["framecount"]), \
                   "header inconsistency"
            chunks = [(f.read(setheader["c" + k]), setheader["u" + k]) for k in "1234"]
            for chunk in chunks:
                crc = zlib.crc32(chunk[0], crc)
            return (J2A.Set(setheader, chunks), crc)

        def unpack(self):
            if self._chunks:
                animinfo, frameinfo, imagedata, sampledata = \
                    (zlib.decompress(c, zlib.MAX_WBITS, usize) for c,usize in self._chunks)

                animinfo  = (J2A.Animation._Header.iter_unpack(animinfo))
                frameinfo = list(J2A.Frame._Header.iter_unpack(frameinfo))

                offsets = sorted((info[key], 2*i+j) for i, info in enumerate(frameinfo) for j, key in enumerate(("imageoffset", "maskoffset")))
                offsets.append((len(imagedata), 2*len(frameinfo)))
                if take(1, (o for o,i in offsets if o != -1 and o & 3)):
                    warning("unaligned offset found while unpacking")
                data = sorted((i1, None) if o1 == -1 else (i1, imagedata[o1:o2]) for (o1, i1), (o2, i2) in pairwise(offsets))
                frames = (J2A.Frame.read(info, img, mask) for ((i1, img), (i2, mask)), info in zip(grouper(data, 2), frameinfo))
                self._anims = [J2A.Animation(frames=take(info["framecount"], frames), fps=info["fps"]) for info in animinfo]

                self._samples = []
                offset, length = 0, len(sampledata)
                while offset < length:
                    sample, offset = J2A.Sample.read(sampledata, offset)
                    if sample is None:
                        break
                    self._samples.append(sample)
                if len(self._samples) != self._samplecount:
                    warning("internal sample miscount (expected: %d, got: %d)", self._samplecount, len(self._samples))

                self._chunks = None
                del self._samplecount
            return self

        @staticmethod
        def _compress(animinfo, frameinfo, imagedata, sampledata, config):
            compress_method = config["compress_method"]
            udata = (animinfo, frameinfo, imagedata, sampledata)

            if isinstance(compress_method, int):
                return [(zlib.compress(as_compressible(c), compress_method), len(c)) for c in udata]

            def compress_ext(raw, *pargs):
                c_obj = zlib.compressobj(*pargs)
                return c_obj.compress(as_compressible(raw)) + c_obj.flush()

            if isinstance(compress_method, tuple):
                return [(compress_ext(c, *compress_method), len(c)) for c in udata]
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
                    raise J2A.PackingError("'samplesbaseindex' member must be set when packing a set with samples")

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

        # TODO: this is too slow, it needs to be parallelized
        def pack(self, config):
#             from time import time  # TODO: delme
            if not self._chunks:
#                 print("Start", time())
                animinfo = J2A.Animation._Header.iter_pack(
                    {"framecount": len(a.frames), "fps": a.fps, "reserved": 0} for a in self._anims
                )
#                 print("Animinfo packed", time())
                l_frameinfo = []
                img_data, mask_data = bytearray(), bytearray()
                null_pixmaps = null_masks = 0
                for anim in self._anims:
                    for f in anim.frames:
                        f.encode_image()

#                 print("Frames encoded", time())
                for anim in self._anims:
                    for f in anim.frames:
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
                    J2A._error_action[config["null_image"]]("found %d frames with null images" % null_pixmaps, J2A.PackingError)
                if null_masks > 0:
                    J2A._error_action[config["null_mask" ]]("found %d frames with null masks"  % null_masks  , J2A.PackingError)

                img_length = len(img_data)
                for frame_info in l_frameinfo:
                    maskoffset = frame_info["maskoffset"]
                    frame_info["maskoffset"] = maskoffset + img_length if maskoffset != -1 else -1
#                 print("Image data & mask data packed", time())
                frameinfo = J2A.Frame._Header.iter_pack(l_frameinfo)
#                 print("Image data & mask data packed", time())

                sampledata = b''.join(sample.serialize() for sample in self._samples)
                self._samplecount = len(self._samples)
#                 print("Samples packed", time())

                self._chunks = J2A.Set._compress(animinfo, frameinfo, img_data + mask_data, sampledata, config)
#                 print("Chunks compressed", time())
                del self._anims, self._samples
            return self

        @property
        def animations(self):
            if self._chunks:
                self.unpack()
            return self._anims
        @animations.setter
        def animations(self, value):
            if self._chunks:
                self.unpack()
            self._anims = value

        @property
        def samples(self):
            if self._chunks:
                self.unpack()
            return self._samples
        @samples.setter
        def samples(self, value):
            if self._chunks:
                self.unpack()
            self._samples = value


    class Animation(object):
        __slots__ = ["frames", "fps"]
        _Header = misc.NamedStruct("H|framecount/H|fps/l|reserved")

        def __init__(self, frames=None, fps=10):
            if frames is None:
                frames = []
            self.frames, self.fps = frames, fps


    class Frame(object):
        __slots__ = ["shape", "origin", "coldspot", "gunspot", "_pixmap", "mask", "_rle_encoded_pixmap", "tagged"]
        _Header = misc.NamedStruct("H|width/H|height/h|coldspotx/h|coldspoty/h|hotspotx/h|hotspoty/h|gunspotx/h|gunspoty/l|imageoffset/l|maskoffset")

        def __init__(self, shape=None, origin=None, coldspot=(0,0), gunspot=(0,0), pixmap=None, mask=None, rle_encoded_pixmap=None, tagged=False):
            assert (pixmap is None) ^ (rle_encoded_pixmap is None)
            self.shape, self.origin, self.coldspot, self.gunspot, self.mask, self.tagged = shape, origin, coldspot, gunspot, mask, tagged
            if not rle_encoded_pixmap is None:
                assert not shape is None
                self._rle_encoded_pixmap = bytearray(rle_encoded_pixmap)
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
            assert width == frameinfo["width"] and height == frameinfo["height"]
            return J2A.Frame(
                shape = (width, height),
                origin = (frameinfo["hotspotx"], frameinfo["hotspoty"]),
                coldspot = (frameinfo["coldspotx"], frameinfo["coldspoty"]),
                gunspot = (frameinfo["gunspotx"], frameinfo["gunspoty"]),
                rle_encoded_pixmap = imagedata[4:],
#                 mask = maskdata,
                mask = maskdata and maskdata[:(width * height + 7) >> 3],  # None or an appropriately-sized bytearray
                tagged = tagged
            )

        # TODO: need to stress test these two methods
        def decode_image(self):
            if not hasattr(self, "_pixmap"):
                width, height = self.shape
                raw = self._rle_encoded_pixmap
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
                encoded = bytearray()
                for row in self._pixmap:
                    while True:
                        row = bytearray(row)
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
                            encoded += row[:m]
                            row = row[m:]
                            length -= m
                    encoded.append(0x80)
                self._rle_encoded_pixmap = encoded
                del self._pixmap
            return self

        def autogenerate_mask(self):
            self.decode_image()
            mask = bytearray(b'\x00') * ((self.shape[0] * self.shape[1] + 7) // 8)
            pix_iter = itertools.chain(*self._pixmap)
            for i in range(len(mask)):
                mask[i] = sum(bool(pix) << j for j, pix in enumerate(take(8, pix_iter)))
            self.mask = mask
            return self


    class Sample(object):
        __slots__ = ["_data", "_rate", "volume", "_bits", "_channels", "loop"]
        _Header = misc.NamedStruct("L|total_size/4s|riff_id/L|riff_size/4s|format/4s|sc_id/L|sc_size/L|reserved1_size/32s|reserved1/H|unknown1/h|volume/H|flags/H|unknown2/L|nsamples/L|loop_start/L|loop_end/L|sample_rate/L|has_appendix/L|reserved2")
        _header_defaults = {"riff_id": b'RIFF', "format": b'AS  ', "sc_id": b'SAMP', "reserved1_size": 0x40, "reserved1": b'\x00' * 0x40, "unknown1": 0x4000, "unknown2": 0x0080, "has_appendix": 0, "reserved2": 0}

        def __init__(self, data, sample_rate=None, volume=None, bits=8, channels=1, loop=None):
            assert not sample_rate is None and not volume is None
            assert isinstance(data, (bytes, bytearray))
            assert (len(data) << 3) % (channels * bits) == 0
            self._data, self._rate, self.volume, self._bits, self._channels, self.loop = \
                data, sample_rate, volume, bits, channels, loop

        @staticmethod
        def read(raw, offset):
            try:
                header = J2A.Sample._Header.unpack_from(raw, offset)
            except:
                if not any(raw[offset:]):
                    return None, offset
                else:
                    raise

            is_16bit  = bool(header["flags"] & 0x4)
            is_stereo = bool(header["flags"] & 0x40)
            loop      = (header["loop_start"], header["loop_end"], bool(header["flags"] & 0x10)) if header["flags"] & 0x18 else None
            sample_data_size = header["nsamples"] * (1 + is_16bit) * (1 + is_stereo)
            sample_data_offset = offset + J2A.Sample._Header.size + (0 if not header["has_appendix"] else 0x9e)

            assert (header["riff_id"], header["format"], header["sc_id"]) == (b'RIFF', b'AS  ', b'SAMP'), \
                   "signatures mismatch"
            assert header["total_size"] - header["riff_size"] == 0xc,       "sizes mismatch"
            assert (header["total_size"] - header["sc_size"]) & -2 == 0x18, "sizes mismatch"
            assert header["reserved1_size"] == 0x40,                        "sizes mismatch"
            assert header["sc_size"] == sample_data_size + 0x44,            "sizes mismatch"
            if header["reserved1"].lstrip(b'\x00'):
                warning("found nonzero sample reserved area")

            sample_data = raw[sample_data_offset:sample_data_offset + sample_data_size]

            sample = J2A.Sample(sample_data,
                bits = (1 + is_16bit) * 8,
                channels = 1 + is_stereo,
                loop = loop,
                volume = header["volume"],
                sample_rate = header["sample_rate"]
            )
            return (sample, offset + header["total_size"])

        def serialize(self):
            header = J2A.Sample._header_defaults.copy()
            datalen = len(self._data)
            nsamples = datalen // ((self._bits >> 3) * self._channels)
            total_size = (datalen + 0x5d) & -2
            loop = self.loop or (0, 0, 0)
            header.update(total_size = total_size, riff_size = total_size - 0xc, volume = self.volume, sc_size = datalen + 0x44,
                flags = 0x4 * (self._bits == 16) + 0x8 * (not self.loop is None) + 0x10 * loop[2] + 0x40 * (self._channels == 2),
                nsamples = nsamples, loop_start = loop[0], loop_end = loop[1], sample_rate = self._rate,
            )
            retval = J2A.Sample._Header.pack(**header) + self._data + b'\x00' * (datalen & 1)
            assert len(retval) == total_size
            return retval


    def __init__(self, filename=None, **kwargs):
        ''' initializes class, sets file name '''
        self.sets = []
        self.set_filename(filename)
        self.config = J2A._defaultconfig.copy()
        self.config.update(kwargs)

    def set_filename(self, filename):
        self.filename = filename
        return self

    @staticmethod
    def _open(filespec, mode, *pargs, **kwargs):
        if isinstance(filespec, (int, str)):
            return open(filespec, mode, *pargs, **kwargs)
        else:  # Assume it's a file object
            method = {"r": "read", "w": "write"}[mode[:1]]
            if hasattr(filespec, method):
                return filespec
            else:
                ErrorClass = {"r": J2A.ParsingError, "w": J2A.PackingError}[mode[:1]]
                raise ErrorClass("'filename' must be either a path, a file descriptor or an object with a '%s' method" % method)

    @staticmethod
    def _seek(f, newpos):
        delta = newpos - f.tell()
        if delta > 0:
            warning("skipping over %d bytes", delta)
            b = f.read(delta)
            assert len(b) == delta
        elif delta < 0:
            raise J2A.ParsingError("File is not a valid J2A file (overlapping sets)")

    def _get_setlist(self, *pargs):
        if not pargs:  # Default to all sets, unless explicitly passed an empty iterable
            return self.sets
        if len(pargs) == 1 and hasattr(pargs[0], "__iter__"):
            pargs = pargs[0]

        def get_set(arg):
            if isinstance(arg, int):
                return self.sets[arg]
            elif arg in self.sets:
                return arg
            raise ValueError("arguments must be either indexes or sets already belonging to the J2A instance")

        return list(map(get_set, pargs))

    def read(self, filename=None):
        ''' reads whole J2A file, parses ALIB and ANIM headers and collects all sets '''
        if filename is None:
            filename = self.filename
            if filename is None:
                raise J2A.ParsingError("no filename specified")

        with J2A._open(filename, "rb") as j2afile:
            # TODO: maybe add a separate check for ALIB version?
            try:
                alibheader = self._Header.unpack(j2afile.read(self._Header.size))
                setcount = alibheader["setcount"]
                assert (alibheader["signature"], alibheader["magic"], alibheader["headersize"], alibheader["version"]) == \
                       (b'ALIB', 0x00BEBA00, self._Header.size + 4*setcount, 0x0200)
                if alibheader["unknown"] != 0x1808:
                    warning("minor difference found in ALIB header. Ignoring...")
                raw = j2afile.read(4*setcount)
                setoffsets = struct.unpack('<%dL' % setcount, raw)
                crc = zlib.crc32(raw)
                assert setoffsets[0] == alibheader["headersize"]
                self.sets = []
                for offset in setoffsets:
                    # The shareware demo removes some of the animsets to save on filesize, but leaves the
                    # order of animations intact, causing gaping holes with offsets of zero in the .j2a file
                    if offset == 0:
                        self.sets.append(J2A.Set())
                    else:
                        J2A._seek(j2afile, offset)
                        s, crc = J2A.Set.read(j2afile, crc)
                        self.sets.append(s)
                raw = j2afile.read()
                if raw:
                    warning("extra %d bytes found at the end of J2A file %s. Ignoring...", len(raw), self.filename)
                    crc = zlib.crc32(raw, crc)
                if crc & 0xffffffff != alibheader["crc32"]:
                    warning("CRC32 mismatch in J2A file %s. Ignoring...", self.filename)
            except (AssertionError, struct.error):
                raise J2A.ParsingError("File %s is not a valid J2A file" % self.filename)

        return self

    def unpack(self, *pargs):
        for s in self._get_setlist(*pargs):
            s.unpack()
        return self

    def write(self, filename=None):
        if filename is None:
            filename = self.filename
            if filename is None:
                raise J2A.PackingError("no filename specified")
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
                    raise J2A.PackingError("Can't fake filesize and CRC32, obtained size is too large")
                extra_data = b'\x00' * (target_filesize - cur_offset - 4)
                crc = zlib.crc32(extra_data, crc)
                salt = misc.fake_crc(target_crc)
                salt = struct.pack("<L", crc ^ salt)
                extra_data += salt
                crc = zlib.crc32(salt, crc)
                cur_offset += len(extra_data)
                assert crc == target_crc and cur_offset == target_filesize

        with J2A._open(filename, "wb") as f:
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

    def pack(self, *pargs):
        for s in self._get_setlist(*pargs):
            s.pack(self.config)
        return self


class FrameConverter(object):
    __slots__ = ["_palette", "_palette_flat", "_img"]

    class ParsingError(Exception):
        pass

    def __init__(self, palette=None, palette_data=None, palette_file=None):
        if palette is None and palette_data is None and palette_file is None:
            raise ValueError("argument required")

        if palette is None and palette_data is None:
            with open(palette_file, "rb") as f:
                palette_data = bytearray(f.read())

        if palette is None:
            palette_data = palette_data if isinstance(palette_data, bytearray) else bytearray(palette_data)
            palette = FrameConverter._read_palette(palette_data)

        if not isinstance(palette, (list, tuple)):
            raise ValueError("palette must be a list or tuple of integer triplets (R, G, B)")

        self.palette = palette

    @property
    def palette(self):
        return self._palette[:]
    @palette.setter
    def palette(self, value):
        "Expects 'value' to be a list of 3-tuples"
        self._palette = value
        self._palette_flat = bytes(bytearray(itertools.chain(*value)))
        self._img = Image.new("P", (0, 0), None)
        self._img.im.putpalette("RGB", self._palette_flat)

    def to_image(self, frame, mode="P"):
        img = Image.new(mode, frame.shape)
        if mode in ("RGB", "RGBA"):
            img_data = img.load()
            pal = self._palette
            for x, row in enumerate(frame.decode_image()._pixmap):
                for y, index in enumerate(row):
                    if index > 1:
                        img_data[y, x] = pal[index]
        elif mode == "P":
            img.putdata(list(itertools.chain(*frame.decode_image()._pixmap)))
            img.putpalette(self._palette_flat)
            # This alone won't actually export the transparency in Pillow version < 2.3
            img.info["transparency"] = 0
        else:
            raise ValueError("unsupported mode")
        return img

    def from_image(self, image, perform_conversion = False, **kwargs):
        if not (image.mode == "P" and image.palette.getdata()[1] == self._palette_flat):
            if perform_conversion:
                image = image.quantize(palette = self._img)
            elif image.mode != "P":
                raise ValueError("image is not paletted")
            else:
                raise ValueError("image has wrong palette")
        width, height = image.size
        pixmap = [bytearray(row) for row in grouper(image.tobytes(), width)]
        return J2A.Frame(shape = (width, height), pixmap = pixmap, **kwargs)

    #########################################################################################################

    @staticmethod
    def _read_palette(palette_data):  # Must be a bytearray
        for parser in FrameConverter._pal_parsers:
            try:
                result = parser(palette_data)
                if result:
                    assert len(result) == 256
                    return result
            except:
#                 from traceback import print_exc; print_exc()
                pass
        raise FrameConverter.ParsingError("unrecognized palette format")

    @staticmethod
    def _parse_JASC_PAL(palette_data):  # JASC .pal file (PaintShop Pro)
        if palette_data.startswith(b'JASC-PAL\r\n0100\r\n256\r\n'):
            palette_data = palette_data.splitlines()[3:]
            return [(int(r), int(g), int(b)) for r, g, b in map(bytearray.split, palette_data)]

    @staticmethod
    def _parse_GIMP_PAL(palette_data):  # GIMP palette
        if palette_data.startswith(b'GIMP Palette'):
            palette_data = palette_data.splitlines()[1:]
            if palette_data[0].startswith(b'Name: '):
                palette_data.pop(0)
            if palette_data[0].startswith(b'Columns: '):
                palette_data.pop(0)
            filtered_split = (row.split()[:3] for row in filter(lambda row : not row.startswith(b'#'), palette_data))
            return [(int(r), int(g), int(b)) for r, g, b in filtered_split]

    @staticmethod
    def _parse_JJ2_PAL(palette_data):  # format used by JJ2 in Data.j2d
        if len(palette_data) == 256 * 4 and not any(palette_data[3::4]):
            return [(r, g, b) for r, g, b, _ in grouper(palette_data, 4)]

    @staticmethod
    def _parse_ACT(palette_data):  # .ACT file (Adobe Photoshop color table)
        if len(palette_data) == 256 * 3 + 4:
            assert palette_data[-4:] == b'\x01\x00\x00\x00'
            palette_data = palette_data[:-4]
        if len(palette_data) == 256 * 3:
            return [(r, g, b) for r, g, b in grouper(palette_data, 3)]

    @staticmethod
    def _parse_ACO(palette_data):  # .ACO file (Adobe Photoshop color swatch)
        if len(palette_data) >= 2564 and palette_data.startswith(b'\x00\x01\x01\x00'):
            color_data = struct.unpack_from(">1280H", palette_data, 4)
            if ( any(color_data[0::5])  # only RGB supported...
                    or any(color_data[4::5]) ):  # .. which means these should be 0 as well
                return
            if len(palette_data) > 2564:  # Validate version 2 data
                assert palette_data[2564:2568] == b'\x00\x02\x01\x00'
                extra_data = struct.unpack_from(">%dH" % ((len(palette_data) - 2568) // 2), palette_data, 2568)
                offset = 0
                for color in grouper(color_data, 5):
                    assert color == extra_data[offset:offset+5]
                    name_len = (extra_data[offset + 5] << 16) + extra_data[offset + 6]
                    offset += 5 + 2 + name_len  # Skipping over name as well (no Unicode validation done)
                assert offset == len(extra_data)
            return list(grouper((color_data[5*i+j] >> 8 for i in range(256) for j in (1, 2, 3)), 3))

FrameConverter._pal_parsers = [FrameConverter._parse_JASC_PAL, FrameConverter._parse_GIMP_PAL, FrameConverter._parse_JJ2_PAL,
    FrameConverter._parse_ACT, FrameConverter._parse_ACO]


def main():
    from sys import argv
    filename = argv[1] if (len(argv) >= 2) else r"C:\Games\Jazz2\Anims.j2a"
    try:
        anims = J2A(filename).read()
    except IOError:
        warning("File %s could not be read!", filename)
        return 1

    set_num, anim_num, frame_num = (9, 0, 6)
    frame = anims.sets[set_num].animations[anim_num].frames[frame_num]
    FrameConverter(palette_file = "Diamondus_2.pal").to_image(frame).save("preview.png", "PNG")

if __name__ == "__main__":
    sys.exit(main())
