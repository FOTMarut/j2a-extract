import re
import struct

class NamedStruct(struct.Struct):
    _fmtvalidate = re.compile(r"^((\d+[sp]|[xcbB?hHiIlLqQnNefd])\|\w+/)*(\d+[sp]|[xcbB?hHiIlLqQnNefd])\|\w+$")

    def __init__(self, fmt):
        assert(NamedStruct._fmtvalidate.match(fmt))
        pairs = [it.split("|", 1) for it in fmt.split("/")]
        self._names = tuple(p[1] for p in pairs)
        super(NamedStruct, self).__init__("<" + "".join(p[0] for p in pairs))

    def pack(self, **kwargs):
        l = [None] * len(kwargs)
        for k,v in kwargs.items():
            l[self._names.index(k)] = v
        return super(NamedStruct, self).pack(*l)

    def pack_into(self, buffer, offset, **kwargs):
        l = [None] * len(kwargs)
        for k,v in kwargs.items():
            l[self._names.index(k)] = v
        return super(NamedStruct, self).pack_into(buffer, offset, *l)

    def unpack(self, *pargs, **kwargs):
        up = super(NamedStruct, self).unpack(*pargs, **kwargs)
        return {k: v for k,v in zip(self._names, up)}

    def unpack_from(self, *pargs, **kwargs):
        up = super(NamedStruct, self).unpack_from(*pargs, **kwargs)
        return {k: v for k,v in zip(self._names, up)}

    def iter_unpack(self, buffer, *pargs, **kwargs):
        offset = 0
        length = len(buffer)
        while offset < length:
            yield self.unpack_from(buffer, offset, *pargs, **kwargs)
            offset += self.size

    def iter_pack(self, iterable):
        return b''.join(self.pack(**d) for d in iterable)

# TODO: replace dict comprehensions with dict() constructor, if possible

#--- Obsoleted ---
# def named_unpack(format, string):
#     ''' hacky wrapper for struct.unpack() that allows for easier format specification '''
#     format = re.sub("[^0-9a-zA-Z?/|]", "", format)
#     sizes = {"x": 1, "c": 1, "b": 1, "B": 1, "?": 1, "h": 2, "H": 2, "i": 4, "I": 2, "l": 4, "L": 4, "q": 8, "Q": 8, "f": 0, "d": 0} #don"t use d or f!
#     items = format.split("/")
#     ret = dict()
#     for item in items:
#         base = item.split("|")
#         count = re.sub("[^0-9]", "", base[0])
#         byte = str(re.sub("[^a-zA-Z?]", "", base[0]))
#         count = int(count) if len(count) != 0 else 1
#         if byte not in ["p", "s"]:
#             for i in range(1, count+1):
#                 sub = string[0:sizes[byte]]
#                 suffix = str(i) if count > 1 else ""
#                 key = base[1]+suffix if len(base[1]) > 0 else int(suffix)
#                 ret[key] = struct.unpack("<" + byte, sub)[0]
#                 string = string[sizes[byte]:]
#         else:
#             sub = string[0:count]
#             if byte != "x":
#                 ret[base[1]] = struct.unpack("<" + str(count) + byte, sub)[0]
#             string = string[count:]
# 
#     return ret
#--- Obsoleted ---
