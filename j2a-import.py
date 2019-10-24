from __future__ import print_function
import os
import sys
import glob
import struct
import re
import zlib
from PIL import Image
import misc

if sys.version_info[0] <= 2:
    input = raw_input

#leaf frame order:
#0 1 2 1 0 3 4 5 6 7 8 9 3 4 5 6 7 8 9 0 1 2 1 0 10 11 12 13 14 15 16

#http://stackoverflow.com/questions/2669059/how-to-sort-alpha-numeric-set-in-python
convert = lambda text: int(text) if text.isdigit() else text
alphanum_key = lambda key: [ convert(c) for c in re.split('(\d+)', key) ]

def write_frame(frame, data2, data3, imageaddress):
    frameinfo = re.match(r"^\d+(t?),(-?\d+),(-?\d+),(-?\d+),(-?\d+),(-?\d+),(-?\d+).png$", os.path.basename(frame)).groups()
    image = Image.open(frame)
    data = list(image.getdata())
    width, height = image.size
    data2 += struct.pack("<HHhhhhhhl",
        width,
        height,
        int(frameinfo[3]), #coldspotx
        int(frameinfo[4]), #y
        int(frameinfo[1]), #hotspotx
        int(frameinfo[2]), #y
        int(frameinfo[5]), #gunspotx
        int(frameinfo[6]), #y
        imageaddress
    )
    data3 += struct.pack("<HH",
        width if (len(frameinfo[0]) == 0) else (width + 0x8000),
        height
    )
    pixelstodraw = list()
    mask = bytearray()
    maskbyte = 0
    maskbitposition = 1
    skipped = data[0] != 0
    column = 1
    for i in range(len(data)):
        byte = data.pop(0)
        skipping = byte == 0
        if not skipping:
            maskbyte |= maskbitposition
        if maskbitposition == 128:
            maskbitposition = 1
            mask += bytearray((maskbyte, ))
            maskbyte = 0
        else:
            maskbitposition <<= 1
        if column == width:
            if skipped:
                if not skipping:
                    data3 += bytearray((len(pixelstodraw), 0x81, byte))
            else:
                if not skipping:
                    pixelstodraw.append(byte)
                while len(pixelstodraw) > 0x7f:
                    data3 += bytearray((0x7f + 0x80, ))
                    data3 += bytearray(pixelstodraw[:0x7f])
                    pixelstodraw = pixelstodraw[0x7f:]
                data3 += bytearray((len(pixelstodraw) + 0x80, ))
                data3 += bytearray(pixelstodraw)
            data3 += b'\x80'
            if len(data) == 0:
                if maskbitposition != 1: #not on a byte boundary
                    mask += bytearray((maskbyte, ))
                break;
            column = 1
            skipped = data[0] != 0
            pixelstodraw = list()
            continue
        elif skipping == skipped and len(pixelstodraw) < 127:
            pass
        elif len(pixelstodraw) > 0:
            if not skipped:
                data3 += bytearray((len(pixelstodraw) + 0x80, ))
                data3 += bytearray(pixelstodraw)
            else:
                data3 += bytearray((len(pixelstodraw), ))
            pixelstodraw = list()
        pixelstodraw.append(byte)
        skipped = skipping
        column += 1
    data2 += struct.pack("<l", len(data3))
    #data3.ljust((len(data3)+7)/8*8, '\0')  # Pad the image and mask out to multiples of eight bytes. I don't know why these would be
    #mask.ljust((len(mask)+7)/8*8, '\0')    # needed, but if anything does go wrong somewhere, uncommenting these lines might fix it?
    data3 += mask
    #print([bin(ord(c))[2:].rjust(8,'0')[::-1] for c in mask])

def main():
    dirname = sys.argv[1] if (len(sys.argv) >= 2) else \
        input("Please type the folder you wish to import:\n")
    dirname = os.path.normpath(dirname)
    outfilename = sys.argv[2] if (len(sys.argv) >= 3) else \
            dirname.replace("-j2a", ".j2a")
    if not os.path.isdir(dirname):
        print("Folder does not exist!")
        return 1
    if not re.search(r"-j2a$", os.path.basename(dirname)):
        print("Folder name is improperly formatted (must be named ***-j2a)!")
        return 1
    setdirlist = [os.path.join(dirname, name) for name in sorted(os.listdir(dirname)) if os.path.isdir(os.path.join(dirname, name))]
    setcount = len(setdirlist)
    setdirlist.sort(key=alphanum_key)
    if setcount == 0:
        print("No sets were found in that folder. No .j2a file will be compiled.", file=sys.stderr)
        return 1
    setpositions = [(7+setcount)*4] #starts at header length
    j2a = open(outfilename, "wb")
    j2a.write(struct.pack("<4sllhhlll",
        b'ALIB', #magic
        0x00BEBA00, #also magic
        setpositions[0], #header length
        0x0200, #version
        0x1808, #?
        0, #filesize
        0, #crc
        0 #setcount
    ))
    j2a.write(b'\0\0\0\0' * setcount) #set positions
    for setdir in setdirlist:
        j2a.write(b'ANIM')
        animdirlist = [os.path.join(setdir, name) for name in sorted(os.listdir(setdir)) if name.isdigit() and os.path.isdir(os.path.join(setdir, name))]
        animcount = len(animdirlist)
        animdirlist.sort(key=alphanum_key)
        framecount = 0
        data1, data2, data3, data4 = (bytearray() for _ in range(4))
        imageaddress = 0
        for animdir in animdirlist:
            framelist = glob.glob( os.path.join(animdir, '*.png') )
            thisframecount = len(framelist)
            framecount += thisframecount
            framelist.sort(key=alphanum_key)
            fpsfile = glob.glob( os.path.join(animdir, 'fps.*') )
            if len(fpsfile) == 1:
                fps = int(os.path.splitext(fpsfile[0])[1][1:])
            else:
                fps = 10
            data1 += struct.pack("<hhl",
                thisframecount,
                fps,
                0
            )
            for frame in framelist:
                write_frame(frame, data2, data3, imageaddress)
                imageaddress = len(data3)
        j2a.write(struct.pack("<bbhl",
            animcount,
            0, #samplecount
            framecount,
            0 #priorsamplecount
        ))
        datas = [bytes(d) for d in (data1, data2, data3, data4)]
        ulengths,clengths,compressedblocks = [],[],[]
        for i in range(4):
            ulengths.append(len(datas[i]))
            compressedblocks.append(zlib.compress(datas[i], 9))
            clengths.append(len(compressedblocks[i]))
#         print("# set {:4} ulengths {:3} {:5} {:8} {:1}  clengths {:3} {:5} {:6} {:1}".format(
#             setdir.rsplit('/', 1)[1] + ":",
#             *ulengths,
#             *clengths
#             )) #TODO: delme
        for i in range(4):
            j2a.write(struct.pack("<ll", clengths[i], ulengths[i]))
        for i in range(4):
            j2a.write(compressedblocks[i])
        setpositions.append(setpositions[-1] + 44 + sum(clengths))
    j2a.seek(16);
    j2a.write(struct.pack("<lll", setpositions[-1], 0, setcount)) #filesize
    for i in range(setcount):
        j2a.write(struct.pack("<l", setpositions[i]))
    j2a.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
