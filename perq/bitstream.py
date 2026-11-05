"""
One-image bitstream file: a 10-byte header (magic, version, height, width, rate point) and the entropy-coded bytes.
"""
import struct


MAGIC = b'PERQ'
VERSION = 1
HEADER = '>4sBHHB'


def write_bitstream(path, bit_stream, height, width, rp):
    with open(path, 'wb') as f:
        f.write(struct.pack(HEADER, MAGIC, VERSION, height, width, rp))
        f.write(bit_stream)


def read_bitstream(path):
    with open(path, 'rb') as f:
        data = f.read()
    n = struct.calcsize(HEADER)
    magic, version, height, width, rp = struct.unpack(HEADER, data[:n])
    assert magic == MAGIC, f'{path} is not a PerQ bitstream (magic {magic!r})'
    assert version == VERSION, f'{path}: bitstream version {version}, this code reads version {VERSION}'
    return data[n:], {'height': height, 'width': width, 'rp': rp}