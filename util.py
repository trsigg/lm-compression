import math


def calc_bytelen(x):
    return math.ceil(math.log2(x+1))

def write_expanding_num(x, file):
    bytelen = calc_bytelen(x)
    if calc_bytelen(bytelen) > 1:
        raise ValueError('something is very wrong')
    file.write(x.to_bytes(bytelen, byteorder='big'))
    file.write(bytelen.to_bytes(1, byteorder='big'))

def read_expanding_num(file):
    bytelen = int.from_bytes(file.read(1), byteorder='big')
    file.seek(file.tell() - bytelen - 1)
    return int.from_bytes(file.read(bytelen), byteorder='big')
