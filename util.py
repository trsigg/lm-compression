import math


def calc_bytelen(x):
    return math.ceil(math.ceil(math.log2(x+1)) / 8)


def write_expanding_bytes(b, file, forward):
    len_bytes = len(b).to_bytes(1, byteorder='big')

    if len(len_bytes) > 1:
        raise OverflowError('fk')

    if forward:
        file.write(len_bytes)
        file.write(b)
    else:
        file.write(b)
        file.write(len_bytes)

    return len(b) + 1


def read_expanding_bytes(file, forward):
    bytelen = int.from_bytes(file.read(1), byteorder='big')
    if not forward:
        file.seek(file.tell() - bytelen - 1)
    return file.read(bytelen)


def write_expanding_num(x, file, forward):
    bytelen = calc_bytelen(x)
    x_bytes = x.to_bytes(bytelen, byteorder='big')
    return write_expanding_bytes(x_bytes, file, forward)


def read_expanding_num(file, forward):
    return int.from_bytes(read_expanding_bytes(file, forward), byteorder='big')


def write_expanding_string(s, file, forward):
    s_bytes = s.encode('utf-8')
    return write_expanding_bytes(s_bytes, file, forward)


def read_expanding_string(file, forward):
    return read_expanding_bytes(file, forward).decode('utf-8')
