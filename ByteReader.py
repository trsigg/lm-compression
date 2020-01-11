import os


class ByteReader:
    def __init__(self, path, forward=True):
        self.input_path = path
        self.forward = forward

        self.input = None
        self.input_size = float('nan')
        self.position = float('nan')

    def read(self, num_bytes):
        if self.forward:
            target = self.position  # unnecesarily slow in forward case but w/ev
            self.position += num_bytes
        else:
            self.position -= num_bytes
            target = self.position + 1

        self.bounds_check(target, num_bytes)

        self.input.seek(target)
        return self.input.read(num_bytes)

    def step(self, distance):
        self.position += distance
        self.bounds_check(self.position)

    def set_mode(self, forward):
        self.forward = forward

    def tell(self):
        return self.position

    def seek(self, pos):
        self.position = pos

    def go_to_end(self):
        self.position = self.input_size - 1

    def go_to_start(self):
        self.position = 0

    def open(self):
        self.input_size = os.path.getsize(self.input_path)
        self.input = open(self.input_path, 'rb')
        if self.forward:
            self.position = 0
        else:
            self.position = self.input_size

    def close(self):
        self.input.close()
        self.input = None
        self.input_size = float('nan')
        self.position = float('nan')

    def bounds_check(self, start, length=0):
        if start < 0 or self.input_size <= start + length:
            raise IndexError(f'range {start}-{start+length} out of range')
