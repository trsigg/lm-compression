import util


class rANSEncoder:
    def __init__(self, output_path, precision=20, state_bytes=4,
                 bytes_per_write=1):
        self.output_path = output_path

        self.precision = precision
        self.state_bytes = state_bytes
        self.L = 1 << (8 * (state_bytes - bytes_per_write) - 1)
        self.bits_per_write = bytes_per_write * 8
        self.bytes_per_write = bytes_per_write

        self.out = None
        self.position_table = []

    def open(self):
        self.out = open(self.output_path, 'wb')

    def close(self):
        # write pos table
        pos_table_start = self.out.tell()
        util.write_expanding_num(len(self.position_table), self.out, True)
        for pos in self.position_table:
            util.write_expanding_num(pos, self.out, True)
        util.write_expanding_num(pos_table_start, self.out, False)

        self.out.close()
        self.out = None

    def get_probs_from_dist(self, sym, dist):
        cs = 0
        for key, prob in enumerate(dist):
            remaining = (1 << self.precision) - cs
            if remaining == 1:  # overflow
                return 1, cs, True

            fs = max(1, int(prob * (1 << self.precision)))
            if remaining <= fs:  # truncate
                fs = remaining - 1
            if sym == key:  # match
                return fs, cs, False
            cs += fs
        return (1 << self.precision) - cs, cs, True

    def write_seq(self, probabilities, num_symbols):
        if self.out is None:
            raise ValueError('encoder is not in write mode')

        x = self.L
        for i in range(num_symbols-1, -1, -1):
            fs, cs = probabilities[i]

            # renormalize
            x_max = ((self.L << self.bits_per_write) >> self.precision) * fs
            while x > x_max:
                low_bytes = x & ((1 << self.bits_per_write) - 1)
                self.out.write(low_bytes.to_bytes(self.bytes_per_write, byteorder='big'))
                x >>= self.bits_per_write

            # encode
            x = ((x // fs) << self.precision) + (x % fs) + cs

        self.out.write(x.to_bytes(self.state_bytes, byteorder='big'))

        self.position_table.append(self.out.tell() - 1)
