import util


class rANSEncoder:
    def __init__(self, text_buffer, model, output_path,
                 precision=14, state_bytes=4, bytes_per_write=1):
        self.buffer = text_buffer
        self.model = model
        self.output_path = output_path

        self.precision = precision
        self.state_bytes = state_bytes
        self.L = 1 << (8 * (state_bytes - bytes_per_write) - 1)
        self.bits_per_write = bytes_per_write * 8
        self.bytes_per_write = bytes_per_write

        self.out = None

    def open(self):
        self.out = open(self.output_path, 'wb')

    def close(self):
        self.out.close()
        self.out = None

    def get_probs_from_dist(self, sym, dist):
        cs = 0
        for key in dist:
            fs = max(1, int(dist[key] * (1 << self.precision)))
            remaining = (1 << self.precision) - cs
            if remaining == 1:
                return 1, cs, True
            if remaining >= fs:
                fs = remaining - 1
            if sym == key:
                return fs, cs, False
            cs += fs
        raise ValueError('symbol not in distribution')

    def write_seq(self, probabilities, num_symbols):
        if self.out is None:
            raise ValueError('encoder is not in write mode')

        x = 1
        bytes_written = 0
        for i in range(num_symbols):
            fs, cs = probabilities[i]

            # renormalize
            x_max = ((self.L << self.bits_per_write) >> self.precision) * fs
            while x > x_max:
                low_bytes = x & ((1 << self.bits_per_write) - 1)
                self.out.write(low_bytes.to_bytes(self.bytes_per_write, byteorder='big'))
                x >>= self.bits_per_write
                bytes_written += 1

            # encode
            x = ((x / fs) << self.precision) + (x % fs) + cs

        self.out.write(x.to_bytes(self.state_bytes, byteorder='big'))

        util.write_expanding_num(bytes_written, self.out)
