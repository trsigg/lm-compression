class rANSDecoder:
    def __init__(self, input_reader, precision=14, state_bytes=4,
                 bytes_per_read=1):
        self.input = input_reader

        self.precision = precision
        self.normalize_fctr = (1 << self.precision)
        self.state_bytes = state_bytes
        self.L = 1 << (8 * (state_bytes - bytes_per_read) - 1)
        self.bits_per_write = bytes_per_read * 8
        self.bytes_per_read = bytes_per_read

        self.x = None

    def init_chunk(self):
        self.input.set_mode(False)
        self.x = int.from_bytes(self.input.read(self.state_bytes), byteorder='big')

    def extract_probs_and_symbol(self, dist, residue):
        cs = 0
        for key in dist:
            remaining = self.normalize_fctr - cs
            if remaining == 1:  # overflow
                return 1, cs, None

            fs = max(1, int(dist[key] * self.normalize_fctr))
            if remaining <= fs:  # truncate
                fs = remaining - 1
            if cs + fs > residue:  # match
                return fs, cs, key

            cs += fs

        return 1, cs, None

    def decode_token(self, dist, end_pos):
        residue = self.x & (self.normalize_fctr - 1)
        fs, cs, symbol = self.extract_probs_and_symbol(dist, residue)
        self.x = fs * (self.x >> self.precision) + residue - cs

        should_continue = True

        while self.x < self.L:
            if self.input.tell() <= end_pos:
                should_continue = False
                break
            self.x <<= self.bits_per_write
            self.x += int.from_bytes(self.input.read(self.bytes_per_read),
                                     byteorder='big')

        return symbol, should_continue
