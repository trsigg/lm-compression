class rANSDecoder:
    def __init__(self, input_reader, precision=14, state_bytes=4,
                 bytes_per_write=1):
        self.input = input_reader

        self.precision = precision
        self.state_bytes = state_bytes
        self.L = 1 << (8 * (state_bytes - bytes_per_write) - 1)
        self.bits_per_write = bytes_per_write * 8
        self.bytes_per_write = bytes_per_write

    def decode_token(self, dist):
        for key in dist:
