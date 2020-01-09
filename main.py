import rANSEncoder
import rANSDecoder
import ByteReader
import util


def encode(input_file, out_path, model, precision, chunk_size):
    model.open(input_file)
    encoder = rANSEncoder(out_path + '/ans.lm', precision)
    encoder.open()  # should probably enable <with> syntax for these
    encoding_probabilities = [(0, 0)] * chunk_size
    next_token = model.get_next_sym()  # maybe make iterator?

    with open(out_path + '/unk.lm', 'wb') as unk_out:
        while next_token:
            tokens_encoded = 0
            overflow_bytes = 0
            while next_token and tokens_encoded < chunk_size:
                next_token_encoded = model.encode(next_token)
                fs, cs, overflow = encoder.get_probs_from_dist(next_token_encoded,
                                                               model.predict())
                model.update(next_token_encoded)

                encoding_probabilities.append((fs, cs))
                if overflow or model.is_unk(next_token_encoded):
                    next_token_bin = next_token.encode('utf-8')
                    token_len = len(next_token_bin).to_bytes(1, byteorder='big')
                    if len(token_len) > 1:
                        raise ValueError('fk')
                    unk_out.write(token_len)
                    unk_out.write(next_token_bin)
                    overflow_bytes += len(next_token_bin) + 1

                tokens_encoded += 1
                next_token = model.get_next_sym()

            encoder.write_seq(encoding_probabilities)
            model.reset()
            util.write_expanding_num(overflow_bytes, unk_out)
        model.close()
        encoder.close()


def decode(input_path, out_file, model, precision):
    decoder = rANSDecoder(out_file)
    decoder.open()
    ans_reader = ByteReader(input_path + '/ans.lm', True)
    ans_reader.open()
    ans_reader.go_to_end()
    unk_reader = ByteReader(input_path + '/unk.lm')
    unk_reader.open()

    while


def main():
    encode('../test/full/in/1.txt', '../test/full/out/0', None, 14, 10000)

main()
