import rANSEncoder
import rANSDecoder
import ByteReader
import util


def encode(input_file, out_path, model, chunk_size):
    model.open(input_file)
    encoder = rANSEncoder(out_path + '/ans.lm')
    encoder.open()  # should probably enable <with> syntax for these
    encoding_probabilities = [(0, 0)] * chunk_size
    next_token = model.get_next_sym()  # maybe make iterator?

    with open(out_path + '/unk.lm', 'wb') as unk_out:
        while next_token:
            tokens_encoded = 0
            while next_token and tokens_encoded < chunk_size:
                next_token_encoded = model.encode(next_token)
                fs, cs, overflow = encoder.get_probs_from_dist(
                                            next_token_encoded, model.predict())
                model.update(next_token_encoded)

                encoding_probabilities.append((fs, cs))
                if overflow or model.is_unk(next_token_encoded):
                    util.write_expanding_string(next_token, unk_out, True)

                tokens_encoded += 1
                next_token = model.get_next_sym()

            encoder.write_seq(encoding_probabilities)
            model.reset()
        model.close()
        encoder.close()


def decode(input_path, out_file, model, precision):
    ans_reader = ByteReader(input_path + '/ans.lm')
    ans_reader.open()
    ans_reader.go_to_end()
    decoder = rANSDecoder(ans_reader)
    decoder.open()

    ans_reader.seek(util.read_expanding_num(ans_reader, False))
    num_chunks = util.read_expanding_num(ans_reader, True)
    pos_table_ptr = ans_reader.tell()
    curr_end = -1

    with open(out_file, 'w') as out, open(input_path + '/unk.lm') as unk_reader:
        for i in range(num_chunks):
            ans_reader.set_mode(True)
            ans_reader.seek(pos_table_ptr)
            prev_end = curr_end
            curr_end = util.read_expanding_num(ans_reader)
            pos_table_ptr = ans_reader.tell()
            ans_reader.seek(curr_end)
            ans_reader.set_mode(False)

            while ans_reader.tell() >= prev_end:
                next_token_encoded = decoder.decode_token(model.predict())

                if next_token_encoded is None or model.is_unk(next_token_encoded):
                    next_token = util.read_expanding_string(unk_reader, True).decode('utf-8')
                    if next_token_encoded is None:
                        next_token_encoded = model.encode(next_token)
                else:
                    next_token = model.decode(next_token_encoded)

                model.update(next_token_encoded)
                out.write(next_token)


def main():
    encode('../test/full/in/1.txt', '../test/full/out/0', None, 14, 10000)

main()
