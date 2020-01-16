from rANSEncoder import rANSEncoder
from rANSDecoder import rANSDecoder
from ByteReader import ByteReader
import util
import os


def encode(input_file, out_path, model, chunk_size):
    print('beginning encoding...')

    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    model.open(input_file)
    encoder = rANSEncoder(out_path + '/ans.lm')
    encoder.open()  # should probably enable <with> syntax for these

    encoding_probabilities = [(0, 0)] * chunk_size
    next_token = model.get_next_sym()  # maybe make iterator?

    with open(out_path + '/unk.lm', 'wb') as unk_out:
        while next_token is not None:
            tokens_encoded = 0
            model.reset()
            while next_token is not None and tokens_encoded < chunk_size:
                fs, cs, overflow = encoder.get_probs_from_dist(next_token,
                                                               model.predict())
                model.update(next_token)

                encoding_probabilities[tokens_encoded] = (fs, cs)
                if overflow or model.is_unk(next_token):
                    util.write_expanding_string(model.decode(next_token), unk_out, True)
                    util.write_expanding_num(next_token, unk_out, True)

                tokens_encoded += 1
                next_token = model.get_next_sym()

            encoder.write_seq(encoding_probabilities, tokens_encoded)
        model.close()
        encoder.close()

        print('encoding finished\n')


def decode(input_path, out_file, model):
    print('beginning decoding')

    ans_reader = ByteReader(input_path + '/ans.lm')
    ans_reader.open()
    ans_reader.go_to_end()
    decoder = rANSDecoder(ans_reader)

    ans_reader.seek(util.read_expanding_num(ans_reader, False))
    num_chunks = util.read_expanding_num(ans_reader, True)
    pos_table_ptr = ans_reader.tell()
    curr_end = -1

    with open(out_file, 'w') as out, open(input_path + '/unk.lm', 'rb') as unk_reader:
        for i in range(num_chunks):
            ans_reader.set_mode(True)
            ans_reader.seek(pos_table_ptr)
            prev_end = curr_end
            curr_end = util.read_expanding_num(ans_reader, True)
            pos_table_ptr = ans_reader.tell()
            ans_reader.seek(curr_end)

            decoder.init_chunk()
            model.reset()


            while True:
                next_token, should_continue = decoder.decode_token(
                    model.predict(), prev_end)

                if not should_continue:
                    break

                if next_token is None or model.is_unk(next_token):
                    next_token_decoded = util.read_expanding_string(unk_reader, True)
                    if next_token is None:
                        next_token = util.read_expanding_num(unk_reader, True)
                else:
                    next_token_decoded = model.decode(next_token)

                model.update(next_token)
                out.write(next_token_decoded)

    print('decoding finished\n')
