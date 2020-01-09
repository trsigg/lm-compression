class LanguageModel:
    def __init__(self):
        self.input = None

    def get_probs(self, input_file, precision):
        raise NotImplemented

    def update(self, token):
        raise NotImplemented

    def encode(self, token):
        raise NotImplemented

    def is_unk(self, token):
        raise NotImplemented

    def reset(self):
        raise NotImplemented

    def open(self, input_path):
        self.input = open(input_path)

    def close(self):
        self.input.close()
