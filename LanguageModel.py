class LanguageModel:
    def __init__(self):
        self.input = None

    def predict(self):
        raise NotImplemented

    def update(self, token):
        raise NotImplemented

    def get_next_sym(self):
        raise NotImplemented

    def encode(self, token):
        raise NotImplemented

    def decode(self, token):
        raise NotImplemented

    def is_unk(self, token):
        raise NotImplemented

    def reset(self):
        raise NotImplemented

    def open(self, input_path):
        self.input = open(input_path, 'r')

    def close(self):
        self.input.close()
