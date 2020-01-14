from LanguageModel import LanguageModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn.functional as F


class GPT2Model(LanguageModel):
    UNK = 50256

    def __init__(self):
        self.input = None

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.reset()

    def predict(self):
        return F.softmax(self.output).tolist()

    def update(self, token):
        if type(token) != list:
            token = [token]

        for t in token:
            self.output, self.past = self.model(torch.tensor([t]),
                                                past=self.past)
        self.output = self.output[0, :]

    def get_next_sym(self):
        try:
            return next(self.input)
        except StopIteration:
            return None

    def encode(self, token):
        return self.tokenizer.encode(token)

    def decode(self, token):
        return self.tokenizer.decode(token, clean_up_tokenization_spaces=False)

    def is_unk(self, token):
        return token == self.UNK

    def reset(self):
        self.past = None
        self.update(self.UNK)

    def open(self, input_path):
        with open(input_path, 'r') as in_file:
            self.input = iter(self.tokenizer.encode(in_file.read()))
        self.reset()

    def close(self):
        self.input = None
