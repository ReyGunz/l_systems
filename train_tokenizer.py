import os
import sentencepiece as spm
from torch.nn.utils.rnn import pad_sequence

def check_file_exists(filename):
    # Check if the file exists in the current directory
    if os.path.isfile(filename):
        print(f"File '{filename}' exists in the current directory.")
        return True
    else:
        print(f"File '{filename}' does not exist in the current directory.")
        return False

class SentencePieceTokenizer:
    def __init__(self, vocab_size, model_type='bpe'):
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.sp_model = None

    def build_vocab(self, texts, model_prefix='spm'):
        if(not check_file_exists(model_prefix + '.model')):
            # Write texts to a temporary file
            with open('temp.txt', 'w', encoding='utf-8') as f:
                for text in texts:
                    f.write(text + '\n')

            # Train SentencePiece model
            spm.SentencePieceTrainer.train(input='temp.txt', model_prefix=model_prefix,
                                            vocab_size=self.vocab_size, model_type=self.model_type)

            # Load model
        else:
            self.sp_model = spm.SentencePieceProcessor()
            self.sp_model.load(f'{model_prefix}.model')

    
    def tokenize(self, text):
        return self.sp_model.encode(text, out_type=int)

    def detokenize(self, tokens):
        return self.sp_model.decode(tokens)

def collate_batch(batch):
    return pad_sequence(batch, batch_first=True, padding_value=0)

data_files = [
                "stochastic_tree1_samples",
                "stochastic_tree2_samples",
                "stochastic_tree3_samples",
                "stochastic_tree4_samples",
            ]

texts = []
for file in data_files:
    texts += open('data/' + file + '.txt').readlines()

vocab_size = 17
tokenizer = SentencePieceTokenizer(vocab_size=vocab_size, model_type='bpe')
tokenizer.build_vocab(texts)