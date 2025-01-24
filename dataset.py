import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    def __init__(self, ds,seq_len: int, src_lang, src_tokenizer, tgt_lang, tgt_tokenizer) -> None:
        #Using the tokenizer we create Input and Output tokens (as per transformer rules) out of the raw dataset
        super().__init__()
        self.ds = ds #either train or val dataset eg.{ "en": "CHAPTER I", "it": "PARTE PRIMA" }
        self.seq_len = seq_len # 350 words
        self.src_lang = src_lang #english
        self.tgt_lang = tgt_lang #italian
        self.src_tokenizer = src_tokenizer #tokenizer_en.json
        self.tgt_tokenizer = tgt_tokenizer #tokenizer_it.json

        self.sos_token = torch.tensor(tgt_tokenizer.token_to_id("[SOS]"), dtype = torch.int64) #PyTorch tensors of type torch.int64
        self.eos_token = torch.tensor(tgt_tokenizer.token_to_id("[EOS]"), dtype = torch.int64)
        self.pad_token = torch.tensor(tgt_tokenizer.token_to_id("[PAD]"), dtype = torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index): # fetch and process a single data sample from a dataset for use in training or evaluating
          src_tgt_pair = self.ds[index] #
          src_text = src_tgt_pair['translation'][self.src_lang]
          tgt_text = src_tgt_pair['translation'][self.tgt_lang]

          enc_input_tokens = self.src_tokenizer.encode(src_text).ids
          dec_input_tokens = self.tgt_tokenizer.encode(tgt_text).ids

          enc_pad_len = self.seq_len - len(enc_input_tokens) - 2 # "[SOS]" and "[EOS]"
          dec_pad_len = self.seq_len - len(dec_input_tokens) - 1 # "[SOS]" or "[EOS]"

          if enc_pad_len < 0 or dec_pad_len < 0: #If source or target text is too long to fit within the specified sequence length 
               raise ValueError("Sentence is too long")
            #Terminating the input in an (EOS) token signals to the encoder that when it receives that input, the output needs to be the finalized embedding. 
          encoder_input = torch.cat(
            [
                self.sos_token.unsqueeze(0),
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token.unsqueeze(0),
                torch.tensor([self.pad_token] * enc_pad_len, dtype=torch.int64),
            ]
          )
               #Right shift the target sentence when passing it to the transformer decoder.the decoder will progress by taking the tokens it emits as inputs (along with the embedding and hidden state, or using the embedding to initialize the hidden state), so before it has emitted anything it needs a token of some kind to start with. Hence, the SOS token.
          decoder_input = torch.cat(
            [
                self.sos_token.unsqueeze(0),
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_pad_len, dtype=torch.int64),
            ]
          )
               #Without an "end" token, we would have no idea when the decoder is done talking to us and continuing to emit tokens will produce gibberish.
          label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token.unsqueeze(0),
                torch.tensor([self.pad_token] * dec_pad_len, dtype=torch.int64),
            ]
          )

          assert len(encoder_input) == self.seq_len
          assert len(decoder_input) == self.seq_len
          assert len(label) == self.seq_len

          return {
               "encoder_input": encoder_input,#(seq_len) here and For a batch:(B, seq_len)
               "decoder_input": decoder_input, #(seq_len) here and For a batch:(B, seq_len)
               "encoder_mask" : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), #(1 , 1 , self.seq_len) here and For a batch: (B, 1, 1, seq_len)
               # .int() converts the boolean values to integers
               # For decoder mask should be such that a 2D matrix words (seq_len,seq_len) where words can't see the next word
               "decoder_mask" : (encoder_input != self.pad_token).unsqueeze(0).int()  & causal_mask(decoder_input.size(0)),  #(1, seq_len) & (1, 1, seq_len) = (1, seq_len, seq_len)  here and for a batch: (B, 1, seq_len, seq_len)
               "label" : label, #(seq_len) here and For a batch:(B, seq_len)
               "src_text" : src_text,
               "tgt_text" : tgt_text
          }
    
def causal_mask(size):
     mask = torch.triu(torch.ones((1, size, size)) , diagonal =1).type(torch.int) ## Returns matrix with only upper triangle having ones
     #Need to reverse it as we dont want decoder to see the next word only the prev word so upper triangle becomes 0(next word Padded) and lower triangle becomes 1
     return mask == 0 




            



