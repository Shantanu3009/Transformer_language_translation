from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import warnings
from tqdm import tqdm
import os
from pathlib import Path

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):

    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)

    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        out =model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        prob = model.project(out[:,-1])

        _, next_word =torch.max(prob, dim = 1)

        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim= 1)

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, writer, num_examples =5):
    count=0
    model.eval()
    console_width = 80
    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input =batch['encoder_input'].to(device)
            encoder_mask =batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1 , "Validation Batch size should be one only"

            model_output = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_output.detach().cpu().numpy())
            print_msg("-"*console_width)
            print_msg(f"SOURCE TEXT: {source_text}")
            print_msg(f"TARGET TEXT: {target_text}")
            print_msg(f"PREDICTED TEXT: {model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break





def get_all_sentences(ds, lang):  #(ds, 'en')
    for item in ds:
        #eg. translation row 1: { "en": "Source: Project Gutenberg", "it": "Source: www.liberliber.it/Audiobook available here" }
        yield item['translation'][lang] 

def get_or_build_tokenizer(config, ds, lang): # raw dataset and src or tgt language
    tokenizer_path = Path(config['tokenizer_file'].format(lang)) #tokenizer_en.json for english or tokenizer_it.json for italian
    if not Path.exists(tokenizer_path): #IFtokenizer filefor that language doesn't exist
        
        print(f"Creating a new tokenizer file for: {lang}")
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]")) # UNKtokenfor unknown words
        tokenizer.pre_tokenizer = Whitespace() #use whitespace for splitting text into tokens
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer) #Training tokenizer on all sentences present train+val
        tokenizer.save(str(tokenizer_path))

    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer

def get_ds(config):

    # load_dataset("opus_books", "en-it") 
    ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src']) #tokenizer_en.json
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt']) #tokenizer_it.json

    train_ds_size = int(0.9 * len(ds_raw))  
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size]) #90% training 10% validation

    train_ds = BilingualDataset( train_ds_raw, config['seq_len'], config['lang_src'], tokenizer_src, config['lang_tgt'], tokenizer_tgt)
    val_ds = BilingualDataset( val_ds_raw, config['seq_len'], config['lang_src'], tokenizer_src, config['lang_tgt'], tokenizer_tgt)

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}') #checking the max len we input is lower thanthe max transformer config length
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size = config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size = 1, shuffle=True) # batch size only 1 as we want to validate only one sentence at a time

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):

    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'],d_model=config['d_model'])
    return model

def train_model(config):
    device = "cuda" if torch.cuda.is_available() else "cpu" #compute unified device architecture (For GPUs)
    device = torch.device(device)
    print(f"Using device: {device}")

    #Parents here mean suppose I want to create a dir like /a/b/c but /a and a/b don't exist it will also create these 
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)  #Create this directory opus_books_weights and it's parents if it doesn't exist

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    model =get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps = 1e-9) #Makes sure all learnable parameters (like weights and biases) in the model are updated

    initial_epoch = 0  
    global_step = 0

    preload = config['preload'] #latest
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}') #transformer_
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict']) #We saveall these below states at the end of our model run
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    #label_smoothing=0.1 means that 10% of the probability mass will be spread over the other classes, reducing the probability of the correct class by a small amount. This can help prevent the model from becoming overconfident and improve generalization.
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator: #iterates over batches of data

            encoder_input = batch['encoder_input'].to(device) # (B, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # proj_output.view(-1, tokenizer_tgt.get_vocab_size()) reshapes proj_output from (Batch, seq_len, tgt_vocab_size) to (Batch * seq_len, tgt_vocab_size)
            # label.view(-1) reshapes label from (Batch, seq_len) to (Batch * seq_len)
            # The target tensor (labels) should be a 1D tensor of length N, where each value corresponds to the true class index for the corresponding sample.
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))  #compute the loss for each of the Batch * seq_len samples.
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"}) #display metrics such as loss at the end of bar

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of every epoch
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)
        
        # Save the model at the end of every epoch
        print(f"Saving Model at Epoch: {epoch} & Global Step: {global_step} ")
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)