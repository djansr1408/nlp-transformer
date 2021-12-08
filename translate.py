import torch
from torchtext.data import Example
import argparse

from utils.data_preparing import get_data_loaders, tokenize_trg
from utils.helper import form_subsequent_mask
from config import TRANSFORMER_CONFIG
import utils.constants as C
from models import Transformer


def translate_sentence(model, src_seq, src_pad_token_id, trg_tokenizer, device, beam_size):
    """
        src_seq - Input of size [1, seq_len], batch with just one example
    """
    src_seq = src_seq.repeat(beam_size, 1)
    src_mask = (src_seq != src_pad_token_id).unsqueeze(1).unsqueeze(2).to(device)  # (batch_size, seq_len) -> (batch_size, 1, seq_len)
    enc_output = model.encode(src_seq, src_mask)
    
    trg_start_token_id = trg_tokenizer.vocab.stoi[C.SOS_WORD]
    trg_pad_token_id = trg_tokenizer.vocab.stoi[C.PAD_WORD]
    
    trg_seq = torch.full((beam_size, C.MAX_SEQ_LEN), fill_value=trg_pad_token_id, dtype=torch.long, device=device)
    trg_seq[:, 0] = trg_start_token_id

    trg_log_probs = torch.zeros((beam_size, 1), device=device)
    
    for step in range(1, C.MAX_SEQ_LEN):
        trg_mask = (trg_seq[:, :step] != trg_pad_token_id).unsqueeze(-2).to(device) & form_subsequent_mask(step, device=device)
        trg_mask = trg_mask.unsqueeze(1).to(device)

        dec_output, scores = model.decode(enc_output, trg_seq[:, :step], trg_mask, src_mask)
        probs, indices = scores[:, -1, :].topk(beam_size)   # probs shape: (batch_size=beam_size, beam_size), indices shape: (batch_size=beam_size, beam_size)

        if step == 1:
            trg_log_probs = torch.log(probs[0, :]).view(beam_size, 1)
            trg_seq[:, step] = indices[0, :].squeeze()
            continue

        trg_log_probs = torch.log(probs) + trg_log_probs        
        trg_log_probs, trg_possible_indices = trg_log_probs.view(-1).topk(beam_size) # trg_log_probs shape: (beam_size, )
        
        rows, cols = trg_possible_indices // beam_size, trg_possible_indices % beam_size

        trg_seq[:, :step] = trg_seq[rows, :step]
        trg_seq[:, step] = indices[rows, cols].squeeze() 

    translated_sentences = []
    for i in range(trg_seq.shape[0]):
        sent = ' '.join(trg_tokenizer.vocab.itos[token] for token in trg_seq[i])
        translated_sentences.append(sent)
        
    return translated_sentences


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser(description='Arguments for sentence translation.')
    args_parser.add_argument("--sentence", type=str)
    args_parser.add_argument("--beam_size", type=int, default=5)

    args = args_parser.parse_args()

    device = torch.device(TRANSFORMER_CONFIG['device'])
    _, _, _, src_tokenizer, trg_tokenizer = get_data_loaders(dataset_path=C.DATASET_DIR, 
                                                             batch_size=TRANSFORMER_CONFIG["batch_size"], 
                                                             device=TRANSFORMER_CONFIG["device"],  
                                                             load_cached=TRANSFORMER_CONFIG['load_cached'])
    src_pad_token_id = src_tokenizer.vocab.stoi[C.PAD_WORD]
    
    sentence = args.sentence
    beam_size = args.beam_size
    
    example = Example.fromlist([sentence], fields=[('src', src_tokenizer)])
    src_sentence_tokens = example.src
    src_sentence_tokens_batch = src_tokenizer.process([src_sentence_tokens], device=device)
    
    src_vocab_size = len(src_tokenizer.vocab)
    trg_vocab_size = len(trg_tokenizer.vocab)
    model = Transformer(src_vocab_size=src_vocab_size, 
                        trg_vocab_size=trg_vocab_size, 
                        config=TRANSFORMER_CONFIG)
    model.to(device)
    model.eval()

    with torch.no_grad():
        translated_sentences = translate_sentence(model, src_sentence_tokens_batch, src_pad_token_id, trg_tokenizer, device, beam_size)

    print(translated_sentences)

