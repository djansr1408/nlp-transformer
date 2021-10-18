import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from utils.data_preparing import get_data_loaders
import utils.constants as C
from models import Transformer

from config import TRANSFORMER_CONFIG

import matplotlib.pyplot as plt

def form_subsequent_mask(seq_len, device):
    mask = (1 - torch.triu(torch.ones((1, seq_len, seq_len), device=device), diagonal=1)).bool()
    return mask

class CustomOptimizer():
    def __init__(self, base_optimizer, d_model, warmup_steps):
        self.num_steps = 0
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.base_optimizer_ = base_optimizer

    def step(self):
        self.num_steps += 1
        lr = self.d_model**(-0.5) * min(self.num_steps**(-0.5), self.num_steps * self.warmup_steps**(-1.5))
        for param_group in self.base_optimizer_.param_groups:
            param_group['lr'] = lr
        self.base_optimizer_.step()

        return lr

    def zero_grad(self):
        self.base_optimizer_.zero_grad()

def calculate_loss(logits, target, trg_pad_token_id, smoothing=True, eps=0.1):
    """
        Scores of shape: (batch_size, trg_seq_len, trg_vocab_size)
        Target of shape: (batch_size, trg_seq_len)
    """
    scores = nn.Softmax()(logits)
    trg_vocab_size = scores.size(2)
    scores = scores.view(-1, trg_vocab_size)
    
    target = target.contiguous().view(-1, 1)
    
    if smoothing:
        one_hot_target = torch.zeros_like(scores).scatter(1, target, 1)         # 1st argument in scatter relates to dim (axis along which will impute src value (in this case value of 1))
                                                                                # shape (batch_size * trg_seq_len x trg_vocab_size)
        
        one_hot_target_smoothed = (1 - eps) * one_hot_target + eps * (1 - one_hot_target) / trg_vocab_size
        no_pad_mask = target == trg_pad_token_id
        one_hot_target_smoothed.masked_fill_(no_pad_mask, 0.)

        loss = F.kl_div(torch.log(scores), one_hot_target_smoothed, reduction='batchmean') # batchmean behaves as original math definition of KL Divergence
        print("kl_div loss: ", loss)
    else:
        loss = F.cross_entropy(scores, target.squeeze(), ignore_index=trg_pad_token_id, reduction='mean')
        print("cross entropy loss: ", loss)
    
    return loss


def train_or_val_epoch(model, optimizer, data_loader, device, src_pad_token_id, trg_pad_token_id, training=True):
    if training is True:
        model.train()
    else:
        model.eval()
    
    for batch_id, batch in enumerate(data_loader):
        src_batch, src_batch_mask, trg_batch_input, trg_batch_output, trg_batch_mask_input = prepare_batch(batch, src_pad_token_id, trg_pad_token_id, device)

        optimizer.zero_grad()

        logits = model(src_batch, trg_batch_input, src_batch_mask, trg_batch_mask_input)
        loss = calculate_loss(logits, trg_batch_output, trg_pad_token_id, smoothing=False)
        
        writer.add_scalar("Loss/train", loss, optimizer.num_steps)

        if training:
            loss.backward()
            optimizer.step()

        

def prepare_batch(batch, src_pad_token_id, trg_pad_token_id, device):
    src_batch = batch.src.to(device)
    trg_batch_input, trg_batch_output = batch.trg[:, :-1].to(device), batch.trg[:, 1:].to(device)
    
    src_batch_mask = (src_batch != src_pad_token_id).unsqueeze(1).unsqueeze(2).to(device)  # (batch_size, seq_len) -> (batch_size, 1, seq_len)
    trg_batch_mask_input = (trg_batch_input != trg_pad_token_id).unsqueeze(-2).to(device) & form_subsequent_mask(trg_batch_input.size(1), device=device)  # (batch_size, seq_len) -> 
                                                                                                                                                                    # (batch_size, 1, seq_len) & (1, seq_len, seq_len) => 
                                                                                                                                                                    # (batch_size, seq_len, seq_len)
    trg_batch_mask_input = trg_batch_mask_input.unsqueeze(1).to(device)

    return src_batch, src_batch_mask, trg_batch_input, trg_batch_output, trg_batch_mask_input


if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(TRANSFORMER_CONFIG['device'])
    print('device: ',device)

    train_loader, val_loader, test_loader, src_tokenizer, trg_tokenizer = get_data_loaders(dataset_path=TRANSFORMER_CONFIG["dataset_path"], device=device) # TODO: filter sentences longer than max_len

    src_vocab_size = len(src_tokenizer.vocab)
    trg_vocab_size = len(trg_tokenizer.vocab)

    model = Transformer(src_vocab_size=src_vocab_size, trg_vocab_size=trg_vocab_size, config=TRANSFORMER_CONFIG)
    model.to(device)

    custom_optimizer = CustomOptimizer(base_optimizer=optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09), 
                                       d_model=TRANSFORMER_CONFIG['d_model'], 
                                       warmup_steps=TRANSFORMER_CONFIG['warmup_steps'])

    print("model cuda out: ", next(model.encoder.encoder_parts[0].multi_head_attention.WQ.parameters()).is_cuda)
    src_pad_token_id = src_tokenizer.vocab.stoi[C.PAD_WORD]
    print("trg_pad_token_id: ", src_pad_token_id)

    trg_pad_token_id = trg_tokenizer.vocab.stoi[C.PAD_WORD]
    print("trg_pad_token_id trg: ", trg_pad_token_id)
    
    writer = SummaryWriter()

    for epoch_id in range(TRANSFORMER_CONFIG['num_epochs']):
        train_or_val_epoch(model=model, 
                           optimizer=custom_optimizer, 
                           data_loader=train_loader, 
                           device=device, 
                           src_pad_token_id=src_pad_token_id, 
                           trg_pad_token_id=trg_pad_token_id, 
                           training=True)


    # for batch_id, batch in enumerate(train_loader):
    #     src_batch, src_batch_mask, trg_batch_input, trg_batch_output, trg_batch_mask = prepare_batch(batch, src_pad_token_id, trg_pad_token_id, device)

    #     scores = model(src_batch, trg_batch_input, src_batch_mask, trg_batch_mask)

    #     print("Scores shape: ", scores.shape)
    #     quit(0)
    