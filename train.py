from dotenv import load_dotenv
load_dotenv()

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from utils.data_preparing import get_data_loaders
import utils.constants as C
from utils.helper import load_checkpoint, form_subsequent_mask
from models import Transformer

from config import TRANSFORMER_CONFIG

import matplotlib.pyplot as plt
import time

torch.manual_seed(0)

class CustomOptimizer():
    def __init__(self, base_optimizer, **config):
        model_dir = os.path.join(C.STORAGE_DIR, config["model_alias"])
        checkpoint = load_checkpoint(model_dir)
        self.num_steps = 0
        self.warmup_steps = config["warmup_steps"]
        self.d_model = config["d_model"]
        self.base_optimizer_ = base_optimizer

        if checkpoint is not None:
            self.num_steps = checkpoint["num_steps"]
            self.base_optimizer_.load_state_dict(checkpoint["optimizer_state_dict"])

    def step(self):
        self.num_steps += 1
        num_steps_scaled = (self.num_steps // 96) + 1
        lr = self.d_model**(-0.5) * min(num_steps_scaled**(-0.5), num_steps_scaled * self.warmup_steps**(-1.5))
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
    scores = nn.Softmax(dim=-1)(logits)

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
    else:
        loss = F.cross_entropy(logits.view(-1, trg_vocab_size), target.squeeze(), ignore_index=trg_pad_token_id, reduction='mean')
    
    return loss


def train_and_val_epoch(model, optimizer, data_loader, device, src_pad_token_id, trg_pad_token_id, training=True, smoothing=True, log_every_n_steps=50):
    train_losses = []
    model.train()
    for batch_id, batch in enumerate(data_loader):
        src_batch, src_batch_mask, trg_batch_input, trg_batch_output, trg_batch_mask_input = prepare_batch(batch, src_pad_token_id, trg_pad_token_id, device)

        optimizer.zero_grad()

        logits, scores = model(src_batch, trg_batch_input, src_batch_mask, trg_batch_mask_input)

        loss = calculate_loss(logits, trg_batch_output, trg_pad_token_id, smoothing)
        train_losses.append(loss.item())

        if training:
            loss.backward()
            lr = optimizer.step()

        if optimizer.num_steps % log_every_n_steps == 0:
            train_loss = np.mean(train_losses)            
            val_loss = evaluate(model, val_loader, src_pad_token_id, trg_pad_token_id, device, smoothing)
            writer.add_scalars(f'Loss/', {
                                            'train': train_loss,
                                            'val': val_loss,
                                        }, optimizer.num_steps)
            train_losses = []
            model.train()
        writer.add_scalar("Batch Loss/train", loss, optimizer.num_steps)
    
    save_path = os.path.join(model.model_dir, f"trained_model_steps_{optimizer.num_steps}.pt")
    torch.save({
                'num_steps': optimizer.num_steps, 
                'model_state_dict': model.state_dict(), 
                'optimizer_state_dict': optimizer.base_optimizer_.state_dict(), 
                'train_loss': train_loss
            }, save_path)
    return train_loss, val_loss, save_path
    

def evaluate(model, val_loader, src_pad_token_id, trg_pad_token_id, device, smoothing):
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            src_batch, src_batch_mask, trg_batch_input, trg_batch_output, trg_batch_mask_input = prepare_batch(batch, src_pad_token_id, trg_pad_token_id, device)
            logits, _ = model(src_batch, trg_batch_input, src_batch_mask, trg_batch_mask_input)
            loss = calculate_loss(logits, trg_batch_output, trg_pad_token_id, smoothing)
            val_losses.append(loss.item())
    
    return np.mean(val_losses)


def prepare_batch(batch, src_pad_token_id, trg_pad_token_id, device, src_tokenizer=None, trg_tokenizer=None):
    src_batch = batch.src.to(device)
    trg_batch_input, trg_batch_output = batch.trg[:, :-1].to(device), batch.trg[:, 1:].to(device)
    
    src_batch_mask = (src_batch != src_pad_token_id).unsqueeze(1).unsqueeze(2).to(device)  # (batch_size, seq_len) -> (batch_size, 1, seq_len)
    trg_batch_mask_input = (trg_batch_input != trg_pad_token_id).unsqueeze(-2).to(device) & form_subsequent_mask(trg_batch_input.size(1), device=device)  # (batch_size, seq_len) -> 
    trg_batch_mask_input = trg_batch_mask_input.unsqueeze(1).to(device)

    return src_batch, src_batch_mask, trg_batch_input, trg_batch_output, trg_batch_mask_input


if __name__ == "__main__":
    device = torch.device(TRANSFORMER_CONFIG['device'])

    train_loader, val_loader, test_loader, src_tokenizer, trg_tokenizer = get_data_loaders(dataset_path=C.DATASET_DIR, 
                                                                                           batch_size=TRANSFORMER_CONFIG["batch_size"], 
                                                                                           device=device, 
                                                                                           load_cached=TRANSFORMER_CONFIG['load_cached'])
    
    print(len(train_loader))
    print(len(val_loader))
    print(len(test_loader))
    quit()

    src_vocab_size = len(src_tokenizer.vocab)
    trg_vocab_size = len(trg_tokenizer.vocab)
    src_pad_token_id = src_tokenizer.vocab.stoi[C.PAD_WORD]
    trg_pad_token_id = trg_tokenizer.vocab.stoi[C.PAD_WORD]

    model = Transformer(src_vocab_size=src_vocab_size, trg_vocab_size=trg_vocab_size, config=TRANSFORMER_CONFIG)
    model.to(device)

    custom_optimizer = CustomOptimizer(base_optimizer=optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09), **TRANSFORMER_CONFIG)
    
    writer = SummaryWriter()
    for epoch_id in range(TRANSFORMER_CONFIG['num_epochs']):
        train_loss, val_loss, save_path = train_and_val_epoch(model=model, 
                                                              optimizer=custom_optimizer, 
                                                              data_loader=train_loader, 
                                                              device=device, 
                                                              src_pad_token_id=src_pad_token_id, 
                                                              trg_pad_token_id=trg_pad_token_id, 
                                                              training=True, 
                                                              smoothing=TRANSFORMER_CONFIG["smoothing"], 
                                                              log_every_n_steps=TRANSFORMER_CONFIG["log_every_n_steps"])

