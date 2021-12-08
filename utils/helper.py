import re
import os
import torch


def load_checkpoint(dir, training=False):
    filename_tuples = [(int(re.findall(r'\d+',f)[0]), f) for f in os.listdir(dir) if f.endswith(".pt")]
    if len(filename_tuples) > 0:
        latest_checkpoint_filename = sorted(filename_tuples, key=lambda x: x[0], reverse=True)[0][1]
        latest_checkpoint_filepath = os.path.join(dir, latest_checkpoint_filename)
        if training:
            checkpoint = torch.load(latest_checkpoint_filepath)
        else:
            checkpoint = torch.load(latest_checkpoint_filepath, map_location="cpu")
        return checkpoint
    else: return None


def form_subsequent_mask(seq_len, device):
    mask = (1 - torch.triu(torch.ones((1, seq_len, seq_len), device=device), diagonal=1)).bool()
    return mask