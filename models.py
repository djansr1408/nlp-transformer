import torch
import torch.nn as nn
import math
from typing import List


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
    
    def __call__(self, Q, K, V, mask=None):
        """
            Q - Tensor of shape: (batch_size x n_heads x N x d_k) representing Query projection of the input embedding matrix (batch_size X N x emb_size)
            K - Tensor of shape: (batch_size x n_heads x N x d_k) representing Key projection of the input embedding matrix (batch_size X N x emb_size)
            V - Tensor of shape: (batch_size x n_heads x N x d_v) representing Value projection of the input embedding matrix (batch_size X N x emb_size)
            N - length of input sentence (including padding)
            emb_size - Size of input embedding for words, in original paper called d_model
        """
        d_k = K.size(1)
        K_t = torch.transpose(K, 2, 3)                          # Transpose K tensor to (batch_size x n_heads x d_k x N)
        res = torch.matmul(Q, K_t) / math.sqrt(d_k)             # (batch_size x n_heads x N x N)
        if mask is not None:
            res.masked_fill_(mask, 1e-12)                       # Apply mask to scores
        attention = nn.Softmax(dim=3)(res)                # Apply Softmax to scaled product
        attention = self.dropout(attention)        # Apply dropout to Softmax scores
        context = torch.matmul(attention, V)                    # Multiply Value with calculated attention
        

        return attention, context


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, dropout=0.1):
        """
            d_model - embedding size
            d_k - dimension of Query and Key projection matrices
            d_v - dimension of Value projection matrix
            n_heads - number of attention heads 
            Notes: In original paper dk = d_v = d_model / n_heads
        """
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.WQ = nn.Linear(d_model, n_heads * d_k)  # Weights WQ multiplies input and results with Query (Q), contains weights for all head attentions (for the ease of computation)
        self.WK = nn.Linear(d_model, n_heads * d_k)  # Weights WK multiplies input and results with Key (K)
        self.WV = nn.Linear(d_model, n_heads * d_v)  # Weights WV multiplies input and results with Value (V)
        
        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.w_linear = nn.Linear(n_heads * d_v, d_model) # Weights of linear layer after scaled dot product

    def __call__(self, Q, K, V, mask):
        """
            X - Tensor of shape: (batch_size, N, d_model)
            Note: d_q = d_k
        """
        batch_size = Q.size(0)

        Qp = self.WQ(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) # (batch_size, N, n_heads * d_k) -> (batch_size, N, n_heads, d_k) -> (batch_size, n_heads, N, d_k)
        Kp = self.WK(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) # (batch_size, N, n_heads * d_k) -> (batch_size, N, n_heads, d_k) -> (batch_size, n_heads, N, d_k)
        Vp = self.WV(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2) # (batch_size, N, n_heads * d_v) -> (batch_size, N, n_heads, d_v) -> (batch_size, n_heads, N, d_v)
        
        attention, context = self.attention(Qp, Kp, Vp, mask)
        context_concatenated = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)  # (batch_size, n_heads, N, d_k) -> (batch_size, N, n_heads, d_k) -> (batch_size, N, n_heads * d_k)

        output = self.w_linear(context_concatenated)

        return output


class PositionWiseFCNet(nn.Module):
    def __init__(self, input_size, inner_layer_size, output_size, dropout=0.1):
        super(PositionWiseFCNet, self).__init__()
        self.w_1 = nn.Linear(input_size, inner_layer_size)
        self.w_2 = nn.Linear(inner_layer_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
    
    def __call__(self, X):
        """
            X of shape: (batch_size, N, d_model)
        """
        output = self.dropout(self.relu(self.w_1(X)))
        output = self.w_2(output)

        return output


class EncoderPart(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, inner_layer_size, dropout=0.1):
        super(EncoderPart, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, d_k, d_v, n_heads, dropout)
        self.fc = PositionWiseFCNet(input_size=d_model, inner_layer_size=inner_layer_size, output_size=d_model, dropout=dropout)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
    
    def __call__(self, X, mask):
        """
            X of shape: (batch_size, N, d_model)
            mask of shape: (batch_size, 1, N)
            Returns: output of shape (batch_size, N, d_model)
        """
        output = self.multi_head_attention(X, X, X, mask)
        output = self.norm_1(output + X)

        residual = output
        output = self.fc(output)
        output = self.norm_2(output + residual)

        return output


class Encoder(nn.Module):
    def __init__(self, num_parts, d_model, d_k, d_v, n_heads, inner_layer_size, dropout):
        super(Encoder, self).__init__()
        self.encoder_parts = nn.ModuleList([EncoderPart(d_model, d_k, d_v, n_heads, inner_layer_size, dropout) for _ in range(num_parts)])
    
    def __call__(self, X, mask=None):
        """
            X of shape: (batch_size, N, d_model)
            mask of shape: (batch_size, 1, N)
            Returns: output of shape (batch_size, N, d_model)
        """
        output = X
        for enc_part in self.encoder_parts:
            output = enc_part(output, mask)

        return output


class DecoderPart(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, inner_layer_size, dropout=0.1):
        super(DecoderPart, self).__init__()
        self.masked_multi_head_attention = MultiHeadAttention(d_model, d_k, d_v, n_heads, dropout)
        self.multi_head_attention = MultiHeadAttention(d_model, d_k, d_v, n_heads, dropout)
        self.fc = PositionWiseFCNet(input_size=d_model, inner_layer_size=inner_layer_size, output_size=d_model, dropout=dropout)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)
    
    def __call__(self, inputs, encoder_outputs, src_mask, trg_mask):
        """
            X of shape: (batch_size, N, d_model)
            Returns: output of shape (batch_size, N, d_model)
        """
        output = self.masked_multi_head_attention(inputs, inputs, inputs, mask=trg_mask)
        output = self.norm_1(output + inputs)
        residual = output

        output = self.multi_head_attention(Q=output, K=encoder_outputs, V=encoder_outputs, mask=src_mask)
        output = self.norm_2(output + residual)
        residual = output

        output = self.fc(output)
        output = self.norm_3(output + residual)

        return output


class Decoder(nn.Module):
    def __init__(self, num_parts, d_model, d_k, d_v, n_heads, inner_layer_size, dropout):
        super(Decoder, self).__init__()
        self.decoder_parts = nn.ModuleList([DecoderPart(d_model, d_k, d_v, n_heads, inner_layer_size, dropout) for _ in range(num_parts)])

        self.w_linear = nn.Linear(n_heads * d_v, d_model)

    def __call__(self, X, encoder_output, src_mask, trg_mask):
        output = X
        for dec_part in self.decoder_parts:
            output = dec_part(output, encoder_output, src_mask, trg_mask)

        return output


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, config):
        super(Transformer, self).__init__()
        self.src_word_embedding = nn.Embedding(num_embeddings=src_vocab_size, embedding_dim=config['d_model']).float()
        self.src_positional_embedding = PositionalEncoding(max_position_num=1000, d_model=config['d_model'], dropout=0.1, device=config['device'])

        self.trg_word_embedding = nn.Embedding(num_embeddings=trg_vocab_size, embedding_dim=config['d_model']).float()
        self.trg_positional_embedding = PositionalEncoding(max_position_num=1000, d_model=config['d_model'], dropout=0.1, device=config['device'])
        
        
        self.encoder = Encoder(num_parts=config['num_parts_encoder'], 
                                d_model=config['d_model'], 
                                d_k=config['d_k'], 
                                d_v=config['d_v'], 
                                n_heads=config['n_heads'], 
                                inner_layer_size=config['inner_layer_size'], 
                                dropout=config['dropout'])
        
        self.decoder = Decoder(num_parts=config['num_parts_decoder'], 
                               d_model=config['d_model'], 
                               d_k=config['d_k'], 
                               d_v=config['d_v'], 
                               n_heads=config['n_heads'], 
                               inner_layer_size=config['inner_layer_size'], 
                               dropout=config['dropout'])

        self.w_linear = nn.Linear(config['d_model'], trg_vocab_size, bias=False)

        # self.softmax = nn.Softmax()


    def __call__(self, src_batch, trg_batch, src_mask, trg_mask):
        """
            src_batch: shape (batch_size, seq_len)
            trg_batch: shape (batch_size, seq_len) 
            src_mask: shape (batch_size, 1, seq_len)
            trg_mask: shape (batch_size, seq_len, seq_len)
        """
        encoder_input = self.src_word_embedding(src_batch)
        encoder_input = self.src_positional_embedding(encoder_input)
        encoder_output = self.encoder(encoder_input, src_mask)

        decoder_input = self.trg_word_embedding(trg_batch)              # shape (batch_size, seq_len, d_model)
        decoder_input = self.trg_positional_embedding(decoder_input)    # shape (batch_size, seq_len, d_model)
        
        decoder_output = self.decoder(decoder_input, encoder_output, src_mask, trg_mask)  # shape (batch_size, seq_len, d_model)
        logits = self.w_linear(decoder_output)                                  # shape (batch_size, seq_len, trg_vocab_size)
        # scores = self.softmax(logits)

        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, max_position_num, d_model, dropout=0.1, device="cpu"):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        position_ids = torch.arange(start=0, end=max_position_num, step=1).unsqueeze(1) # shape (max_position_num, 1)
        freq_ids = torch.arange(start=0, end=d_model, step=1)
        freqs = torch.pow(10000, exponent=-2 * freq_ids / d_model).unsqueeze(1).transpose(1, 0) # shape (1, d_model)
        self.sinusoid_table = torch.zeros(max_position_num, d_model, device=device)
        self.sinusoid_table[:, 0::2] = torch.sin(position_ids * freqs[:, 0::2])
        self.sinusoid_table[:, 1::2] = torch.cos(position_ids * freqs[:, 1::2])

        self.register_buffer('positional_encodings', self.sinusoid_table) # This is to be saved to state_dict but not to be trained as other parameters
        
    def __call__(self, X):
        """
            X is embedding batch of shape (batch_size, num_tokens, d_model)
        """
        return self.dropout(X + self.sinusoid_table[:X.size(1), :])



# if __name__ == "__main__":
#     batch_size = 10
#     N = 5
#     d_model = 512
#     X = torch.randn(batch_size, N, d_model)

#     pos = PositionalEncoding(max_position_num=50, d_model=d_model, dropout=0.1)
#     res= pos(X)
#     print(res.shape)




# if __name__ == "__main__":
#     # a = ScaledDotProductAttention()

#     num_parts = 6
#     N = 50
#     d_model = 512
#     d_k = 64
#     d_v = 64
#     n_heads = 8
#     inner_layer_size = 2048
#     batch_size = 32

#     # Q = torch.randn(batch_size, n_heads, N, d_k)
#     # K = torch.randn(batch_size, n_heads, N, d_k)
#     # V = torch.randn(batch_size, n_heads, N, d_k)
#     X = torch.randn(batch_size, N, d_model)

#     # b = MultiHeadAttention(d_model=d_model, d_k=d_k, d_v=d_v, n_heads=n_heads)(X, None)
#     # print("b size: ", b.size())

#     # c = PositionWiseFCNet(input_size=d_model, inner_layer_size=2048, output_size=d_model)
#     # c(b)

#     # d = EncoderPart(N, d_model, d_k, d_v, n_heads, inner_layer_size)
#     # d(X)

#     e = Encoder(num_parts, d_model, d_k, d_v, n_heads, inner_layer_size)
#     encoder_output = e(X)
#     print("encoder output size: ", encoder_output.size())

#     # d = DecoderPart(N, d_model, d_k, d_v, n_heads, inner_layer_size)
#     # d(X, encoder_output, mask=None)

#     d = Decoder(num_parts, d_model, d_k, d_v, n_heads, inner_layer_size)
#     decoder_output = d(X, encoder_output, mask=None)
#     print("Decoder output: ", decoder_output.size)




