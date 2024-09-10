import math

import torch
import torch.nn as nn


class InputEmbeddings(nn.Module):

    def __init__(self, model_dimension: int, vocab_size: int) -> None:

        super().__init__()
        self.model_dimension = model_dimension
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, model_dimension)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.model_dimension)
    
class PositionalEmbedding(nn.Module):

    def __init__(self, model_dimension: int, sequence_length, dropout: float) -> None:
        """
        sequence_length: Maximum length of the sentence.
        model_dimension: dimension of each vector embedding
        """
        super().__init__()
        self.model_dimension = model_dimension
        self.sequence_length = sequence_length
        self.dropout = nn.Dropout(dropout)

        # This is a new way to add positional encoding, but this functions the same way as described in the paper. 
        positional_encoding = torch.zeros(sequence_length, model_dimension)

        position = torch.arange(0, sequence_length, dtype=torch.float32).unsqueeze(1)
        division_term = torch.exp(torch.arange(0, model_dimension).float() * (math.log(10000.0) / model_dimension))

        # Apply the sine and cosine to even and odd positions
        positional_encoding[:, 0::2] = torch.sin(position * division_term)
        positional_encoding[:, 1::2] = torch.cos(position * division_term)

        # Add the batch dimension to the positional embedding
        positional_encoding = positional_encoding.unsqueeze(0) 

        # When a tensor that has to be saved with the model, but not as a learned parameter, you save it as a buffer. 
        # This will be saved along with the state of the model.
        self.register_buffer(positional_encoding)

    def forward(self, x):
        x = x + (self.positional_encoding[:, :x.shape[1], :]).required_grad_(False)
        return self.dropout(x)
    

class LayerNormalization(nn.Module):

    def __init__(self, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplicative
        self.bias = nn.Parameter(torch.ones(1)) # Additive

    def forward(self, x):

        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return self.alpha * (x - mean) / (std + self.eps) * self.bias
    
class FeedForwardBlock(nn.Module):

    def __init__(self, model_dimension: int, ff_dimension: int, dropout: float) -> None:
        """
        This class defines the Feed Forward block from the paper.
        
        paper config: 
        model_dimension = 512
        ff_dimension = 2048

        transformation: 
        (Batch, sequence_length, model_dimension) --> (Batch, sequence_length, ff_dimension) --> (Batch, sequence_length, model_dimension)
        """
        super().__init__()
        self.linear_1 = nn.Linear(model_dimension, ff_dimension)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(ff_dimension, model_dimension)

    def forward(self, x):
        
        # x = self.linear_1(x)
        # x = torch.relu(x)
        # x = self.dropout(x)
        # x = self.linear_2(x)

        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, model_dimension: int, num_of_heads: int, dropout: float) -> None: 
        super().__init__()
        self.model_dimension = model_dimension
        self.num_of_heads = num_of_heads
        
        assert model_dimension % num_of_heads == 0, "model_dimension must be divisible by num of heads" # Replace this with try / except 

        self.d_k = model_dimension // num_of_heads
        self.W_q = nn.Linear(model_dimension, model_dimension)
        self.W_k = nn.Linear(model_dimension, model_dimension)
        self.W_v = nn.Linear(model_dimension, model_dimension)

        self.W_o = nn.Linear(model_dimension, model_dimension)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def calculate_attention_score(query, key, value, mask, dropout: nn.Dropout):

        # dimension of the key matrix
        d_k = query.shape[-1]

        # (Batch, num_of_heads, seq_len, d_k) --> (Batch, num_of_heads, seq_len, seq_len)
        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_score.masked_fill_(mask==0, -1e9)
        
        attention_scores = attention_scores.softmax(dim=-1) # (Batch, num_of_heads, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores
    
    def forward(self, q, k , v, mask):

        query = self.W_q(q) # (Batch, Seq_len, model_dimension) --> (Batch, Seq_len, model_dimension)
        key = self.W_k(k) # (Batch, Seq_len, model_dimension) --> (Batch, Seq_len, model_dimension)
        value = self.W_v(v) # (Batch, Seq_len, model_dimension) --> (Batch, Seq_len, model_dimension)


        # (Batch, seq_len, model_dimension) --> (Batch, seq_len, num_of_heads, d_k) --> (Batch, num_heads, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.num_of_heads, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.num_of_heads, self.d_k).transpose(1,2)
        value = key.view(value.shape[0], value.shape[1], self.num_of_heads, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttentionBlock.calculate_attention_score(query, key, value, mask. self.dropout)


        # (Batch, num_of_heads, seq_len, d_k) --> (Batch, seq_len, num_of_heads, d_k) --> (Batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.num_of_heads * self.d_k)

        # (Batch, seq_len, model_dimension) --> (Batch, seq_len, model_dimension)
        return self.W_o(x)

class ResidualConnection(nn.Module):

    def __init__(self, dropout: float) -> None: 
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x , previous_layer):
        return x + self.dropout(previous_layer(self.norm(x)))


class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x , x, src_mask)) # x is the query, key, value
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    

class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)


class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_foward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_foward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, target_mask):

        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, target_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)

        return x
    