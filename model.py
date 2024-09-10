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
    
