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
    
class PositionalEncoding(nn.Module):

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
    

class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:

        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, target_mask):

        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, target_mask)

        return self.norm(x)
    

class ProjectionLayer(nn.Module):

    def __init__(self, model_dimension: int, vocab_size: int) -> None:

        super().__init__()
        self.projection = nn.Linear(model_dimension, vocab_size)

    def forward(self, x):
        # (Batch, seq_len, model_dimension) --> (Batch, seq_len, vocab_size)
        return torch.log_softmax(self.projection(x), dim=-1)
    

class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embedding: InputEmbeddings, target_embedding: InputEmbeddings,
                 src_pos_embedding: PositionalEncoding, target_pos_embedding: PositionalEncoding, projection_layer: ProjectionLayer):
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.target_embedding = target_embedding
        self.src_pos_embedding = src_pos_embedding
        self.target_pos_embedding = target_pos_embedding
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        """
        This is the encoder block which uses the source embeddings 
        and source positional embeddings.
        """
        src = self.src_embedding(src)
        src = self.src_pos_embedding(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, target_mask, target):
        """
        This is for the decoder block which uses the target embeddings, 
        target positional embeddings and the key and value pairs from the 
        encoder. 
        """
        target = self.target_embedding(target)
        target = self.target_pos_embedding(target)
        return self.decoder(target, encoder_output, src_mask, target_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    

def build_transformer(src_vocab_size: int , target_vocab_size: int, src_seq_len: int, target_seq_len: int,
                      model_dimension: int = 512, num_of_blocks: int = 6, num_of_heads: int = 8, 
                      dropout: float = 0.1, feed_forward_units: int = 2048) -> Transformer:
    """
    src_vocab_size: Size of the source vocabulary.
    target_vocab_size: Size of the target vocabulary.
    src_seq_len: Length of the max sequence in the source.
    target_seq_len: Length of the max sequence in the target.
    model_dimension: Dimension of the input and output embeddings.
    num_of_units: Number of encoder and decoder blocks.
    num_of_heads: Number of attention heads.
    dropout: probability of dropout
    feed_forward_units: number of hidden units in the FCNN in the feed forward layer. 
    """
    
    # Creating the embedding layers
    src_embedding = InputEmbeddings(model_dimension=model_dimension, vocab_size=src_vocab_size)
    target_embedding = InputEmbeddings(model_dimension=model_dimension, vocab_size=target_vocab_size)

    # Creating the positional encoding layers
    src_positional_encoding = PositionalEncoding(model_dimension=model_dimension, sequence_length=src_seq_len, dropout=dropout)
    target_positional_encoding = PositionalEncoding(model_dimension=model_dimension, sequence_length=target_seq_len, dropout=dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(num_of_blocks):
        encoder_self_attention_block = MultiHeadAttentionBlock(model_dimension=model_dimension, num_of_heads=num_of_heads,
                                                               dropout=dropout)
        feed_forward_block = FeedForwardBlock(model_dimension=model_dimension, ff_dimension=feed_forward_units, dropout=dropout)
        encoder_block = EncoderBlock(self_attention_block=encoder_self_attention_block, feed_forward_block=feed_forward_block)
        encoder_blocks.append(encoder_block)

    # Creating the decoder blocks
    decoder_blocks = []
    for _ in range(num_of_blocks):
        decoder_self_attention_block = MultiHeadAttentionBlock(model_dimension=model_dimension, num_of_heads=num_of_heads, dropout=dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(model_dimension=model_dimension, num_of_heads=num_of_heads, dropout=dropout)
        feed_forward_block = FeedForwardBlock(model_dimension=model_dimension, ff_dimension=feed_forward_units, dropout=dropout)
        decoder_block = DecoderBlock(self_attention_block=decoder_self_attention_block, cross_attention_block=decoder_cross_attention_block, feed_foward_block=feed_forward_block, dropout=dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and the decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(model_dimension=model_dimension, vocab_size=target_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder=encoder, decoder=decoder, src_embedding=src_embedding, target_embedding=target_embedding,
                              src_pos_embedding=src_positional_encoding, target_pos_embedding=target_positional_encoding,
                              projection_layer=projection_layer)
    
    # Initialize the parameters
    for parameter in transformer.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)

    return transformer