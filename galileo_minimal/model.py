"""
model.py - Symbolic Transformer (encoder-decoder)
"""
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class SymbolicTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        feature_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        
        self.feature_proj = nn.Linear(feature_dim, d_model)
        self.feature_norm = nn.LayerNorm(d_model)
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        
        self.output_proj = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, features: torch.Tensor, tgt: torch.Tensor, tgt_mask=None):
        memory = self.feature_norm(self.feature_proj(features)).unsqueeze(1)
        tgt_embed = self.pos_encoder(self.token_embed(tgt) * math.sqrt(self.d_model))
        
        if tgt_mask is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        output = self.transformer(memory, tgt_embed, tgt_mask=tgt_mask)
        return self.output_proj(output)

    def generate(self, features: torch.Tensor, vocab, max_len: int = 64) -> list[int]:
        """Autoregressive greedy generation."""
        self.eval()
        device = features.device
        tokens = [vocab.bos_token]
        
        with torch.no_grad():
            for _ in range(max_len):
                tgt = torch.tensor([tokens], dtype=torch.long, device=device)
                logits = self(features.unsqueeze(0), tgt)
                next_token = torch.argmax(logits[0, -1]).item()
                if next_token == vocab.eos_token:
                    break
                tokens.append(next_token)
        
        return tokens[1:]
