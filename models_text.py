from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.01)

    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class NeuMFWithGRU(nn.Module):
    """
    NeuMF + GRU text encoder.

    GMF branch:
        user embedding * item embedding

    MLP branch:
        concat(user embedding, item embedding) -> dense layers

    Text branch:
        token ids -> word embeddings -> GRU -> last hidden state

    Fusion:
        concat(GMF vector, MLP vector, text vector) -> Dense -> Sigmoid
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        vocab_size: int,
        gmf_embedding_dim: int = 8,
        mlp_layers: Sequence[int] = (64, 32, 16, 8),
        word_embedding_dim: int = 64,
        gru_hidden_dim: int = 64,
        dropout: float = 0.0,
        padding_idx: int = 0,
    ):
        super().__init__()

        if len(mlp_layers) < 2:
            raise ValueError("mlp_layers must contain at least input size and one hidden size, e.g. [64, 32].")

        if mlp_layers[0] % 2 != 0:
            raise ValueError("mlp_layers[0] must be even because it is split between user and item embeddings.")

        mlp_embedding_dim = mlp_layers[0] // 2

        # GMF embeddings
        self.gmf_user_embedding = nn.Embedding(num_users, gmf_embedding_dim)
        self.gmf_item_embedding = nn.Embedding(num_items, gmf_embedding_dim)

        # MLP embeddings
        self.mlp_user_embedding = nn.Embedding(num_users, mlp_embedding_dim)
        self.mlp_item_embedding = nn.Embedding(num_items, mlp_embedding_dim)

        # MLP branch
        mlp_modules = []
        for in_dim, out_dim in zip(mlp_layers[:-1], mlp_layers[1:]):
            mlp_modules.append(nn.Linear(in_dim, out_dim))
            mlp_modules.append(nn.ReLU())
            if dropout > 0:
                mlp_modules.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*mlp_modules)

        # Text branch
        self.word_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=word_embedding_dim,
            padding_idx=padding_idx,
        )

        self.gru = nn.GRU(
            input_size=word_embedding_dim,
            hidden_size=gru_hidden_dim,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        fusion_dim = gmf_embedding_dim + mlp_layers[-1] + gru_hidden_dim
        self.output = nn.Linear(fusion_dim, 1)
        self.sigmoid = nn.Sigmoid()

        self.apply(init_weights)

    def forward(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        # GMF branch
        gmf_user = self.gmf_user_embedding(user_idx)
        gmf_item = self.gmf_item_embedding(item_idx)
        gmf_vector = gmf_user * gmf_item

        # MLP branch
        mlp_user = self.mlp_user_embedding(user_idx)
        mlp_item = self.mlp_item_embedding(item_idx)
        mlp_input = torch.cat([mlp_user, mlp_item], dim=-1)
        mlp_vector = self.mlp(mlp_input)

        # Text branch
        embedded_tokens = self.word_embedding(token_ids)
        _, hidden = self.gru(embedded_tokens)
        text_vector = hidden[-1]
        text_vector = self.dropout(text_vector)

        # Fusion
        fusion = torch.cat([gmf_vector, mlp_vector, text_vector], dim=-1)

        logits = self.output(fusion)
        probs = self.sigmoid(logits)

        return probs.squeeze(-1)
