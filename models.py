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


class GMF(nn.Module):
    """
    Generalized Matrix Factorization.
        user_id -> user embedding
        item_id -> item embedding
        element-wise product
        linear layer
        sigmoid
    """

    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 8):
        super().__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.output = nn.Linear(embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()

        self.apply(init_weights)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)

        # GMF core: element-wise multiplication
        gmf_vector = user_emb * item_emb

        logits = self.output(gmf_vector)
        probs = self.sigmoid(logits)

        return probs.squeeze(-1)


class MLP(nn.Module):

    def __init__(
        self,
        num_users: int,
        num_items: int,
        layers: Sequence[int] = (64, 32, 16, 8),
        dropout: float = 0.0,
    ):
        super().__init__()

        if len(layers) < 2:
            raise ValueError("layers must contain at least input size and one hidden size, e.g. [64, 32].")

        if layers[0] % 2 != 0:
            raise ValueError("layers[0] must be even because it is split between user and item embeddings.")

        embedding_dim = layers[0] // 2

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        mlp_layers = []
        for in_dim, out_dim in zip(layers[:-1], layers[1:]):
            mlp_layers.append(nn.Linear(in_dim, out_dim))
            mlp_layers.append(nn.ReLU())

            if dropout > 0:
                mlp_layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*mlp_layers)
        self.output = nn.Linear(layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

        self.apply(init_weights)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)

        # MLP core: concatenate user and item embeddings
        x = torch.cat([user_emb, item_emb], dim=-1)

        mlp_vector = self.mlp(x)

        logits = self.output(mlp_vector)
        probs = self.sigmoid(logits)

        return probs.squeeze(-1)


class NeuMF(nn.Module):
    """
        GMF branch:
            user_id -> GMF user embedding
            item_id -> GMF item embedding
            element-wise product

        MLP branch:
            user_id -> MLP user embedding
            item_id -> MLP item embedding
            concat embeddings
            Dense + ReLU layers

        Fusion:
            concat(GMF vector, MLP vector)
            linear layer
            sigmoid
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        gmf_embedding_dim: int = 8,
        mlp_layers: Sequence[int] = (64, 32, 16, 8),
        dropout: float = 0.0,
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
        
        layers = []
        for in_dim, out_dim in zip(mlp_layers[:-1], mlp_layers[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)

        fusion_dim = gmf_embedding_dim + mlp_layers[-1]
        self.output = nn.Linear(fusion_dim, 1)
        self.sigmoid = nn.Sigmoid()

        self.apply(init_weights)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        # GMF branch
        gmf_user = self.gmf_user_embedding(user_idx)
        gmf_item = self.gmf_item_embedding(item_idx)
        gmf_vector = gmf_user * gmf_item

        # MLP branch
        mlp_user = self.mlp_user_embedding(user_idx)
        mlp_item = self.mlp_item_embedding(item_idx)
        mlp_input = torch.cat([mlp_user, mlp_item], dim=-1)
        mlp_vector = self.mlp(mlp_input)

        # Fusion
        fusion = torch.cat([gmf_vector, mlp_vector], dim=-1)

        logits = self.output(fusion)
        probs = self.sigmoid(logits)

        return probs.squeeze(-1)


def get_model(
    model_name: str,
    num_users: int,
    num_items: int,
    embedding_dim: int = 8,
    layers: Sequence[int] = (64, 32, 16, 8),
    dropout: float = 0.0,
) -> nn.Module:
    """
    Helper function to create a model by name.
    """
    name = model_name.lower()

    if name == "gmf":
        return GMF(num_users=num_users, num_items=num_items, embedding_dim=embedding_dim)

    if name == "mlp":
        return MLP(num_users=num_users, num_items=num_items, layers=layers, dropout=dropout)

    if name == "neumf":
        return NeuMF(
            num_users=num_users,
            num_items=num_items,
            gmf_embedding_dim=embedding_dim,
            mlp_layers=layers,
            dropout=dropout,
        )

    raise ValueError(f"Unknown model_name={model_name}. Choose from: gmf, mlp, neumf.")
