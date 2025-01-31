import torch.nn as nn
import torch
from layers.SelfAttention_Family import FullAttention, AttentionLayer
import torch.nn.functional as F
from layers.Embed import PositionalEmbedding
import numpy as np

class FlattenHead(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super(EnEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars
    
class ConfigurableEncoderLayer(nn.Module):
    def __init__(self, use_attention, use_cross, self_attention, cross_attention, d_model, d_cross, d_ff=None,
                 dropout=0.1, activation="relu"):
        """
        Configurable Encoder Layer that uses attention-based or linear transformations
        and keeps convolutional layers for local feature extraction.
        Args:
            use_attention (bool): Whether to use attention mechanisms.
            self_attention: Self-attention module.
            cross_attention: Cross-attention module.
            d_model (int): Dimension of the input embeddings.
            d_ff (int): Dimension of the feedforward network. Defaults to 4 * d_model.
            dropout (float): Dropout rate for regularization.
            activation (str): Activation function ('relu' or 'gelu').
        """
        super(ConfigurableEncoderLayer, self).__init__()
        self.use_attention = use_attention
        self.use_cross = use_cross

        d_ff = d_ff or 4 * d_model

        if self.use_attention:
            # Attention-based components
            self.self_attention = self_attention
            if self.use_cross:
                self.cross_attention = cross_attention
        else:
            # Small linear transformations
            self.linear1_1 = nn.Linear(d_model, d_ff)
            self.linear1_2 = nn.Linear(d_model, d_ff)
            self.linear2 = nn.Linear(d_ff, d_model)
            self.linear_y = nn.Linear(d_cross, 2)

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Activation function
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        """
        Forward pass through the encoder layer.
        Args:
            x (Tensor): Input tensor of shape [Batch, Time, Features].
            cross (Tensor): Cross-attention input tensor.
            x_mask: Self-attention mask (if applicable).
            cross_mask: Cross-attention mask (if applicable).
        Returns:
            Tensor: Processed output tensor.
        """
        if self.use_attention:
            B, L, D = cross.shape

            # Self-attention block
            x = x + self.dropout(self.self_attention(
                x, x, x,
                attn_mask=x_mask,
                tau=tau, delta=None
            )[0])
            x = self.norm1(x)

            if self.use_cross:
                x_glb_ori = x[:, -1, :].unsqueeze(1)
                x_glb = torch.reshape(x_glb_ori, (B, -1, D))
                x_glb_attn = self.dropout(self.cross_attention(
                    x_glb, cross, cross,
                    attn_mask=cross_mask,
                    tau=tau, delta=delta
                )[0])
                x_glb_attn = torch.reshape(x_glb_attn,
                                        (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)
                x_glb = x_glb_ori + x_glb_attn
                x_glb = self.norm2(x_glb)

                y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)
            else:
                # Global token creation without cross-attention
                x_glb_ori = x[:, -1, :].unsqueeze(1)
                # Example: Using mean pooling to create the global token
                x_glb = torch.mean(x, dim=1, keepdim=True)  # Average pooling across the sequence
                x_glb = self.norm2(x_glb)  # Normalize the global token

                y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)
        else:
            # Linear transformation block
            y = self.dropout(self.activation(self.linear1_1(x)))

            # Integrate `cross` tensor in the linear path
            cross_transformed = self.dropout(self.activation(self.linear1_2(cross)))  # Transform cross
            
            y = torch.cat([y, cross_transformed], dim=1)

            y = self.dropout(self.linear2(y))
            
            y = y.transpose(1, 2)
            
            y = self.linear_y(y)
            y = y.transpose(1, 2)

            x = self.norm1(x + y)

        # Convolutional block (shared for both configurations)
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))  # [Batch, Features, Time]
        y = self.dropout(self.conv2(y)).transpose(-1, 1)  # Back to [Batch, Time, Features]

        return self.norm3(x + y)

class ConfigurableEncoder(nn.Module):
    def __init__(self, configs, d_cross):
        """
        Configurable Encoder with support for attention-based or linear-based layers.
        Args:
            use_attention (bool): Whether to use attention mechanisms.
            d_model (int): Dimension of the input embeddings.
            d_ff (int): Dimension of the feedforward network.
            n_layers (int): Number of encoder layers.
            dropout (float): Dropout rate for regularization.
            activation (str): Activation function ('relu' or 'gelu').
        """
        super(ConfigurableEncoder, self).__init__()
        self.layers = nn.ModuleList([
            ConfigurableEncoderLayer(
                use_attention=configs.use_attention,
                use_cross=configs.use_cross,
                self_attention=AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads) if configs.use_attention else None,
                cross_attention=AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads) if configs.use_attention else None,
                d_model=configs.d_model,
                d_cross=d_cross,
                d_ff=configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation
            )
            for _ in range(configs.e_layers)
        ])
        self.norm = nn.LayerNorm(configs.d_model)

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        """
        Forward pass through the encoder.
        Args:
            x (Tensor): Input tensor of shape [Batch, Time, Features].
            cross (Tensor): Cross-attention input tensor.
        Returns:
            Tensor: Processed output tensor.
        """
        for layer in self.layers:
            x = layer(x, cross, x_mask, cross_mask, tau, delta)
        return self.norm(x)

class LearnableCombination(nn.Module):
    def __init__(self, d_model_bool, d_model_non_bool, d_model_final):
        super(LearnableCombination, self).__init__()
        # Linear layers to project boolean and non-boolean embeddings
        self.bool_proj = nn.Linear(d_model_bool, d_model_final)
        self.non_bool_proj = nn.Linear(d_model_non_bool, d_model_final)

    def forward(self, ex_embed_non_bool, ex_embed_bool):
        """
        ex_embed_non_bool: [Batch, Time, d_model_non_bool]
        ex_embed_bool: [Batch, Time, d_model_bool]
        """
        # Project each embedding to the same final dimension
        ex_embed_non_bool_proj = self.non_bool_proj(ex_embed_non_bool)
        ex_embed_bool_proj = self.bool_proj(ex_embed_bool)

        # Combine embeddings via summation
        combined_embed = ex_embed_non_bool_proj + ex_embed_bool_proj
        return combined_embed
