import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding, BooleanEmbedding
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

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.features = configs.features
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.patch_len = configs.patch_len
        self.patch_num = int(configs.seq_len // configs.patch_len)
        self.n_vars = 1 if configs.features == 'MS' else configs.enc_in
        self.boolean_indices = []
        self.n_stamp_features = 4

        self.use_boolean = configs.use_boolean
        if self.use_boolean:
            self.boolean_indices = []
            self.non_boolean_indices = []
            
        self.use_learnable_combination = configs.use_learnable_combination
        self.use_flatten_head = configs.use_flatten_head
            
        # Embedding
        self.en_embedding = EnEmbedding(self.n_vars, configs.d_model, self.patch_len, configs.dropout)
        self.ex_embedding_non_bool = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                   configs.dropout)
        if self.use_boolean:
            self.ex_embedding_bool = BooleanEmbedding(configs.seq_len, configs.d_model, configs.dropout)

        if self.use_learnable_combination:
            # Learnable combination of embeddings
            self.learnable_combination = LearnableCombination(
                d_model_bool=configs.d_model,
                d_model_non_bool=configs.d_model,
                d_model_final=configs.d_model
            )
        
        # Encoder-only architecture
        self.encoder = ConfigurableEncoder(configs=configs, d_cross=configs.n_vars_num + configs.n_vars_time_features + 2)
        self.head_nf = configs.d_model * (self.patch_num + 1)
        if self.use_flatten_head:
            self.head = FlattenHead(self.head_nf, configs.pred_len,
                                head_dropout=configs.dropout)
        
        self.initialize_weights()
        
    def update_data_infos(self, infos_dict):
        self.boolean_indices = infos_dict['boolean_indices']
        self.n_stamp_features = infos_dict['n_stamp_features']
        
    def initialize_weights(self):
        """
        Initialize the weights of the model using recommended methods.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # Xavier initialization for linear layers
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # Initialize biases to 0
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # Kaiming for Conv1d layers
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # Initialize biases to 0
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)  # Set LayerNorm bias to 0
                nn.init.constant_(m.weight, 1.0)  # Set LayerNorm weight to 1.0

            
    def forecast(self, x_enc, x_mark_enc): 
        if self.use_boolean:
            x_enc_bool = x_enc[:, :, self.boolean_indices]  # Boolean features

        self.non_bool_indices = [i for i in range(x_enc.shape[-1]) if i not in self.boolean_indices] #TODO do this more efficiently

        # Separate boolean and non-boolean features
        x_enc_non_bool = x_enc[:, :, self.non_bool_indices]  # Non-boolean features [batch_size, seq_len, num_non_bool_features]
    
        # Normalize non-boolean data if self.use_norm is True
        if self.use_norm:
            means = x_enc_non_bool.mean(1, keepdim=True).detach()
            stdev = torch.sqrt(torch.var(x_enc_non_bool, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc_non_bool = (x_enc_non_bool - means) / stdev
        else:
            means = stdev = None
            
        # Embeddings
        en_embed, n_vars = self.en_embedding(x_enc_non_bool[:, :, -1].unsqueeze(-1).permute(0, 2, 1))  # Target embedding treated as [batch_size, num_non_bool_features, seq_len], return [batch_size, 1+1, 2*seq_len]
        ex_embed_non_bool = self.ex_embedding_non_bool(x_enc_non_bool, x_mark_enc)  # Non-boolean embedding

        if self.use_boolean:
            ex_embed_bool = self.ex_embedding_bool(x_enc_bool)  # Boolean embedding

            if self.use_learnable_combination:
                # Learnable combination of boolean and non-boolean embeddings
                ex_embed = self.learnable_combination(ex_embed_non_bool, ex_embed_bool)
            else: 
                # concat non-boolean and boolean embeddings
                ex_embed = torch.cat([ex_embed_non_bool, ex_embed_bool], dim=1)
        else:
            ex_embed = ex_embed_non_bool
            
        # Pass through encoder
        enc_out = self.encoder(en_embed, ex_embed)
        
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out


    def forward(self, x_enc, x_mark_enc, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]

        
