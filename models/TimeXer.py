import torch
import torch.nn as nn
from layers.Embed import DataEmbedding_inverted, BooleanEmbedding
from layers.Encoder import FlattenHead, EnEmbedding, LearnableCombination, ConfigurableEncoder


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
        
        self.ex_embedding_non_bool = DataEmbedding_inverted(configs.seq_len, 
                                                            configs.d_model, 
                                                            configs.embed, 
                                                            configs.freq,
                                                            configs.dropout)
        
        self.ex_embedding_forecast_non_bool = DataEmbedding_inverted(configs.pred_len, 
                                                                     configs.d_model, 
                                                                     configs.embed, 
                                                                     configs.freq, 
                                                                     configs.dropout)
        if self.use_boolean:
            self.ex_embedding_bool = BooleanEmbedding(configs.seq_len, configs.d_model, configs.dropout)
            self.ex_embedding_forecast_bool = BooleanEmbedding(configs.pred_len, configs.d_model, configs.dropout)

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

            
    def forecast(self, x_enc, x_mark_enc, x_forecast, x_forecast_mark_enc): 
               
        if self.use_boolean:
            x_enc_bool = x_enc[:, :, self.boolean_indices]  # Boolean features
            x_forecast_bool = x_forecast[:, :, self.boolean_indices]

        self.non_bool_indices = [i for i in range(x_enc.shape[-1]) if i not in self.boolean_indices] #TODO do this more efficiently

        # Separate boolean and non-boolean features
        x_enc_non_bool = x_enc[:, :, self.non_bool_indices]  # Non-boolean features [batch_size, seq_len, num_non_bool_features]
        x_forecast_non_bool = x_forecast[:, :, self.non_bool_indices[:-1]]
        
        # Normalize non-boolean data if self.use_norm is True
        if self.use_norm:
            means = x_enc_non_bool.mean(1, keepdim=True).detach()
            stdev = torch.sqrt(torch.var(x_enc_non_bool, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc_non_bool = (x_enc_non_bool - means) / stdev
            
            means_forecast = x_forecast_non_bool.mean(1, keepdim=True).detach()
            stdev_forecast = torch.sqrt(torch.var(x_forecast_non_bool, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_forecast_non_bool = (x_forecast_non_bool - means_forecast) / stdev_forecast
            
        else:
            means = stdev = None
            
        # Embeddings
        en_embed, n_vars = self.en_embedding(x_enc_non_bool[:, :, -1].unsqueeze(-1).permute(0, 2, 1))  # Target embedding treated as [batch_size, num_non_bool_features, seq_len], return [batch_size, 1+1, 2*seq_len]

        ex_embed_non_bool = self.ex_embedding_non_bool(x_enc_non_bool, x_mark_enc)  # Non-boolean embedding
        ex_embed_forecast_non_bool = self.ex_embedding_forecast_non_bool(x_forecast, x_forecast_mark_enc)  # Non-boolean embedding
        
        if self.use_boolean:
            ex_embed_bool = self.ex_embedding_bool(x_enc_bool)  # Boolean embedding
            ex_embed_forecast_bool = self.ex_embedding_forecast_bool(x_forecast_bool)

            if self.use_learnable_combination:
                # Learnable combination of boolean and non-boolean embeddings
                ex_embed = self.learnable_combination(ex_embed_non_bool, ex_embed_bool)
            else: 
                # concat non-boolean and boolean embeddings
                ex_embed = torch.cat([ex_embed_non_bool, ex_embed_bool, ex_embed_forecast_non_bool, ex_embed_forecast_bool], dim=1)
        else:
            # ex_embed = ex_embed_non_bool
            ex_embed = torch.cat([ex_embed_non_bool, ex_embed_forecast_non_bool], dim=1)
            
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


    def forward(self, x_enc, x_mark_enc, x_forecast, x_forecast_mark_enc, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_forecast, x_forecast_mark_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]

        
