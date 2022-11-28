from torch import Tensor
from typing import Optional, Tuple

import numpy as np
import math

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

from .utils import compute_mask_indices


class FeatureEncoder(nn.Module):
    def __init__(self, output_dim=768):
        super(FeatureEncoder, self).__init__()
        self.encoder = nn.ModuleList(
            [
                nn.Conv1d(in_channels=1, out_channels=512, kernel_size=10, stride=5),
                nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=2),
                nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=2),
                nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=2),
                nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=2),
                nn.Conv1d(in_channels=512, out_channels=512, kernel_size=2, stride=2),
                nn.Conv1d(
                    in_channels=512, out_channels=output_dim, kernel_size=2, stride=2
                ),
            ]
        )
        self.gelu = nn.GELU()

    def forward(self, x):
        for conv_layer in self.encoder:
            x = conv_layer(x)
            x = self.gelu(x)
        return torch.permute(x, (0, 2, 1))


class ScaledDotProductAttention(nn.Module):
    def __init__(
        self,
        key_dim: int,
    ) -> None:
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_key_dim = np.sqrt(key_dim)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        key = key.transpose(1, 2)
        attn_distribution = torch.bmm(query, key) / self.sqrt_key_dim

        if mask is not None:
            attn_distribution = attn_distribution.masked_fill(mask, -np.inf)

        attn_distribution = F.softmax(attn_distribution, dim=-1)
        context = torch.bmm(attn_distribution, value)

        return context, attn_distribution


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
    ) -> None:
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.scaled_dot = ScaledDotProductAttention(self.head_dim)
        self.query_fc = nn.Linear(model_dim, num_heads * self.head_dim)
        self.key_fc = nn.Linear(model_dim, num_heads * self.head_dim)
        self.value_fc = nn.Linear(model_dim, num_heads * self.head_dim)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        batch = query.size(0)

        query = self.query_fc(query).view(batch, -1, self.num_heads, self.head_dim)
        key = self.key_fc(key).view(batch, -1, self.num_heads, self.head_dim)
        value = self.value_fc(value).view(batch, -1, self.num_heads, self.head_dim)

        query = (
            query.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch * self.num_heads, -1, self.head_dim)
        )
        key = (
            key.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch * self.num_heads, -1, self.head_dim)
        )
        value = (
            value.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch * self.num_heads, -1, self.head_dim)
        )

        if mask is not None:
            mask = mask.repeat(self.num_heads, 1, 1)

        context, attn_distribution = self.scaled_dot(query, key, value, mask)

        context = context.view(batch, self.num_heads, -1, self.head_dim)
        context = (
            context.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch, -1, self.num_heads * self.head_dim)
        )  # (B, T, D)

        return context, attn_distribution


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 512, max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, length: int) -> Tensor:
        return self.pe[:, :length, :]


class PositionWiseFeedForward(nn.Module):
    """
    Implement position-wise feed forward layer.
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """

    def __init__(
        self,
        model_dim: int = 512,
        ff_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super(PositionWiseFeedForward, self).__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(ff_dim, model_dim),
            nn.Dropout(p=dropout),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.feed_forward(inputs)


class LayerNorm(nn.Module):
    def __init__(
        self,
        model_dim: int,
        eps: float = 1e-6,
    ) -> None:
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(model_dim))
        self.b_2 = nn.Parameter(torch.zeros(model_dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class EncoderLayer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
    ) -> None:
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(model_dim, num_heads)
        self.layer_norm1 = LayerNorm(model_dim)
        self.feed_forward = PositionWiseFeedForward(model_dim, ff_dim, dropout)
        self.layer_norm2 = LayerNorm(model_dim)

    def forward(
        self,
        inputs: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        context, _ = self.self_attn(inputs, inputs, inputs, mask)
        out = self.layer_norm1(inputs + context)
        out = self.layer_norm2(out + self.feed_forward(out))
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
    ) -> None:
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.pos_encoding = PositionalEncoding(model_dim)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(model_dim, num_heads, ff_dim, dropout)
                for _ in range(self.num_layers)
            ]
        )

    def forward(
        self,
        inputs: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        length = inputs.size(1)
        out = inputs + self.pos_encoding(length)
        for i in range(self.num_layers):
            out = self.layers[i](out, mask)
        return out


class DecoderLayer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
    ) -> None:
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(model_dim, num_heads)
        self.layer_norm1 = LayerNorm(model_dim)
        self.src_attn = MultiHeadAttention(model_dim, num_heads)
        self.layer_norm2 = LayerNorm(model_dim)
        self.feed_forward = PositionWiseFeedForward(model_dim, ff_dim, dropout)
        self.layer_norm3 = LayerNorm(model_dim)

    def forward(
        self,
        encodings: Tensor,
        mask: Optional[Tensor] = None,
    ):
        context, _ = self.self_attn(encodings, encodings, encodings, mask)
        out = self.layer_norm1(encodings + context)
        context, _ = self.src_attn(out, out, out, mask)
        out = self.layer_norm2(out + context)
        out = self.layer_norm3(out + self.feed_forward(out))
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
    ) -> None:
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.pos_encoding = PositionalEncoding(model_dim)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(model_dim, num_heads, ff_dim, dropout)
                for _ in range(self.num_layers)
            ]
        )

    def forward(
        self,
        encodings: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        length = encodings.size(1)
        out = encodings + self.pos_encoding(length)
        for i in range(self.num_layers):
            out = self.layers[i](out, mask)
        return out


@dataclass
class ModelOutput:
    encoder_output: torch.Tensor
    decoder_output: torch.Tensor


class WavAutoEncoder(nn.Module):
    def __init__(
        self,
        model_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super(WavAutoEncoder, self).__init__()
        self.encoder = Encoder(model_dim, num_layers, num_heads, ff_dim, dropout)
        self.decoder = Decoder(model_dim, num_layers, num_heads, ff_dim, dropout)

    def forward(
        self,
        inputs: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        encoding_output = self.encoder(inputs, mask)
        decoding_output = self.decoder(encoding_output, mask)
        return encoding_output, decoding_output


class WavAutoEncoderConfig(PretrainedConfig):
    def __init__(
        self,
        model_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.pruned_heads = {}
        self.initializer_range = 0.02

    def __repr__(self):
        return str(self.to_json_string())


class WavAutoEncoderPreTrainedModel(PreTrainedModel):
    def _init_weights(self, module):
        if isinstance(module, Encoder):
            for layer in module.layers:
                if isinstance(layer, MultiHeadAttention):
                    layer.query_fc.weight.data.normal_(
                        mean=0.0, std=self.config.initializer_range
                    )
                    layer.key_fc.weight.data.normal_(
                        mean=0.0, std=self.config.initializer_range
                    )
                    layer.value_fc.weight.data.normal_(
                        mean=0.0, std=self.config.initializer_range
                    )
                    layer.query_fc.bias.data.zero_()

                if isinstance(layer, PositionWiseFeedForward):
                    layer.feed_forward.weight.data.normal_(
                        mean=0.0, std=self.config.initializer_range
                    )
                    layer.feed_forward.bias.data.zero_()
        elif isinstance(module, Decoder):
            for layer in module.layers:
                if isinstance(layer, MultiHeadAttention):
                    layer.query_fc.weight.data.normal_(
                        mean=0.0, std=self.config.initializer_range
                    )
                    layer.key_fc.weight.data.normal_(
                        mean=0.0, std=self.config.initializer_range
                    )
                    layer.value_fc.weight.data.normal_(
                        mean=0.0, std=self.config.initializer_range
                    )
                    layer.query_fc.bias.data.zero_()

                if isinstance(layer, PositionWiseFeedForward):
                    layer.feed_forward.weight.data.normal_(
                        mean=0.0, std=self.config.initializer_range
                    )
                    layer.feed_forward.bias.data.zero_()


class WavAutoEncoderModel(WavAutoEncoderPreTrainedModel):
    def __init__(self, config: WavAutoEncoderConfig):
        super().__init__(config)
        self.config = config
        self.wav_auto_encoder = WavAutoEncoder(
            model_dim=config.model_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            dropout=config.dropout,
        )
        self.features_Encoder = FeatureEncoder()
        # self.compute_masked_indices = compute_mask_indices
        self.init_weights()

    def forward(
        self,
        inputs: Tensor,
        mask: Optional[Tensor] = None,
    ) -> ModelOutput:

        assert (inputs.dim() == 3), "Input shape should be [batch_size, num_frames, num_features]"

        input_features = self.features_Encoder(inputs)

        mask_indices = compute_mask_indices(input_features, 0.2, 10, 10)

        input_masked = input_features.masked_fill(mask_indices.unsqueeze(-1), 0)

        encoder_output, decoder_output = self.wav_auto_encoder(input_masked, mask)
        return ModelOutput(
            encoder_output=encoder_output,
            decoder_output=decoder_output,
        )


if __name__ == "__main__":

    config = WavAutoEncoderConfig()
    model = WavAutoEncoderModel(config)
    wav_signal = torch.randn(2, 1, 16000)

    outputs = model(wav_signal)
    print(outputs)
