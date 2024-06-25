import glob
import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download
from transformers import AutoConfig

from .language import LanguageModel, TextConfig
from .vision import VisionConfig, VisionModel


@dataclass
class PerceiverConfig:
    model_type: str = "idefics2"
    num_key_value_heads: int = 4
    resampler_depth: int = 3
    resampler_head_dim: int = 96
    resampler_n_heads: int = 16
    resampler_n_latents: int = 64

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclass
class ModelConfig:
    vision_config: VisionConfig
    auto_map: dict
    model_type: str
    query_num: int
    text_config: TextConfig = TextConfig()
    perceiver_config: PerceiverConfig = PerceiverConfig()
    ignore_index: int = -100
    image_token_index: int = 32001
    vocab_size: int = 151936
    layer_norm_eps: float = 1e-6

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


def get_2d_sincos_pos_embed(embed_dim, image_size):
    """
    image_size: image_size or (image_height, image_width)
    return:
    pos_embed: [image_height, image_width, embed_dim]
    """
    if isinstance(image_size, int):
        grid_h_size, grid_w_size = image_size, image_size
    else:
        grid_h_size, grid_w_size = image_size[0], image_size[1]

    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid_new(
        embed_dim // 2, grid[0]
    )  # (H, W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid_new(
        embed_dim // 2, grid[1]
    )  # (H, W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=-1)  # (H, W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_new(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (H, W)
    out: (H, W, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    out = np.einsum("hw,d->hwd", pos, omega)  # (H, W, D/2), outer product

    emb_sin = np.sin(out)  # (H, W, D/2)
    emb_cos = np.cos(out)  # (H, W, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=-1)  # (H, W, D)
    return emb


class MHA(nn.Module):
    def __init__(
        self,
        dims: int,
        num_heads: int,
        bias: bool = False,
    ):
        super().__init__()

        if (dims % num_heads) != 0:
            raise ValueError(
                "The input feature dimensions should be divisible by the "
                f"number of heads ({dims} % {num_heads}) != 0"
            )

        self.num_heads = num_heads
        head_dim = dims // num_heads
        self.scale = head_dim**-0.5

        self.in_proj = nn.Linear(dims, dims * 3, bias=bias)
        self.out_proj = nn.Linear(dims, dims, bias=bias)

    def __call__(self, queries: mx.array, kv: mx.array, mask=None, cache=None):
        B, L, D = queries.shape

        qkv = self.in_proj(queries)
        _, keys, values = mx.split(qkv, 3, axis=-1)

        num_heads = self.num_heads
        B, L, D = queries.shape
        _, S, _ = keys.shape
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(output)


class Resampler(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_queries = num_queries = config.query_num
        self.embed_dim = embed_dim = config.text_config.hidden_size
        self.num_heads = num_heads = embed_dim // 128
        self.adaptive = True
        self.kv_dim = config.vision_config.hidden_size

        self.ln_q = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.ln_kv = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        if self.kv_dim is not None and self.kv_dim != embed_dim:
            self.kv_proj = nn.Linear(self.kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()
        self.query = mx.zeros((num_queries, embed_dim))
        self.attn = MHA(embed_dim, num_heads, bias=True)
        self.ln_post = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.proj = (self.embed_dim**-0.5) * mx.random.normal(
            (self.embed_dim, self.embed_dim)
        )

    def __call__(self, x: mx.array, mask=None) -> mx.array:
        x = self.kv_proj(x)  # B * L * D
        x = self.ln_kv(x).transpose(1, 0, 2)  # L * B * D

        q = self.ln_q(self.query)  # Q * D

        out = self.attn(q[None, :], x)
        x = out.transpose(1, 0, 2)  # B * Q * D

        x = self.ln_post(x)
        x = x @ self.proj
        return x


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        self.model_type = config.model_type
        self.config = config

        self.vpm = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        self.resampler = Resampler(config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        pixel_attention_mask: Optional[mx.array] = None,
    ):
        if pixel_values is None:
            return self.language_model(input_ids)

        inputs_embeds = self.language_model.embed_tokens(input_ids)

        pooler_output, embeddings, hidden_state = self.vpm(
            pixel_values[0].transpose(0, 2, 3, 1), output_hidden_states=True
        )

        image_features = pooler_output[None, :].astype(pixel_values.dtype)

        image_features = self.resampler(image_features, mask=None)

        final_inputs_embeds = self._prepare_inputs_for_multimodal(
            image_features, inputs_embeds, input_ids
        )
        return final_inputs_embeds

    def _prepare_inputs_for_multimodal(self, image_features, inputs_embeds, input_ids):
        image_token_index = self.config.image_token_index
        num_images, num_image_patches, embed_dim = image_features.shape

        # Positions of <image> tokens in input_ids, assuming batch size is 1
        image_positions = np.where(input_ids[0] == image_token_index)[0].tolist()

        text_segments = []
        start_idx = 0

        for position in image_positions:
            text_segments.append(inputs_embeds[:, start_idx:position])
            start_idx = position + 1

        image_embeddings = mx.split(image_features, image_features.shape[0])
        final_embeddings = [v for p in zip(text_segments, image_embeddings) for v in p]
        final_embeddings += [inputs_embeds[:, start_idx:]]

        # Create a final embedding of shape
        # (1, num_image_patches*num_images + sequence_len, embed_dim)
        return mx.concatenate(final_embeddings, axis=1)

    def __call__(
        self, input_ids: mx.array, pixel_values: mx.array, mask: mx.array, cache=None
    ):
        input_embeddings = self.get_input_embeddings(input_ids, pixel_values)
        logits, cache = self.language_model(
            inputs=input_ids, cache=cache, inputs_embeds=input_embeddings
        )
        return logits, cache

    @staticmethod
    def from_pretrained(path_or_hf_repo: str):
        path = Path(path_or_hf_repo)
        if not path.exists():
            path = Path(
                snapshot_download(
                    repo_id=path_or_hf_repo,
                    allow_patterns=[
                        "*.json",
                        "*.safetensors",
                        "*.py",
                        "tokenizer.model",
                        "*.tiktoken",
                    ],
                )
            )

        with open(path / "config.json", "r") as f:
            config = json.load(f)

        text_config = AutoConfig.from_pretrained(config["text_config"]["model_type"])
        text_config = text_config.to_dict()
        config["text_config"] = text_config
        model_config = ModelConfig.from_dict(config)
        model_config.vision_config = VisionConfig.from_dict(config["vision_config"])
        model_config.text_config = TextConfig.from_dict(config["text_config"])
        model_config.perceiver_config = PerceiverConfig.from_dict(
            config["perceiver_config"]
        )

        model = Model(model_config)
        weight_files = glob.glob(str(path / "*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"No safetensors found in {path}")

        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))

        weights = model.sanitize(weights=weights)
        weights = VisionModel(model_config.vision_config).sanitize(weights=weights)
        weights = LanguageModel(model_config.text_config).sanitize(weights=weights)
        model.load_weights(list(weights.items()))
        return model

    def sanitize(self, weights):
        def replace_key(key):
            # Handle llm.model and llm prefixes
            key = re.sub(
                r"^(llm\.model|llm)",
                lambda m: (
                    "language_model" if m.group(0) == "llm.model" else "language_model"
                ),
                key,
            )

            # Handle specific resampler.attn.in_proj renaming
            key = re.sub(r"^(resampler\.attn\.in_proj)_(weight|bias)$", r"\1.\2", key)

            return key

        return {replace_key(k): v for k, v in weights.items()}
