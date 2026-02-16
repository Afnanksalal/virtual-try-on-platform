# IP-Adapter Module for InstantID

This module provides the IP-Adapter (Image Prompt Adapter) implementation required by InstantID for identity-preserving image generation.

## Source

Based on [tencent-ailab/IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) and adapted for InstantID integration.

## Components

### `resampler.py`
- **Resampler**: Perceiver-based resampler that projects image embeddings into the diffusion model's latent space
- **PerceiverAttention**: Cross-attention mechanism for image feature extraction
- **FeedForward**: Feed-forward network for feature transformation

### `attention_processor.py`
- **AttnProcessor**: Default attention processor
- **IPAttnProcessor**: IP-Adapter attention processor that injects image features
- **AttnProcessor2_0**: PyTorch 2.0 optimized attention processor
- **IPAttnProcessor2_0**: PyTorch 2.0 optimized IP-Adapter processor

## Usage

The module is automatically imported by `instantid_pipeline.py`:

```python
from ip_adapter.resampler import Resampler
from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor
```

## How It Works

1. **Face Embedding Extraction**: InsightFace extracts a 512-dim face embedding
2. **Resampler**: Projects the embedding into the model's cross-attention space
3. **IP-Adapter Attention**: Injects the face features into SDXL's attention layers
4. **ControlNet**: Guides generation using facial keypoints
5. **Result**: Full-body image with user's face natively generated

## Dependencies

- PyTorch 2.0+
- einops
- torch.nn.functional (for scaled_dot_product_attention)

All dependencies are already in `requirements.txt`.

## License

Based on Apache 2.0 licensed code from tencent-ailab/IP-Adapter.

## References

- [IP-Adapter Paper](https://arxiv.org/abs/2308.06721)
- [InstantID Paper](https://arxiv.org/abs/2401.07519)
- [IP-Adapter GitHub](https://github.com/tencent-ailab/IP-Adapter)
- [InstantID GitHub](https://github.com/instantX-research/InstantID)
