"""Chronos-guided code selection for an existing residual VQ module."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from vector_quantize_pytorch import ResidualVQ
from vector_quantize_pytorch import vector_quantize_pytorch as vq_implementation


def chronos_guided_residual_vq(
    vq: ResidualVQ,
    tokens: Tensor,
    chronos_prior: Tensor,
    guidance_lambda: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Quantize ``tokens`` while adding Chronos/code cosine similarity to selection.

    The underlying residual-VQ implementation still owns code lookup, commitment
    loss, EMA updates, and residual stages. Only the logits passed to its existing
    code sampler are changed. A zero weight bypasses this adapter completely, which
    makes it exactly equivalent to calling ``vq(tokens)``.
    """
    if tokens.ndim != 3:
        raise ValueError("tokens must have shape [B, T, D]")
    if chronos_prior.shape != (tokens.shape[0], tokens.shape[-1]):
        raise ValueError(
            "chronos_prior must have shape "
            f"[B, D]={tokens.shape[0], tokens.shape[-1]}, got {tuple(chronos_prior.shape)}"
        )

    guidance_lambda = float(guidance_lambda)
    if guidance_lambda == 0.0:
        return vq(tokens)

    codebook = vq.layers[0]._codebook
    _, sequence_length, _ = tokens.shape
    surrogate_terms: list[Tensor] = []

    def guided_sampler(distance_logits: Tensor, *args, **kwargs):
        # Codebook.forward flattens tokens in batch-major order to [1, B*T, D].
        prior = F.normalize(chronos_prior.float(), dim=-1)
        prior = prior[:, None, :].expand(-1, sequence_length, -1).reshape(1, -1, prior.shape[-1])
        codes = F.normalize(codebook.embed.detach().float(), dim=-1)
        similarity = torch.einsum("hnd,hkd->hnk", prior, codes)
        if similarity.shape != distance_logits.shape:
            raise RuntimeError(
                "Chronos guidance score shape does not match VQ distances: "
                f"{tuple(similarity.shape)} vs {tuple(distance_logits.shape)}"
            )
        # The installed VQ exposes negative Euclidean distance. Squaring its
        # magnitude gives the requested -||x-e||^2 term without changing codes.
        scores = -distance_logits.square() + guidance_lambda * similarity.to(distance_logits.dtype)
        raw_codes = codebook.embed.detach().to(scores.dtype).clone()
        soft_codes = torch.einsum("hnk,hkd->hnd", scores.softmax(dim=-1), raw_codes)
        surrogate_terms.append(soft_codes.reshape(tokens.shape))
        return original_sampler(scores, *args, **kwargs)

    # Newer releases keep the sampler on Codebook; the repository's pinned
    # release resolves it from the package module. Patch only for this call and
    # restore in finally, so ctx/guide VQ and all baseline modes remain untouched.
    sampler_owner = codebook if hasattr(codebook, "gumbel_sample") else vq_implementation
    original_sampler = sampler_owner.gumbel_sample
    sampler_owner.gumbel_sample = guided_sampler
    try:
        quantized, indices, loss = vq(tokens)
        # Hard argmax determines the forward value. This zero-valued
        # straight-through term supplies a useful gradient to the Chronos
        # projection despite the discrete selection operation.
        surrogate = torch.stack(surrogate_terms).sum(dim=0)
        quantized = quantized + surrogate - surrogate.detach()
        return quantized, indices, loss
    finally:
        sampler_owner.gumbel_sample = original_sampler
