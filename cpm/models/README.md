# Models

The box-generator architectures compared in the report, plus their shared
building blocks. Every model exposes the same interface
`forward(volume, latent_z) -> (boxes, z_pred)`, so they are interchangeable as
GLUE's internals generator (see `../../INTEGRATION.md`).

**For the architecture comparison and ablation study, see the project report
(`../../latex/main.pdf`).**

## Files

- **`baseline.py`**: `AutoRegressiveBoxGenerator` (`mlp`) — the non-attentive autoregressive MLP; this is GLUE's original internals model and the baseline to beat
- **`atiss.py`**: `CTATISSBoxGenerator` (`ctatiss`) — ATISS-style autoregressive transformer built from DiT blocks
- **`parallel_transformer.py`**: `ParallelBoxGenerator` (`parallel`) — DETR-style parallel decoder that emits all boxes at once; supports the ablation variants
- **`conditional_parallel.py`**: `ConditionalParallelBoxGenerator` (`cond_parallel`) — the parallel decoder extended with a constraint encoder for signed pairwise adjacency targets
- **`dit_block.py`**: DiT adaLN-Zero transformer blocks (`BoxDiTBlock`, `BoxDiTCrossAttnBlock`, `FinalAdaLN`) and zero-init helper
- **`components.py`**: Shared building blocks reused across models (`FourierEncoding`, `AuxiliaryPredictor`)
