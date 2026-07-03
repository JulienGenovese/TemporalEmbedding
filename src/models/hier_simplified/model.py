"""Simplified variant: hierarchical model with concat+MLP field fusion.

Ablation baseline for the FieldTransformer: everything else (feature
schema, TransactionEncoder, SequenceTransformer, MTM + InfoNCE heads)
is inherited unchanged from ``hier_transformer``.
"""

from src.models.hier_simplified.fusion import ConcatFusion
from src.models.hier_transformer.model import EmbeddingModel, TransactionTransformer


class SimplifiedTransactionTransformer(TransactionTransformer):
    """`TransactionTransformer` with ConcatFusion instead of FieldTransformer.

    ``field_n_layers`` / ``field_n_heads`` are still accepted (the shared
    config carries them) but have no effect here.
    """

    def __init__(
        self,
        *args,
        d_field: int = 64,
        d_model: int = 128,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            *args, d_field=d_field, d_model=d_model, dropout=dropout, **kwargs,
        )
        # replace the attention-based fusion built by the parent; keeping the
        # attribute name lets _encode_client and checkpoints work unchanged
        self.field_transformer = ConcatFusion(
            n_fields=self.encoder.n_fields,
            d_field=d_field,
            d_model=d_model,
            dropout=dropout,
        )


class SimplifiedEmbeddingModel(EmbeddingModel):
    """`EmbeddingModel` wrapper around the simplified backbone."""

    backbone_cls = SimplifiedTransactionTransformer
