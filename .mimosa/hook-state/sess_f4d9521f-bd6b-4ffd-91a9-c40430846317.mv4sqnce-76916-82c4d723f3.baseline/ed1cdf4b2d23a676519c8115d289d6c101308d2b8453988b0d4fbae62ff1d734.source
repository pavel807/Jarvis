class Embedding(Module):
  __parameters__ = ["weight", ]
  __buffers__ = []
  weight : Tensor
  training : bool
  _is_full_backward_hook : NoneType
  scale_grad_by_freq : Final[bool] = False
  sparse : Final[bool] = False
  embedding_dim : Final[int] = 64
  num_embeddings : Final[int] = 85
  max_norm : Final[NoneType] = None
  norm_type : Final[float] = 2.
  padding_idx : Final[NoneType] = None
  def forward(self: __torch__.torch.nn.modules.sparse.___torch_mangle_13.Embedding,
    input: Tensor) -> Tensor:
    _0 = __torch__.torch.nn.functional.embedding
    weight = self.weight
    _1 = _0(input, weight, None, None, 2., False, False, )
    return _1
