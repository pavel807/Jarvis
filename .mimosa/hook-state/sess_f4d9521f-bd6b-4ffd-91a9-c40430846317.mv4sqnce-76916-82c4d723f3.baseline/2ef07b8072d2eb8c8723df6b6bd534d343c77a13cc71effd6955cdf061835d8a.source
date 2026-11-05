class LayerNorm(Module):
  __parameters__ = ["weight", "bias", ]
  __buffers__ = []
  weight : Tensor
  bias : Tensor
  training : bool
  _is_full_backward_hook : NoneType
  elementwise_affine : Final[bool] = True
  normalized_shape : Final[Tuple[int]] = (128,)
  eps : Final[float] = 1.0000000000000001e-05
  def forward(self: __torch__.torch.nn.modules.normalization.LayerNorm,
    input: Tensor) -> Tensor:
    _0 = __torch__.torch.nn.functional.layer_norm
    weight = self.weight
    bias = self.bias
    _1 = _0(input, [128], weight, bias, 1.0000000000000001e-05, )
    return _1
