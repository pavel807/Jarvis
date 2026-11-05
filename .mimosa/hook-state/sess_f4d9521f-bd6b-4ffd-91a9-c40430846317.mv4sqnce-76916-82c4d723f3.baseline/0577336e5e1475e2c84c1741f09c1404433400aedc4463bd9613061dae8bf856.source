class LayerNorm(Module):
  __parameters__ = ["weight", "bias", ]
  __buffers__ = []
  weight : Tensor
  bias : Tensor
  training : bool
  _is_full_backward_hook : NoneType
  elementwise_affine : Final[bool] = True
  normalized_shape : Final[Tuple[int]] = (512,)
  eps : Final[float] = 9.9999999999999995e-07
  def forward(self: __torch__.torch.nn.modules.normalization.___torch_mangle_7.LayerNorm,
    input: Tensor) -> Tensor:
    _0 = __torch__.torch.nn.functional.layer_norm
    weight = self.weight
    bias = self.bias
    _1 = _0(input, [512], weight, bias, 9.9999999999999995e-07, )
    return _1
