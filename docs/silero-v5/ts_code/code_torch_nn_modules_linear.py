class NonDynamicallyQuantizableLinear(Module):
  __parameters__ = ["weight", "bias", ]
  __buffers__ = []
  weight : Tensor
  bias : Tensor
  training : bool
  _is_full_backward_hook : NoneType
  in_features : Final[int] = 128
  out_features : Final[int] = 128
  def forward(self: __torch__.torch.nn.modules.linear.NonDynamicallyQuantizableLinear,
    input: Tensor) -> Tensor:
    weight = self.weight
    bias = self.bias
    return torch.linear(input, weight, bias)
class Linear(Module):
  __parameters__ = ["weight", "bias", ]
  __buffers__ = []
  weight : Tensor
  bias : Tensor
  training : bool
  _is_full_backward_hook : NoneType
  in_features : Final[int] = 128
  out_features : Final[int] = 384
  def forward(self: __torch__.torch.nn.modules.linear.Linear,
    input: Tensor) -> Tensor:
    weight = self.weight
    bias = self.bias
    return torch.linear(input, weight, bias)
