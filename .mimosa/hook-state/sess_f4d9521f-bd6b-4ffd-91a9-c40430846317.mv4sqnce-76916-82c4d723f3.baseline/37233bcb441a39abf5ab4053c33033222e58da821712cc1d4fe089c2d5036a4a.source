class Conv1d(Module):
  __parameters__ = ["weight", "bias", ]
  __buffers__ = []
  weight : Tensor
  bias : Optional[Tensor]
  training : bool
  _is_full_backward_hook : NoneType
  transposed : bool
  _reversed_padding_repeated_twice : List[int]
  padding_mode : Final[str] = "zeros"
  output_padding : Final[Tuple[int]] = (0,)
  dilation : Final[Tuple[int]] = (1,)
  groups : Final[int] = 1
  padding : Final[Tuple[int]] = (1,)
  out_channels : Final[int] = 128
  in_channels : Final[int] = 1
  kernel_size : Final[Tuple[int]] = (3,)
  stride : Final[Tuple[int]] = (1,)
  def forward(self: __torch__.torch.nn.modules.conv.___torch_mangle_2.Conv1d,
    input: Tensor) -> Tensor:
    weight = self.weight
    bias = self.bias
    _0 = (self)._conv_forward(input, weight, bias, )
    return _0
  def _conv_forward(self: __torch__.torch.nn.modules.conv.___torch_mangle_2.Conv1d,
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor]) -> Tensor:
    _1 = torch.conv1d(input, weight, bias, [1], [1], [1])
    return _1
