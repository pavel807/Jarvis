class Dropout(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : NoneType
  p : Final[float] = 0.5
  inplace : Final[bool] = False
  def forward(self: __torch__.torch.nn.modules.dropout.___torch_mangle_14.Dropout,
    input: Tensor) -> Tensor:
    _0 = __torch__.torch.nn.functional.dropout
    training = self.training
    return _0(input, 0.5, training, False, )
