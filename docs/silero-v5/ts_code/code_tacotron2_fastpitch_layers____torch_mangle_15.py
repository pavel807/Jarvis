class PositionalEncoding(Module):
  __parameters__ = ["scale", ]
  __buffers__ = ["pe", ]
  scale : Tensor
  pe : Tensor
  training : bool
  _is_full_backward_hook : Optional[bool]
  dropout : __torch__.torch.nn.modules.dropout.___torch_mangle_14.Dropout
  def forward(self: __torch__.tacotron2.fastpitch_layers.___torch_mangle_15.PositionalEncoding,
    x: Tensor) -> Tensor:
    scale = self.scale
    pe = self.pe
    _0 = torch.slice(pe, 0, None, torch.size(x, 0))
    _1 = torch.mul(scale, torch.slice(_0, 1))
    x0 = torch.add(x, _1)
    dropout = self.dropout
    return (dropout).forward(x0, )
