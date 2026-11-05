class LengthRegulator(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  def forward(self: __torch__.tacotron2.forward_layers.LengthRegulator,
    x: Tensor,
    dur: Tensor) -> Tensor:
    _0 = __torch__.torch.nn.utils.rnn.pad_sequence
    _1 = torch.lt(dur, 0)
    _2 = torch.tensor(0., dtype=ops.prim.dtype(dur), device=ops.prim.device(dur))
    _3 = annotate(List[Optional[Tensor]], [_1])
    _4 = torch.index_put_(dur, _3, _2)
    _5 = annotate(List[Tensor], [])
    for i in range(torch.size(x, 0)):
      _6 = torch.select(x, 0, i)
      _7 = torch.add(torch.select(dur, 0, i), 0.5)
      x_exp = torch.repeat_interleave(_6, torch.to(_7, 4), 0)
      _8 = torch.append(_5, x_exp)
    return _0(_5, True, 0., )
