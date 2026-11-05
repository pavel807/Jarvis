class PQMF(Module):
  __parameters__ = []
  __buffers__ = ["H", "G", "updown_filter", ]
  H : Tensor
  G : Tensor
  updown_filter : Tensor
  training : bool
  _is_full_backward_hook : Optional[bool]
  N : int
  taps : int
  cutoff : float
  beta : float
  def forward(self: __torch__.vocoder.hifigan.pqmf.PQMF,
    x: Tensor) -> Tensor:
    return (self).analysis(x, 1, )
  def analysis(self: __torch__.vocoder.hifigan.pqmf.PQMF,
    x: Tensor,
    resample: int=1) -> Tensor:
    if bool(resample):
      stride = self.N
    else:
      stride = 1
    H = self.H
    taps = self.taps
    _0 = torch.floordiv(taps, 2)
    x0 = torch.conv1d(x, H, None, [stride], [_0])
    _1 = torch.le(torch.max(torch.abs(x0)), 1.)
    if bool(_1):
      x1 = x0
    else:
      x1 = torch.tanh(x0)
    return x1
