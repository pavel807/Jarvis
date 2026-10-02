class ISTFTHead(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  out : __torch__.torch.nn.modules.linear.___torch_mangle_12.Linear
  istft : __torch__.tts_package.package_utils.ISTFT
  def forward(self: __torch__.vocoder.hifigan.vocos.vocos.heads.ISTFTHead,
    x: Tensor) -> Tensor:
    out = self.out
    x0 = torch.transpose((out).forward(x, ), 1, 2)
    mag, p, = torch.chunk(x0, 2, 1)
    mag0 = torch.exp(mag)
    mag1 = torch.clamp(mag0, None, 100.)
    x1 = torch.cos(p)
    y = torch.sin(p)
    S = torch.mul(mag1, torch.add(x1, torch.mul(y, 0.+1.j)))
    istft = self.istft
    return (istft).forward(S, )
