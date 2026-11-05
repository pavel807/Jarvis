class JitV4(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  multiband : bool
  pqmf_6 : __torch__.vocoder.hifigan.pqmf.PQMF
  pqmf_2 : __torch__.vocoder.hifigan.pqmf.PQMF
  backbone : __torch__.vocoder.hifigan.jit_vocoder.JitV4Backbone
  head : __torch__.vocoder.hifigan.vocos.vocos.heads.ISTFTHead
  def forward(self: __torch__.vocoder.hifigan.jit_vocoder.JitV4,
    x: Tensor,
    sample_rate: int,
    coef: float=0.,
    inference: bool=True) -> Tensor:
    backbone = self.backbone
    x0 = (backbone).forward(x, )
    head = self.head
    audio = torch.unsqueeze((head).forward(x0, ), 1)
    if torch.eq(sample_rate, 24000):
      pqmf_2 = self.pqmf_2
      _0 = torch.slice((pqmf_2).forward(audio, ))
      audio1 = torch.slice(torch.slice(_0, 1, None, 1), 2)
      audio0 = audio1
    else:
      audio0 = audio
    if torch.eq(sample_rate, 8000):
      pqmf_6 = self.pqmf_6
      _1 = torch.slice((pqmf_6).forward(audio0, ))
      audio3 = torch.slice(torch.slice(_1, 1, None, 1), 2)
      audio2 = audio3
    else:
      audio2 = audio0
    return torch.select(audio2, 0, 0)
class JitV4Backbone(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  input_channels : int
  embed : __torch__.torch.nn.modules.conv.___torch_mangle_6.Conv1d
  norm : __torch__.torch.nn.modules.normalization.___torch_mangle_7.LayerNorm
  convnext : __torch__.torch.nn.modules.container.___torch_mangle_11.ModuleList
  final_layer_norm : __torch__.torch.nn.modules.normalization.___torch_mangle_7.LayerNorm
  def forward(self: __torch__.vocoder.hifigan.jit_vocoder.JitV4Backbone,
    x: Tensor) -> Tensor:
    embed = self.embed
    x1 = (embed).forward(x, )
    norm = self.norm
    x2 = (norm).forward(torch.transpose(x1, 1, 2), )
    x3 = torch.transpose(x2, 1, 2)
    convnext = self.convnext
    _0 = getattr(convnext, "0")
    _1 = getattr(convnext, "1")
    _2 = getattr(convnext, "2")
    _3 = getattr(convnext, "3")
    _4 = getattr(convnext, "4")
    _5 = getattr(convnext, "5")
    _6 = getattr(convnext, "6")
    _7 = getattr(convnext, "7")
    x4 = (_0).forward(x3, )
    x5 = (_1).forward(x4, )
    x6 = (_2).forward(x5, )
    x7 = (_3).forward(x6, )
    x8 = (_4).forward(x7, )
    x9 = (_5).forward(x8, )
    x10 = (_6).forward(x9, )
    x11 = (_7).forward(x10, )
    final_layer_norm = self.final_layer_norm
    x12 = (final_layer_norm).forward(torch.transpose(x11, 1, 2), )
    return x12
class JitConvNeXtBlock(Module):
  __parameters__ = ["gamma", ]
  __buffers__ = []
  gamma : Tensor
  training : bool
  _is_full_backward_hook : Optional[bool]
  dwconv : __torch__.torch.nn.modules.conv.___torch_mangle_8.Conv1d
  norm : __torch__.torch.nn.modules.normalization.___torch_mangle_7.LayerNorm
  pwconv1 : __torch__.torch.nn.modules.linear.___torch_mangle_9.Linear
  act : __torch__.torch.nn.modules.activation.GELU
  pwconv2 : __torch__.torch.nn.modules.linear.___torch_mangle_10.Linear
  def forward(self: __torch__.vocoder.hifigan.jit_vocoder.JitConvNeXtBlock,
    x: Tensor) -> Tensor:
    dwconv = self.dwconv
    x13 = (dwconv).forward(x, )
    x14 = torch.transpose(x13, 1, 2)
    norm = self.norm
    x15 = (norm).forward(x14, )
    pwconv1 = self.pwconv1
    x16 = (pwconv1).forward(x15, )
    act = self.act
    x17 = (act).forward(x16, )
    pwconv2 = self.pwconv2
    x18 = (pwconv2).forward(x17, )
    gamma = self.gamma
    x19 = torch.mul(gamma, x18)
    x20 = torch.transpose(x19, 1, 2)
    return torch.add(x, x20)
