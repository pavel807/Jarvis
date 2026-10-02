class ForwardTransformer(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  d_model : int
  style_norm : bool
  pos_encoder : __torch__.tacotron2.fastpitch_layers.PositionalEncoding
  layers : __torch__.torch.nn.modules.container.ModuleList
  norm : __torch__.torch.nn.modules.normalization.LayerNorm
  def forward(self: __torch__.tacotron2.fastpitch_layers.ForwardTransformer,
    x: Tensor,
    src_pad_mask: Optional[Tensor]=None,
    style: Optional[Tensor]=None) -> Tensor:
    x0 = torch.transpose(x, 0, 1)
    pos_encoder = self.pos_encoder
    x1 = (pos_encoder).forward(x0, )
    x2 = torch.transpose(x1, 0, 1)
    layers = self.layers
    _0 = getattr(layers, "0")
    _1 = getattr(layers, "1")
    _2 = getattr(layers, "2")
    _3 = getattr(layers, "3")
    x3 = (_0).forward(x2, src_pad_mask, style, )
    x4 = (_1).forward(x3, src_pad_mask, style, )
    x5 = (_2).forward(x4, src_pad_mask, style, )
    x6 = (_3).forward(x5, src_pad_mask, style, )
    norm = self.norm
    return (norm).forward(x6, )
class PositionalEncoding(Module):
  __parameters__ = ["scale", ]
  __buffers__ = ["pe", ]
  scale : Tensor
  pe : Tensor
  training : bool
  _is_full_backward_hook : Optional[bool]
  dropout : __torch__.torch.nn.modules.dropout.Dropout
  def forward(self: __torch__.tacotron2.fastpitch_layers.PositionalEncoding,
    x: Tensor) -> Tensor:
    scale = self.scale
    pe = self.pe
    _0 = torch.slice(pe, 0, None, torch.size(x, 0))
    _1 = torch.mul(scale, torch.slice(_0, 1))
    x7 = torch.add(x, _1)
    dropout = self.dropout
    return (dropout).forward(x7, )
class FFTBlock(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  use_linear : bool
  style_norm : bool
  conditional_cross_attention : bool
  cca : NoneType
  pitch_pred : bool
  self_attn : __torch__.torch.nn.modules.activation.MultiheadAttention
  dropout : __torch__.torch.nn.modules.dropout.Dropout
  conv1 : __torch__.torch.nn.modules.conv.Conv1d
  conv2 : __torch__.torch.nn.modules.conv.___torch_mangle_1.Conv1d
  norm1 : __torch__.torch.nn.modules.normalization.LayerNorm
  norm2 : __torch__.torch.nn.modules.normalization.LayerNorm
  dropout1 : __torch__.torch.nn.modules.dropout.Dropout
  dropout2 : __torch__.torch.nn.modules.dropout.Dropout
  activation : __torch__.torch.nn.modules.activation.ReLU
  def forward(self: __torch__.tacotron2.fastpitch_layers.FFTBlock,
    src: Tensor,
    src_pad_mask: Optional[Tensor]=None,
    style: Optional[Tensor]=None) -> Tensor:
    self_attn = self.self_attn
    _2 = (self_attn).forward(src, src, src, src_pad_mask, True, None, True, )
    src2 = (_2)[0]
    dropout1 = self.dropout1
    src0 = torch.add(src, (dropout1).forward(src2, ))
    norm1 = self.norm1
    src1 = (norm1).forward(src0, )
    use_linear = self.use_linear
    if torch.__not__(use_linear):
      src3 = torch.transpose(src1, 1, 2)
    else:
      src3 = src1
    conv1 = self.conv1
    src24 = (conv1).forward(src3, )
    activation = self.activation
    src25 = (activation).forward(src24, )
    conv2 = self.conv2
    src26 = (conv2).forward(src25, )
    pitch_pred = self.pitch_pred
    if pitch_pred:
      activation0 = self.activation
      src28 = (activation0).forward(src26, )
      dropout2 = self.dropout2
      src4 = torch.add(src3, (dropout2).forward(src28, ))
      src7 = torch.add(src4, src28)
    else:
      dropout20 = self.dropout2
      src5 = torch.add(src3, (dropout20).forward(src26, ))
      src7 = src5
    use_linear0 = self.use_linear
    if torch.__not__(use_linear0):
      src6 = torch.transpose(src7, 1, 2)
    else:
      src6 = src7
    norm2 = self.norm2
    return (norm2).forward(src6, )
class HourGlassTransformer(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  style_norm : bool
  shorten_factor : int
  pos_encoder : __torch__.tacotron2.fastpitch_layers.PositionalEncoding
  norm : __torch__.torch.nn.modules.normalization.LayerNorm
  pre_vanilla_layers : __torch__.torch.nn.modules.container.___torch_mangle_3.ModuleList
  shorten_layers : __torch__.torch.nn.modules.container.___torch_mangle_4.ModuleList
  post_vanilla_layers : __torch__.torch.nn.modules.container.___torch_mangle_3.ModuleList
  downsample : __torch__.tacotron2.fastpitch_layers.AvgPoolDownsample
  upsample : __torch__.tacotron2.fastpitch_layers.LinearUpsample
  def forward(self: __torch__.tacotron2.fastpitch_layers.HourGlassTransformer,
    x: Tensor,
    src_pad_mask: Optional[Tensor]=None,
    style: Optional[Tensor]=None) -> Tensor:
    _3 = __torch__.tacotron2.fastpitch_layers.pad_to_multiple
    _4 = uninitialized(Tensor)
    L = (torch.size(x))[1]
    x8 = torch.transpose(x, 0, 1)
    pos_encoder = self.pos_encoder
    x9 = (pos_encoder).forward(x8, )
    x10 = torch.transpose(x9, 0, 1)
    pre_vanilla_layers = self.pre_vanilla_layers
    _0 = getattr(pre_vanilla_layers, "0")
    x11 = (_0).forward(x10, src_pad_mask, style, )
    shorten_factor = self.shorten_factor
    x12 = _3(x11, shorten_factor, 0., )
    if torch.__isnot__(src_pad_mask, None):
      src_pad_mask1 = unchecked_cast(Tensor, src_pad_mask)
      shorten_factor0 = self.shorten_factor
      padded_mask0 = _3(src_pad_mask1, shorten_factor0, 0., )
      src_pad_mask0, padded_mask = src_pad_mask1, padded_mask0
    else:
      src_pad_mask0, padded_mask = src_pad_mask, src_pad_mask
    x_residual = torch.clone(x12)
    downsample = self.downsample
    downsampled = (downsample).forward(x12, )
    _5 = torch.__isnot__(src_pad_mask0, None)
    if _5:
      src_pad_mask3 = unchecked_cast(Tensor, src_pad_mask0)
      _6 = torch.__isnot__(padded_mask, None)
      if _6:
        padded_mask2 = unchecked_cast(Tensor, padded_mask)
        padded_mask1 = padded_mask2
      else:
        ops.prim.RaiseException("AssertionError: ")
        padded_mask1 = _4
      downsample0 = self.downsample
      _7 = (downsample0).forward(torch.to(padded_mask1, 6), )
      downsampled_mask, src_pad_mask2 = torch.gt(_7, 0), src_pad_mask3
    else:
      downsampled_mask, src_pad_mask2 = None, src_pad_mask0
    shorten_layers = self.shorten_layers
    _00 = getattr(shorten_layers, "0")
    _1 = getattr(shorten_layers, "1")
    x13 = (_00).forward(downsampled, downsampled_mask, style, )
    x14 = (_1).forward(downsampled, downsampled_mask, style, )
    upsample = self.upsample
    x15 = (upsample).forward(x14, )
    x16 = torch.add(x15, x_residual)
    x17 = torch.slice(torch.slice(x16), 1, None, L)
    post_vanilla_layers = self.post_vanilla_layers
    _01 = getattr(post_vanilla_layers, "0")
    x18 = (_01).forward(x17, src_pad_mask2, style, )
    norm = self.norm
    return (norm).forward(x18, )
class AvgPoolDownsample(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  avg_pool : __torch__.torch.nn.modules.pooling.AvgPool1d
  def forward(self: __torch__.tacotron2.fastpitch_layers.AvgPoolDownsample,
    x: Tensor) -> Tensor:
    _8 = torch.eq(torch.len(torch.size(x)), 2)
    if _8:
      avg_pool = self.avg_pool
      x19 = (avg_pool).forward(x, )
    else:
      x20 = torch.transpose(x, 1, 2)
      avg_pool0 = self.avg_pool
      x21 = (avg_pool0).forward(x20, )
      x19 = torch.transpose(x21, 1, 2)
    return x19
class LinearUpsample(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  dim : int
  proj : __torch__.torch.nn.modules.linear.Linear
  def forward(self: __torch__.tacotron2.fastpitch_layers.LinearUpsample,
    x: Tensor) -> Tensor:
    proj = self.proj
    x22 = (proj).forward(x, )
    _9 = (torch.size(x22))[0]
    dim = self.dim
    return torch.view(x22, [_9, -1, dim])
def make_token_len_mask(x: Tensor) -> Tensor:
  return torch.transpose(torch.eq(x, 0), 0, 1)
def pad_to_multiple(tensor: Tensor,
    multiple: int,
    value: float=0.) -> Tensor:
  t_shape = torch.len(torch.size(tensor))
  if torch.eq(t_shape, 2):
    seq_len = (torch.size(tensor))[-1]
  else:
    seq_len = (torch.size(tensor))[-2]
  m = torch.div(seq_len, multiple)
  m_dot = torch.remainder(seq_len, multiple)
  if torch.eq(m_dot, 0):
    _10 = tensor
  else:
    remainder = torch.sub(torch.mul(torch.ceil(m), multiple), seq_len)
    if torch.eq(t_shape, 2):
      _12 = torch.pad(tensor, [0, remainder], "constant", value)
      _11 = _12
    else:
      _13 = torch.pad(tensor, [0, 0, 0, remainder], "constant", value)
      _11 = _13
    _10 = _11
  return _10
