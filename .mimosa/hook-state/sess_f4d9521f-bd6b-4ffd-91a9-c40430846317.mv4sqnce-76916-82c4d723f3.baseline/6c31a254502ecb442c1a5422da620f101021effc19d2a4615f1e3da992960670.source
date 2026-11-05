class ForwardTransformer(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  d_model : int
  style_norm : bool
  pos_encoder : __torch__.tacotron2.fastpitch_layers.___torch_mangle_15.PositionalEncoding
  layers : __torch__.torch.nn.modules.container.___torch_mangle_22.ModuleList
  norm : __torch__.torch.nn.modules.normalization.___torch_mangle_20.LayerNorm
  def forward(self: __torch__.tacotron2.fastpitch_layers.___torch_mangle_23.ForwardTransformer,
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
