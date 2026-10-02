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
  self_attn : __torch__.torch.nn.modules.activation.___torch_mangle_17.MultiheadAttention
  dropout : __torch__.torch.nn.modules.dropout.___torch_mangle_14.Dropout
  conv1 : __torch__.torch.nn.modules.conv.___torch_mangle_18.Conv1d
  conv2 : __torch__.torch.nn.modules.conv.___torch_mangle_19.Conv1d
  norm1 : __torch__.torch.nn.modules.normalization.___torch_mangle_20.LayerNorm
  norm2 : __torch__.torch.nn.modules.normalization.___torch_mangle_20.LayerNorm
  dropout1 : __torch__.torch.nn.modules.dropout.___torch_mangle_14.Dropout
  dropout2 : __torch__.torch.nn.modules.dropout.___torch_mangle_14.Dropout
  activation : __torch__.torch.nn.modules.activation.ReLU
  def forward(self: __torch__.tacotron2.fastpitch_layers.___torch_mangle_21.FFTBlock,
    src: Tensor,
    src_pad_mask: Optional[Tensor]=None,
    style: Optional[Tensor]=None) -> Tensor:
    self_attn = self.self_attn
    _0 = (self_attn).forward(src, src, src, src_pad_mask, True, None, True, )
    src2 = (_0)[0]
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
