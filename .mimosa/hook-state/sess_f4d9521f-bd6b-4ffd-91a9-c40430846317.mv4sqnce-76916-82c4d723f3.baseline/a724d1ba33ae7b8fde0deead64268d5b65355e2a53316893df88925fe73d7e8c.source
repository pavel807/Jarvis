class TransformerSeriesPredictor(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  speaker_emb : str
  embedding : __torch__.torch.nn.modules.sparse.___torch_mangle_13.Embedding
  transformer : __torch__.tacotron2.fastpitch_layers.___torch_mangle_23.ForwardTransformer
  lin : __torch__.torch.nn.modules.linear.___torch_mangle_24.Linear
  speaker_embedding : __torch__.torch.nn.modules.sparse.___torch_mangle_25.Embedding
  emotion_embedding : __torch__.torch.nn.modules.sparse.___torch_mangle_27.Embedding
  def forward(self: __torch__.jit_forward_model.___torch_mangle_28.TransformerSeriesPredictor,
    x: Tensor,
    speaker_ids: Tensor,
    src_pad_mask: Optional[Tensor]=None,
    alpha: float=1.,
    coef_speakers: Optional[Tensor]=None) -> Tensor:
    embedding = self.embedding
    x0 = (embedding).forward(x, )
    seq_len = torch.size(x0, 1)
    if torch.__is__(coef_speakers, None):
      speaker_embedding = self.speaker_embedding
      speaker_embeds0 = (speaker_embedding).forward(speaker_ids, )
      speaker_embeds = speaker_embeds0
    else:
      coef_speakers0 = unchecked_cast(Tensor, coef_speakers)
      speaker_embedding0 = self.speaker_embedding
      weight = speaker_embedding0.weight
      _0 = torch.reshape(coef_speakers0, [-1, 1])
      speaker_embeds_weighted = torch.mul(weight, _0)
      _1 = annotate(List[Optional[Tensor]], [torch.ne(coef_speakers0, 0)])
      _2 = torch.index(speaker_embeds_weighted, _1)
      speaker_embeds1 = torch.unsqueeze(torch.sum(_2, [0]), 0)
      speaker_embeds = speaker_embeds1
    speaker_embeds2 = torch.repeat(torch.unsqueeze(speaker_embeds, 1), [1, seq_len, 1])
    x1 = torch.add(x0, speaker_embeds2)
    transformer = self.transformer
    x2 = (transformer).forward(x1, src_pad_mask, None, )
    lin = self.lin
    output = (lin).forward(x2, )
    return torch.div(output, alpha)
