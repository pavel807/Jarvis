def pad_sequence(sequences: Union[Tensor, List[Tensor]],
    batch_first: bool=False,
    padding_value: float=0.) -> Tensor:
  _0 = __torch__.torch.jit._trace.is_tracing()
  _1 = isinstance(sequences, Tensor)
  if _1:
    sequences1 = unchecked_cast(Tensor, sequences)
    sequences0 = torch.unbind(sequences1)
  else:
    sequences2 = unchecked_cast(List[Tensor], sequences)
    sequences0 = sequences2
  _2 = torch.pad_sequence(sequences0, batch_first, padding_value)
  return _2
