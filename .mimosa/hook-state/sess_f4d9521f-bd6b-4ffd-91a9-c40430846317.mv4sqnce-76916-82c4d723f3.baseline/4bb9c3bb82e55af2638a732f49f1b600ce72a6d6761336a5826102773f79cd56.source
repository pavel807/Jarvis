class ISTFT(Module):
  __parameters__ = []
  __buffers__ = ["window", ]
  window : Tensor
  training : bool
  _is_full_backward_hook : Optional[bool]
  padding : str
  n_fft : int
  hop_length : int
  win_length : int
  def forward(self: __torch__.tts_package.package_utils.ISTFT,
    spec: Tensor) -> Tensor:
    _0 = "Padding must be \'center\' or \'same\'."
    _1 = "AssertionError: Expected a 3D tensor as input"
    _2 = uninitialized(int)
    padding = self.padding
    if torch.eq(padding, "center"):
      n_fft = self.n_fft
      hop_length = self.hop_length
      win_length = self.win_length
      window = self.window
      _4 = torch.istft(spec, n_fft, hop_length, win_length, window)
      _3 = _4
    else:
      padding0 = self.padding
      if torch.eq(padding0, "same"):
        win_length0 = self.win_length
        hop_length0 = self.hop_length
        _5 = torch.sub(win_length0, hop_length0)
        pad = torch.floordiv(_5, 2)
      else:
        ops.prim.RaiseException(_0, "builtins.ValueError")
        pad = _2
      if torch.eq(torch.dim(spec), 3):
        pass
      else:
        ops.prim.RaiseException(_1)
      B, N, T, = torch.size(spec)
      n_fft0 = self.n_fft
      ifft = torch.fft_irfft(spec, n_fft0, 1, "backward")
      window0 = self.window
      _6 = torch.slice(torch.unsqueeze(window0, 0), 1)
      ifft0 = torch.mul(ifft, torch.unsqueeze(_6, 2))
      _7 = torch.sub(T, 1)
      hop_length1 = self.hop_length
      _8 = torch.mul(_7, hop_length1)
      win_length1 = self.win_length
      output_size = torch.add(_8, win_length1)
      win_length2 = self.win_length
      hop_length2 = self.hop_length
      _9 = __torch__.torch.nn.functional.fold(ifft0, [1, output_size], [1, win_length2], [1, 1], [0, 0], [1, hop_length2], )
      _10 = torch.select(torch.slice(_9), 1, 0)
      y = torch.slice(torch.select(_10, 1, 0), 1, pad, torch.neg(pad))
      window1 = self.window
      _11 = torch.expand(torch.square(window1), [1, T, -1])
      window_sq = torch.transpose(_11, 1, 2)
      win_length3 = self.win_length
      hop_length3 = self.hop_length
      _12 = __torch__.torch.nn.functional.fold(window_sq, [1, output_size], [1, win_length3], [1, 1], [0, 0], [1, hop_length3], )
      window_envelope = torch.slice(torch.squeeze(_12), 0, pad, torch.neg(pad))
      _13 = torch.gt(window_envelope, 9.9999999999999994e-12)
      if bool(torch.all(_13)):
        pass
      else:
        ops.prim.RaiseException("AssertionError: ")
      _3 = torch.div(y, window_envelope)
    return _3
