class MultiTTSModel(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  energy_predictor : NoneType
  punkt_symbols : List[int]
  point_id : int
  comma_ids : List[int]
  mean_std_coef : List[float]
  sil_value : float
  tacotron : __torch__.jit_forward_model.JitMultiForward
  vocoder : __torch__.vocoder.hifigan.jit_vocoder.JitV4
  dur_predictor : __torch__.jit_forward_model.JitDurPredictor
  pitch_predictor : __torch__.jit_forward_model.JitOldPitchPredictor
  def forward(self: __torch__.silero_vocoder.jit_model.MultiTTSModel,
    sequence: Tensor,
    speaker_ids: Tensor,
    sr: int=48000,
    symb_durs: Optional[Dict[int, int]]=None,
    durs_rate: Optional[Tensor]=None,
    pitch_coefs: Optional[Tensor]=None,
    gt_durs: Optional[Tensor]=None,
    gt_pitch: Optional[Tensor]=None,
    device: str="cpu",
    accent_word: int=-1,
    question: bool=False) -> Tuple[Tensor, Tensor]:
    _0 = __torch__.tacotron2.fastpitch_layers.make_token_len_mask
    _1 = _0(torch.transpose(sequence, 0, 1), )
    orig_mask = torch.to(_1, torch.device(device))
    if torch.__isnot__(symb_durs, None):
      symb_durs1 = unchecked_cast(Dict[int, int], symb_durs)
      _3 = torch.gt(torch.len(symb_durs1), 0)
      _2, symb_durs0 = _3, symb_durs1
    else:
      _2, symb_durs0 = False, symb_durs
    if _2:
      symb_durs3 = unchecked_cast(Dict[int, int], symb_durs0)
      support_ids0 = (self).get_support_ids(sequence, symb_durs3, )
      support_ids, symb_durs2 = support_ids0, symb_durs3
    else:
      support_ids, symb_durs2 = None, symb_durs0
    if torch.__isnot__(support_ids, None):
      support_ids2 = unchecked_cast(List[int], support_ids)
      support_sequence0 = (self).replace_comma(sequence, support_ids2, )
      support_ids1, support_sequence = support_ids2, support_sequence0
    else:
      support_ids1, support_sequence = support_ids, sequence
    if torch.__is__(gt_durs, None):
      if torch.__isnot__(symb_durs2, None):
        symb_durs6 = unchecked_cast(Dict[int, int], symb_durs2)
        _5 = torch.gt(torch.len(symb_durs6), 0)
        _4, symb_durs5 = _5, symb_durs6
      else:
        _4, symb_durs5 = False, symb_durs2
      if _4:
        symb_durs8 = unchecked_cast(Dict[int, int], symb_durs5)
        _7 = torch.__isnot__(support_ids1, None)
        _6, symb_durs7 = _7, symb_durs8
      else:
        _6, symb_durs7 = False, symb_durs5
      if _6:
        symb_durs10 = unchecked_cast(Dict[int, int], symb_durs7)
        support_ids4 = unchecked_cast(List[int], support_ids1)
        _8 = torch.eq((torch.size(sequence))[0], 1)
        if _8:
          pass
        else:
          ops.prim.RaiseException("AssertionError: ")
        support_sequence1 = torch.to(support_sequence, torch.device(device))
        dur_predictor = self.dur_predictor
        _9 = torch.cat([sequence, support_sequence1])
        _10 = torch.cat([orig_mask, orig_mask])
        pred_log_dur0 = (dur_predictor).forward(_9, speaker_ids, _10, 1., None, )
        pred_log_dur, symb_durs9, support_ids3 = pred_log_dur0, symb_durs10, support_ids4
      else:
        dur_predictor0 = self.dur_predictor
        pred_log_dur1 = (dur_predictor0).forward(sequence, speaker_ids, orig_mask, 1., None, )
        pred_log_dur, symb_durs9, support_ids3 = pred_log_dur1, symb_durs7, support_ids1
      pred_log_dur2 = torch.sub(torch.exp(pred_log_dur), 1)
      _11 = torch.lt(pred_log_dur2, 0.)
      _12 = torch.tensor(0., dtype=ops.prim.dtype(pred_log_dur2), device=ops.prim.device(pred_log_dur2))
      _13 = annotate(List[Optional[Tensor]], [_11])
      _14 = torch.index_put_(pred_log_dur2, _13, _12)
      dur_hat0 = torch.round(pred_log_dur2)
      _15 = torch.select(torch.select(dur_hat0, 0, 0), 0, 0)
      if bool(torch.gt(_15, 5)):
        _16 = torch.select(torch.select(dur_hat0, 0, 0), 0, 0)
        _17 = torch.tensor(5, dtype=ops.prim.dtype(_16), device=ops.prim.device(_16))
        _18 = torch.copy_(_16, _17)
      else:
        pass
      if torch.__isnot__(symb_durs9, None):
        symb_durs12 = unchecked_cast(Dict[int, int], symb_durs9)
        _20 = torch.gt(torch.len(symb_durs12), 0)
        _19, symb_durs11 = _20, symb_durs12
      else:
        _19, symb_durs11 = False, symb_durs9
      if _19:
        symb_durs14 = unchecked_cast(Dict[int, int], symb_durs11)
        _21 = torch.__isnot__(support_ids3, None)
        if _21:
          support_ids5 = unchecked_cast(List[int], support_ids3)
          sdurs = annotate(List[int], [])
          for _22 in range(torch.len(support_ids5)):
            sid = support_ids5[_22]
            _23 = torch.append(sdurs, symb_durs14[sid])
          pause_len = ops.prim.max(sdurs)
          dur_hat3 = (self).process_pauses(pred_log_dur2, support_ids5, pause_len, )
          dur_hat2 = dur_hat3
        else:
          dur_hat2 = dur_hat0
        dur_hat1, symb_durs13 = dur_hat2, symb_durs14
      else:
        dur_hat1, symb_durs13 = dur_hat0, symb_durs11
      if torch.__isnot__(durs_rate, None):
        durs_rate0 = unchecked_cast(Tensor, durs_rate)
        _24 = torch.eq(torch.size(durs_rate0), torch.size(dur_hat1))
        if _24:
          pass
        else:
          ops.prim.RaiseException("AssertionError: ")
        dur_hat5 = torch.div(dur_hat1, durs_rate0)
        dur_hat4 = torch.round(dur_hat5)
      else:
        dur_hat4 = dur_hat1
      _25 = torch.__isnot__(symb_durs13, None)
      if _25:
        symb_durs16 = unchecked_cast(Dict[int, int], symb_durs13)
        _27 = torch.gt(torch.len(symb_durs16), 0)
        _26, symb_durs15 = _27, symb_durs16
      else:
        _26, symb_durs15 = False, symb_durs13
      if _26:
        symb_durs18 = unchecked_cast(Dict[int, int], symb_durs15)
        _28 = torch.items(symb_durs18)
        for _29 in range(torch.len(_28)):
          sid0, sdur, = _28[_29]
          _30 = torch.select(torch.select(dur_hat4, 0, 0), 0, sid0)
          _31 = torch.tensor(sdur, dtype=ops.prim.dtype(_30), device=ops.prim.device(_30))
          _32 = torch.copy_(_30, _31)
        symb_durs17 : Optional[Dict[int, int]] = symb_durs18
      else:
        symb_durs17 = symb_durs15
      _33 = torch.select(torch.select(dur_hat4, 0, 0), 0, 0)
      if bool(torch.gt(_33, 5)):
        _34 = torch.select(torch.select(dur_hat4, 0, 0), 0, 0)
        _35 = torch.tensor(5, dtype=ops.prim.dtype(_34), device=ops.prim.device(_34))
        _36 = torch.copy_(_34, _35)
      else:
        pass
      _37 = torch.select(torch.select(dur_hat4, 0, 0), 0, -1)
      if bool(torch.gt(_37, 7)):
        _38 = torch.select(torch.select(dur_hat4, 0, 0), 0, -1)
        _39 = torch.tensor(7, dtype=ops.prim.dtype(_38), device=ops.prim.device(_38))
        _40 = torch.copy_(_38, _39)
      else:
        pass
      _41 = torch.select(torch.select(dur_hat4, 0, 0), 0, -2)
      _42 = torch.tensor(13, dtype=ops.prim.dtype(_41), device=ops.prim.device(_41))
      _43 = torch.copy_(_41, _42)
      _44 = torch.select(torch.select(dur_hat4, 0, 0), 0, -3)
      if bool(torch.gt(_44, 13)):
        _45 = torch.select(torch.select(dur_hat4, 0, 0), 0, -3)
        _46 = torch.tensor(13, dtype=ops.prim.dtype(_45), device=ops.prim.device(_45))
        _47 = torch.copy_(_45, _46)
      else:
        pass
      dur_hat, symb_durs4 = dur_hat4, symb_durs17
    else:
      dur_hat, symb_durs4 = unchecked_cast(Tensor, gt_durs), symb_durs2
    if torch.__is__(gt_pitch, None):
      _48 = torch.__isnot__(pitch_coefs, None)
      if _48:
        pitch_coefs0 = unchecked_cast(Tensor, pitch_coefs)
        pitch_coefs_exp0 = torch.select(pitch_coefs0, 0, 0)
        pitch_coefs_exp : Optional[Tensor] = pitch_coefs_exp0
      else:
        pitch_coefs_exp = None
      pitch_predictor = self.pitch_predictor
      pitch_hat0 = (pitch_predictor).forward(sequence, speaker_ids, orig_mask, 1., )
      _49 = torch.__isnot__(pitch_coefs_exp, None)
      if _49:
        pitch_coefs_exp1 = unchecked_cast(Tensor, pitch_coefs_exp)
        _50 = torch.unsqueeze(pitch_coefs_exp1, 0)
        pitch_hat2 = (self).update_pitch_coef(pitch_hat0, _50, speaker_ids, )
        pitch_hat1 = pitch_hat2
      else:
        pitch_hat1 = pitch_hat0
      pitch_hat = pitch_hat1
    else:
      pitch_hat = unchecked_cast(Tensor, gt_pitch)
    durs_cum = torch.cumsum(torch.pad(dur_hat, [1, 0]), 0)
    _51 = torch.len(dur_hat)
    _52 = torch.slice(durs_cum, 0, None, -1)
    _53 = torch.slice(durs_cum, 0, 1)
    _54 = [_51, torch.len(_52), torch.len(_53), torch.len(sequence)]
    for idx in range(ops.prim.min(_54)):
      a = torch.select(_52, 0, idx)
      b = torch.select(_53, 0, idx)
      char_id = torch.select(sequence, 0, idx)
      punkt_symbols = self.punkt_symbols
      _55 = torch.__contains__(punkt_symbols, annotate(int, char_id))
      if _55:
        _56 = torch.slice(pitch_hat, 0, annotate(int, a), annotate(int, b))
        _57 = torch.tensor(0, dtype=ops.prim.dtype(_56), device=ops.prim.device(_56))
        _58 = torch.copy_(_56, _57)
      else:
        pass
    tacotron = self.tacotron
    mel_outputs = (tacotron).forward(sequence, speaker_ids, orig_mask, dur_hat, pitch_hat, None, )
    if torch.__isnot__(symb_durs4, None):
      symb_durs20 = unchecked_cast(Dict[int, int], symb_durs4)
      _60 = torch.gt(torch.len(symb_durs20), 0)
      _59, symb_durs19 = _60, symb_durs20
    else:
      _59, symb_durs19 = False, symb_durs4
    if _59:
      symb_durs21 = unchecked_cast(Dict[int, int], symb_durs19)
      mel_outputs1 = (self).fx_pauses(mel_outputs, dur_hat, symb_durs21, )
      mel_outputs0 = mel_outputs1
    else:
      mel_outputs0 = mel_outputs
    vocoder = self.vocoder
    audio = (vocoder).forward(mel_outputs0, sr, 0., True, )
    return (audio, dur_hat)
  def get_support_ids(self: __torch__.silero_vocoder.jit_model.MultiTTSModel,
    sequence: Tensor,
    symb_durs: Dict[int, int]) -> Optional[List[int]]:
    temp_ids = annotate(List[int], [])
    _61 = torch.keys(symb_durs)
    for _62 in range(torch.len(_61)):
      sid = _61[_62]
      _63 = torch.select(torch.select(sequence, 0, 0), 0, sid)
      comma_ids = self.comma_ids
      _64 = torch.__contains__(comma_ids, annotate(int, _63))
      if _64:
        _65 = torch.append(temp_ids, sid)
      else:
        pass
    if torch.gt(torch.len(temp_ids), 0):
      support_ids : Optional[List[int]] = temp_ids
    else:
      support_ids = None
    return support_ids
  def replace_comma(self: __torch__.silero_vocoder.jit_model.MultiTTSModel,
    sequence: Tensor,
    support_ids: List[int]) -> Tensor:
    support_sequence = torch.clone(sequence)
    point_id = self.point_id
    _66 = torch.tensor(support_ids, dtype=4)
    _67 = torch.slice(support_sequence)
    _68 = torch.tensor(point_id, dtype=ops.prim.dtype(_67), device=ops.prim.device(_67))
    _69 = annotate(List[Optional[Tensor]], [None, _66])
    _70 = torch.index_put_(_67, _69, _68)
    return support_sequence
  def process_pauses(self: __torch__.silero_vocoder.jit_model.MultiTTSModel,
    comma_point_durs: Tensor,
    comma_ids: List[int],
    pause_len: int) -> Tensor:
    _71 = torch.tensor(comma_ids, dtype=4)
    _72 = torch.slice(comma_point_durs)
    _73 = annotate(List[Optional[Tensor]], [None, _71])
    _74, _75 = torch.min(torch.index(_72, _73), 1)
    comma_len = torch.item(torch.select(_74, 0, 0))
    point_len = torch.item(torch.select(_74, 0, 1))
    if torch.gt(comma_len, point_len):
      _76 = True
    else:
      _76 = torch.le(pause_len, comma_len)
    if _76:
      final_dur0 = torch.select(comma_point_durs, 0, 0)
      final_dur = final_dur0
    else:
      if torch.ge(pause_len, point_len):
        final_dur2 = torch.select(comma_point_durs, 0, 1)
        final_dur1 = final_dur2
      else:
        point_cf = torch.div(torch.sub(pause_len, comma_len), torch.sub(point_len, comma_len))
        comma_cf = torch.div(torch.sub(point_len, pause_len), torch.sub(point_len, comma_len))
        _77 = torch.select(comma_point_durs, 0, 0)
        _78 = torch.mul(_77, comma_cf)
        _79 = torch.select(comma_point_durs, 0, 1)
        final_dur3 = torch.add(_78, torch.mul(_79, point_cf))
        final_dur1 = final_dur3
      final_dur = final_dur1
    final_dur4 = torch.unsqueeze(final_dur, 0)
    return torch.round(final_dur4)
  def update_pitch_coef(self: __torch__.silero_vocoder.jit_model.MultiTTSModel,
    norm_pitch: Tensor,
    pitch_coef: Tensor,
    speaker_ids: Tensor) -> Tensor:
    _80 = torch.lt(ops.prim.abs(norm_pitch), 0.001)
    _81 = torch.tensor(0., dtype=ops.prim.dtype(norm_pitch), device=ops.prim.device(norm_pitch))
    _82 = annotate(List[Optional[Tensor]], [_80])
    _83 = torch.index_put_(norm_pitch, _82, _81)
    norm_pitch_coefd = torch.mul(norm_pitch, pitch_coef)
    _84 = torch.eq(torch.select(pitch_coef, 0, 0), 0)
    _85 = (torch.where(_84))[0]
    _86 = torch.slice(pitch_coef)
    _87 = torch.tensor(1., dtype=ops.prim.dtype(_86), device=ops.prim.device(_86))
    _88 = annotate(List[Optional[Tensor]], [None, _85])
    _89 = torch.index_put_(_86, _88, _87)
    mean_std_coef = self.mean_std_coef
    _90 = mean_std_coef[int(torch.item(speaker_ids))]
    mean_std_shift = torch.mul(torch.sub(pitch_coef, 1.), _90)
    _91 = torch.select(norm_pitch_coefd, 0, 0)
    _92 = torch.eq(_91, 0)
    _93 = torch.tensor(0, dtype=ops.prim.dtype(mean_std_shift), device=ops.prim.device(mean_std_shift))
    _94 = annotate(List[Optional[Tensor]], [_92])
    _95 = torch.index_put_(mean_std_shift, _94, _93)
    norm_pitch_shift = torch.add(norm_pitch_coefd, mean_std_shift)
    return norm_pitch_shift
  def fx_pauses(self: __torch__.silero_vocoder.jit_model.MultiTTSModel,
    mel_outputs: Tensor,
    dur_hat: Tensor,
    symb_durs: Dict[int, int]) -> Tensor:
    _96 = torch.keys(symb_durs)
    for _97 in range(torch.len(_96)):
      sid = _96[_97]
      _98 = torch.slice(torch.select(dur_hat, 0, 0), 0, None, sid)
      dur_sum = int(torch.item(torch.sum(_98)))
      _99 = torch.select(torch.select(dur_hat, 0, 0), 0, sid)
      sdur = int(torch.item(_99))
      if torch.gt(sdur, 10):
        sil_value = self.sil_value
        _100 = torch.slice(torch.slice(mel_outputs), 1)
        _101 = torch.slice(_100, 2, torch.add(dur_sum, 10), torch.add(dur_sum, sdur))
        _102 = torch.tensor(sil_value, dtype=ops.prim.dtype(_101), device=ops.prim.device(_101))
        _103 = torch.copy_(_101, _102)
      else:
        sil_value0 = self.sil_value
        _104 = torch.slice(torch.slice(mel_outputs), 1)
        _105 = torch.slice(_104, 2, dur_sum, torch.add(dur_sum, sdur))
        _106 = torch.tensor(sil_value0, dtype=ops.prim.dtype(_105), device=ops.prim.device(_105))
        _107 = torch.copy_(_105, _106)
    return mel_outputs
