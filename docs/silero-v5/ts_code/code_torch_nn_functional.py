def embedding(input: Tensor,
    weight: Tensor,
    padding_idx: Optional[int]=None,
    max_norm: Optional[float]=None,
    norm_type: float=2.,
    scale_grad_by_freq: bool=False,
    sparse: bool=False) -> Tensor:
  _0 = "AssertionError: Padding_idx must be within num_embeddings"
  if torch.__isnot__(padding_idx, None):
    padding_idx1 = unchecked_cast(int, padding_idx)
    if torch.gt(padding_idx1, 0):
      _1 = torch.lt(padding_idx1, torch.size(weight, 0))
      if _1:
        pass
      else:
        ops.prim.RaiseException(_0)
      padding_idx2 = padding_idx1
    else:
      if torch.lt(padding_idx1, 0):
        _2 = torch.neg(torch.size(weight, 0))
        if torch.ge(padding_idx1, _2):
          pass
        else:
          ops.prim.RaiseException(_0)
        padding_idx4 = torch.add(torch.size(weight, 0), padding_idx1)
        padding_idx3 = padding_idx4
      else:
        padding_idx3 = padding_idx1
      padding_idx2 = padding_idx3
    padding_idx0 = padding_idx2
  else:
    padding_idx0 = -1
  if torch.__isnot__(max_norm, None):
    input0 = torch.contiguous(input)
  else:
    input0 = input
  _3 = torch.embedding(weight, input0, padding_idx0, scale_grad_by_freq, sparse)
  return _3
def dropout(input: Tensor,
    p: float=0.5,
    training: bool=True,
    inplace: bool=False) -> Tensor:
  _4 = "dropout probability has to be between 0 and 1, but got {}"
  if torch.lt(p, 0.):
    _5 = True
  else:
    _5 = torch.gt(p, 1.)
  if _5:
    ops.prim.RaiseException(torch.format(_4, p), "builtins.ValueError")
  else:
    pass
  if inplace:
    _6 = torch.dropout_(input, p, training)
  else:
    _6 = torch.dropout(input, p, training)
  return _6
def multi_head_attention_forward(query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool=True,
    key_padding_mask: Optional[Tensor]=None,
    need_weights: bool=True,
    attn_mask: Optional[Tensor]=None,
    use_separate_proj_weight: bool=False,
    q_proj_weight: Optional[Tensor]=None,
    k_proj_weight: Optional[Tensor]=None,
    v_proj_weight: Optional[Tensor]=None,
    static_k: Optional[Tensor]=None,
    static_v: Optional[Tensor]=None,
    average_attn_weights: bool=True) -> Tuple[Tensor, Optional[Tensor]]:
  _7 = __torch__.torch.nn.functional._mha_shape_check
  _8 = "was expecting embedding dimension of {}, but got {}"
  _9 = "embed_dim {} not divisible by num_heads {}"
  _10 = "key\'s sequence and batch dims {} do not match value\'s {}"
  _11 = "key shape {} does not match value shape {}"
  _12 = "AssertionError: use_separate_proj_weight is False but in_proj_weight is None"
  _13 = __torch__.torch.nn.functional._in_projection_packed
  _14 = "AssertionError: use_separate_proj_weight is True but q_proj_weight is None"
  _15 = "AssertionError: use_separate_proj_weight is True but k_proj_weight is None"
  _16 = "AssertionError: use_separate_proj_weight is True but v_proj_weight is None"
  _17 = __torch__.torch.nn.functional._in_projection
  _18 = "Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
  _19 = "Only float, byte, and bool types are supported for attn_mask, not {}"
  _20 = "The shape of the 2D attn_mask is {}, but should be {}."
  _21 = "The shape of the 3D attn_mask is {}, but should be {}."
  _22 = "attn_mask\'s dimension {} is not supported"
  _23 = "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
  _24 = "AssertionError: bias cannot be added to static key."
  _25 = "AssertionError: bias cannot be added to static value."
  _26 = "expecting static_k.size(0) of {}, but got {}"
  _27 = "expecting static_k.size(2) of {}, but got {}"
  _28 = "expecting static_v.size(0) of {}, but got {}"
  _29 = "expecting static_v.size(2) of {}, but got {}"
  _30 = "expecting key_padding_mask shape of {}, but got {}"
  _31 = __torch__.torch.nn.functional._scaled_dot_product_attention
  _32 = uninitialized(Optional[Tensor])
  _33 = uninitialized(Optional[Tensor])
  _34 = uninitialized(Tensor)
  is_batched = _7(query, key, value, key_padding_mask, attn_mask, num_heads, )
  if torch.__not__(is_batched):
    query1 = torch.unsqueeze(query, 1)
    key1 = torch.unsqueeze(key, 1)
    value1 = torch.unsqueeze(value, 1)
    _35 = torch.__isnot__(key_padding_mask, None)
    if _35:
      key_padding_mask2 = unchecked_cast(Tensor, key_padding_mask)
      key_padding_mask3 = torch.unsqueeze(key_padding_mask2, 0)
      key_padding_mask1 : Optional[Tensor] = key_padding_mask3
    else:
      key_padding_mask1 = key_padding_mask
    query0, key0, value0, key_padding_mask0 = query1, key1, value1, key_padding_mask1
  else:
    query0, key0, value0, key_padding_mask0 = query, key, value, key_padding_mask
  tgt_len, bsz, embed_dim, = torch.size(query0)
  src_len, _36, _37, = torch.size(key0)
  _38 = torch.eq(embed_dim, embed_dim_to_check)
  if _38:
    pass
  else:
    _39 = torch.format(_8, embed_dim_to_check, embed_dim)
    ops.prim.RaiseException(torch.add("AssertionError: ", _39))
  embed_dim0 = unchecked_cast(int, embed_dim)
  head_dim = torch.floordiv(embed_dim0, num_heads)
  _40 = torch.eq(torch.mul(head_dim, num_heads), embed_dim0)
  if _40:
    pass
  else:
    _41 = torch.format(_9, embed_dim0, num_heads)
    ops.prim.RaiseException(torch.add("AssertionError: ", _41))
  if use_separate_proj_weight:
    _42 = torch.slice(torch.size(key0), None, 2)
    _43 = torch.slice(torch.size(value0), None, 2)
    if torch.eq(_42, _43):
      pass
    else:
      _44 = torch.slice(torch.size(key0), None, 2)
      _45 = torch.slice(torch.size(value0), None, 2)
      _46 = torch.add("AssertionError: ", torch.format(_10, _44, _45))
      ops.prim.RaiseException(_46)
  else:
    _47 = torch.eq(torch.size(key0), torch.size(value0))
    if _47:
      pass
    else:
      _48 = torch.format(_11, torch.size(key0), torch.size(value0))
      _49 = torch.add("AssertionError: ", _48)
      ops.prim.RaiseException(_49)
  _50 = torch.__not__(use_separate_proj_weight)
  if _50:
    _51 = torch.__isnot__(in_proj_weight, None)
    if _51:
      in_proj_weight1 = unchecked_cast(Tensor, in_proj_weight)
      in_proj_weight0 = in_proj_weight1
    else:
      ops.prim.RaiseException(_12)
      in_proj_weight0 = _34
    _52 = _13(query0, key0, value0, in_proj_weight0, in_proj_bias, )
    q0, k0, v0, = _52
    k, v, q = k0, v0, q0
  else:
    _53 = torch.__isnot__(q_proj_weight, None)
    if _53:
      q_proj_weight1 = unchecked_cast(Tensor, q_proj_weight)
      q_proj_weight0 = q_proj_weight1
    else:
      ops.prim.RaiseException(_14)
      q_proj_weight0 = _34
    _54 = torch.__isnot__(k_proj_weight, None)
    if _54:
      k_proj_weight1 = unchecked_cast(Tensor, k_proj_weight)
      k_proj_weight0 = k_proj_weight1
    else:
      ops.prim.RaiseException(_15)
      k_proj_weight0 = _34
    _55 = torch.__isnot__(v_proj_weight, None)
    if _55:
      v_proj_weight1 = unchecked_cast(Tensor, v_proj_weight)
      v_proj_weight0 = v_proj_weight1
    else:
      ops.prim.RaiseException(_16)
      v_proj_weight0 = _34
    if torch.__is__(in_proj_bias, None):
      b_q, b_k, b_v = None, None, None
    else:
      in_proj_bias0 = unchecked_cast(Tensor, in_proj_bias)
      b_q0, b_k0, b_v0, = torch.chunk(in_proj_bias0, 3)
      b_q, b_k, b_v = b_q0, b_k0, b_v0
    _56 = _17(query0, key0, value0, q_proj_weight0, k_proj_weight0, v_proj_weight0, b_q, b_k, b_v, )
    q1, k1, v1, = _56
    k, v, q = k1, v1, q1
  if torch.__isnot__(attn_mask, None):
    attn_mask1 = unchecked_cast(Tensor, attn_mask)
    _57 = torch.eq(ops.prim.dtype(attn_mask1), 0)
    if _57:
      torch.warn(_18)
      attn_mask2 = torch.to(attn_mask1, 11)
    else:
      _58 = torch.is_floating_point(attn_mask1)
      if _58:
        _59 = True
      else:
        _60 = torch.eq(ops.prim.dtype(attn_mask1), 11)
        _59 = _60
      if _59:
        pass
      else:
        _61 = torch.format(_19, ops.prim.dtype(attn_mask1))
        _62 = torch.add("AssertionError: ", _61)
        ops.prim.RaiseException(_62)
      attn_mask2 = attn_mask1
    if torch.eq(torch.dim(attn_mask2), 2):
      correct_2d_size = (tgt_len, src_len)
      _63 = torch.ne(torch.size(attn_mask2), [tgt_len, src_len])
      if _63:
        _64 = torch.format(_20, torch.size(attn_mask2), correct_2d_size)
        ops.prim.RaiseException(_64, "builtins.RuntimeError")
      else:
        pass
      attn_mask3 = torch.unsqueeze(attn_mask2, 0)
    else:
      _65 = torch.eq(torch.dim(attn_mask2), 3)
      if _65:
        _66 = torch.mul(bsz, num_heads)
        correct_3d_size = (_66, tgt_len, src_len)
        _67 = torch.ne(torch.size(attn_mask2), [_66, tgt_len, src_len])
        if _67:
          _68 = torch.format(_21, torch.size(attn_mask2), correct_3d_size)
          ops.prim.RaiseException(_68, "builtins.RuntimeError")
        else:
          pass
      else:
        _69 = torch.format(_22, torch.dim(attn_mask2))
        ops.prim.RaiseException(_69, "builtins.RuntimeError")
      attn_mask3 = attn_mask2
    attn_mask0 : Optional[Tensor] = attn_mask3
  else:
    attn_mask0 = attn_mask
  _70 = torch.__isnot__(key_padding_mask0, None)
  if _70:
    key_padding_mask5 = unchecked_cast(Tensor, key_padding_mask0)
    _72 = torch.eq(ops.prim.dtype(key_padding_mask5), 0)
    _71, key_padding_mask4 = _72, key_padding_mask5
  else:
    _71, key_padding_mask4 = False, key_padding_mask0
  if _71:
    key_padding_mask7 = unchecked_cast(Tensor, key_padding_mask4)
    torch.warn(_23)
    key_padding_mask6 : Optional[Tensor] = torch.to(key_padding_mask7, 11)
  else:
    key_padding_mask6 = key_padding_mask4
  if torch.__isnot__(bias_k, None):
    bias_k1 = unchecked_cast(Tensor, bias_k)
    _73, bias_k0 = torch.__isnot__(bias_v, None), bias_k1
  else:
    _73, bias_k0 = False, bias_k
  if _73:
    bias_k2 = unchecked_cast(Tensor, bias_k0)
    bias_v0 = unchecked_cast(Tensor, bias_v)
    if torch.__is__(static_k, None):
      static_k1 : Optional[Tensor] = static_k
    else:
      ops.prim.RaiseException(_24)
      static_k1 = _33
    if torch.__is__(static_v, None):
      static_v1 : Optional[Tensor] = static_v
    else:
      ops.prim.RaiseException(_25)
      static_v1 = _32
    _74 = [k, torch.repeat(bias_k2, [1, bsz, 1])]
    k3 = torch.cat(_74)
    _75 = [v, torch.repeat(bias_v0, [1, bsz, 1])]
    v3 = torch.cat(_75)
    if torch.__isnot__(attn_mask0, None):
      attn_mask6 = unchecked_cast(Tensor, attn_mask0)
      attn_mask5 : Optional[Tensor] = torch.pad(attn_mask6, [0, 1])
    else:
      attn_mask5 = attn_mask0
    _76 = torch.__isnot__(key_padding_mask6, None)
    if _76:
      key_padding_mask10 = unchecked_cast(Tensor, key_padding_mask6)
      key_padding_mask11 = torch.pad(key_padding_mask10, [0, 1])
      key_padding_mask9 : Optional[Tensor] = key_padding_mask11
    else:
      key_padding_mask9 = key_padding_mask6
    static_k0, k2, static_v0, v2, attn_mask4, key_padding_mask8 = static_k1, k3, static_v1, v3, attn_mask5, key_padding_mask9
  else:
    if torch.__is__(bias_k0, None):
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
    if torch.__is__(bias_v, None):
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
    static_k0, k2, static_v0, v2, attn_mask4, key_padding_mask8 = static_k, k, static_v, v, attn_mask0, key_padding_mask6
  _77 = torch.contiguous(q)
  _78 = [tgt_len, torch.mul(bsz, num_heads), head_dim]
  q2 = torch.transpose(torch.view(_77, _78), 0, 1)
  if torch.__is__(static_k0, None):
    _79 = torch.contiguous(k2)
    _80 = [(torch.size(k2))[0], torch.mul(bsz, num_heads), head_dim]
    k5 = torch.transpose(torch.view(_79, _80), 0, 1)
    k4 = k5
  else:
    static_k2 = unchecked_cast(Tensor, static_k0)
    _81 = torch.eq(torch.size(static_k2, 0), torch.mul(bsz, num_heads))
    if _81:
      pass
    else:
      _82 = torch.format(_26, torch.mul(bsz, num_heads), torch.size(static_k2, 0))
      _83 = torch.add("AssertionError: ", _82)
      ops.prim.RaiseException(_83)
    _84 = torch.eq(torch.size(static_k2, 2), head_dim)
    if _84:
      pass
    else:
      _85 = torch.format(_27, head_dim, torch.size(static_k2, 2))
      _86 = torch.add("AssertionError: ", _85)
      ops.prim.RaiseException(_86)
    k4 = static_k2
  if torch.__is__(static_v0, None):
    _87 = torch.contiguous(v2)
    _88 = [(torch.size(v2))[0], torch.mul(bsz, num_heads), head_dim]
    v5 = torch.transpose(torch.view(_87, _88), 0, 1)
    v4 = v5
  else:
    static_v2 = unchecked_cast(Tensor, static_v0)
    _89 = torch.eq(torch.size(static_v2, 0), torch.mul(bsz, num_heads))
    if _89:
      pass
    else:
      _90 = torch.format(_28, torch.mul(bsz, num_heads), torch.size(static_v2, 0))
      _91 = torch.add("AssertionError: ", _90)
      ops.prim.RaiseException(_91)
    _92 = torch.eq(torch.size(static_v2, 2), head_dim)
    if _92:
      pass
    else:
      _93 = torch.format(_29, head_dim, torch.size(static_v2, 2))
      _94 = torch.add("AssertionError: ", _93)
      ops.prim.RaiseException(_94)
    v4 = static_v2
  if add_zero_attn:
    _95 = torch.mul(bsz, num_heads)
    _96 = ops.prim.dtype(k4)
    _97 = ops.prim.device(k4)
    _98 = torch.zeros([_95, 1, head_dim], dtype=_96, layout=None, device=_97)
    k7 = torch.cat([k4, _98], 1)
    _99 = ops.prim.dtype(v4)
    _100 = ops.prim.device(v4)
    _101 = torch.zeros([_95, 1, head_dim], dtype=_99, layout=None, device=_100)
    v7 = torch.cat([v4, _101], 1)
    if torch.__isnot__(attn_mask4, None):
      attn_mask9 = unchecked_cast(Tensor, attn_mask4)
      attn_mask8 : Optional[Tensor] = torch.pad(attn_mask9, [0, 1])
    else:
      attn_mask8 = attn_mask4
    _102 = torch.__isnot__(key_padding_mask8, None)
    if _102:
      key_padding_mask14 = unchecked_cast(Tensor, key_padding_mask8)
      key_padding_mask15 = torch.pad(key_padding_mask14, [0, 1])
      key_padding_mask13 : Optional[Tensor] = key_padding_mask15
    else:
      key_padding_mask13 = key_padding_mask8
    k6, key_padding_mask12, attn_mask7, v6 = k7, key_padding_mask13, attn_mask8, v7
  else:
    k6, key_padding_mask12, attn_mask7, v6 = k4, key_padding_mask8, attn_mask4, v4
  src_len0 = torch.size(k6, 1)
  _103 = torch.__isnot__(key_padding_mask12, None)
  if _103:
    key_padding_mask16 = unchecked_cast(Tensor, key_padding_mask12)
    _104 = torch.eq(torch.size(key_padding_mask16), [bsz, src_len0])
    if _104:
      pass
    else:
      _105 = torch.format(_30, (bsz, src_len0), torch.size(key_padding_mask16))
      _106 = torch.add("AssertionError: ", _105)
      ops.prim.RaiseException(_106)
    _107 = torch.view(key_padding_mask16, [bsz, 1, 1, src_len0])
    _108 = torch.expand(_107, [-1, num_heads, -1, -1])
    _109 = [torch.mul(bsz, num_heads), 1, src_len0]
    key_padding_mask17 = torch.reshape(_108, _109)
    if torch.__is__(attn_mask7, None):
      attn_mask11 = key_padding_mask17
    else:
      attn_mask12 = unchecked_cast(Tensor, attn_mask7)
      _110 = torch.eq(ops.prim.dtype(attn_mask12), 11)
      if _110:
        attn_mask14 = torch.logical_or(attn_mask12, key_padding_mask17)
        attn_mask13 = attn_mask14
      else:
        attn_mask15 = torch.masked_fill(attn_mask12, key_padding_mask17, -inf)
        attn_mask13 = attn_mask15
      attn_mask11 = attn_mask13
    attn_mask10 : Optional[Tensor] = attn_mask11
  else:
    attn_mask10 = attn_mask7
  if torch.__isnot__(attn_mask10, None):
    attn_mask17 = unchecked_cast(Tensor, attn_mask10)
    _112 = torch.eq(ops.prim.dtype(attn_mask17), 11)
    _111, attn_mask16 = _112, attn_mask17
  else:
    _111, attn_mask16 = False, attn_mask10
  if _111:
    attn_mask19 = unchecked_cast(Tensor, attn_mask16)
    new_attn_mask = torch.zeros_like(attn_mask19, dtype=ops.prim.dtype(q2))
    _113 = torch.masked_fill_(new_attn_mask, attn_mask19, -inf)
    attn_mask18 : Optional[Tensor] = new_attn_mask
  else:
    attn_mask18 = attn_mask16
  if torch.__not__(training):
    dropout_p0 = 0.
  else:
    dropout_p0 = dropout_p
  _114 = _31(q2, k6, v6, attn_mask18, dropout_p0, )
  attn_output, attn_output_weights, = _114
  _115 = torch.contiguous(torch.transpose(attn_output, 0, 1))
  attn_output0 = torch.view(_115, [torch.mul(tgt_len, bsz), embed_dim0])
  attn_output1 = torch.linear(attn_output0, out_proj_weight, out_proj_bias)
  _116 = [tgt_len, bsz, torch.size(attn_output1, 1)]
  attn_output2 = torch.view(attn_output1, _116)
  if need_weights:
    attn_output_weights0 = torch.view(attn_output_weights, [bsz, num_heads, tgt_len, src_len0])
    if average_attn_weights:
      _118 = torch.sum(attn_output_weights0, [1])
      attn_output_weights1 = torch.div(_118, num_heads)
    else:
      attn_output_weights1 = attn_output_weights0
    if torch.__not__(is_batched):
      attn_output4 = torch.squeeze(attn_output2, 1)
      attn_output_weights3 = torch.squeeze(attn_output_weights1, 0)
      attn_output3, attn_output_weights2 = attn_output4, attn_output_weights3
    else:
      attn_output3, attn_output_weights2 = attn_output2, attn_output_weights1
    _119 = (attn_output3, attn_output_weights2)
    _117 = _119
  else:
    if torch.__not__(is_batched):
      attn_output5 = torch.squeeze(attn_output2, 1)
    else:
      attn_output5 = attn_output2
    _117 = (attn_output5, None)
  return _117
def layer_norm(input: Tensor,
    normalized_shape: List[int],
    weight: Optional[Tensor]=None,
    bias: Optional[Tensor]=None,
    eps: float=1.0000000000000001e-05) -> Tensor:
  _120 = torch.layer_norm(input, normalized_shape, weight, bias, eps)
  return _120
def relu(input: Tensor,
    inplace: bool=False) -> Tensor:
  if inplace:
    result = torch.relu_(input)
  else:
    result = torch.relu(input)
  return result
def fold(input: Tensor,
    output_size: List[int],
    kernel_size: List[int],
    dilation: List[int]=[1, 1],
    padding: List[int]=[0, 0],
    stride: List[int]=[1, 1]) -> Tensor:
  _121 = "Input Error: Only unbatched (2D) or batched (3D) input Tensorsare supported (got {}D)"
  _122 = uninitialized(Tensor)
  if torch.eq(torch.dim(input), 3):
    _123 = True
  else:
    _123 = torch.eq(torch.dim(input), 2)
  if _123:
    _125 = torch.col2im(input, output_size, kernel_size, dilation, padding, stride)
    _124 = _125
  else:
    _126 = torch.format(_121, torch.dim(input))
    ops.prim.RaiseException(_126, "builtins.NotImplementedError")
    _124 = _122
  return _124
def _mha_shape_check(query: Tensor,
    key: Tensor,
    value: Tensor,
    key_padding_mask: Optional[Tensor],
    attn_mask: Optional[Tensor],
    num_heads: int) -> bool:
  _127 = "For batched (3-D) `query`, expected `key` and `value` to be 3-D but found {}-D and {}-D tensors respectively"
  _128 = "For batched (3-D) `query`, expected `key_padding_mask` to be `None` or 2-D but found {}-D tensor instead"
  _129 = "For batched (3-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D but found {}-D tensor instead"
  _130 = "For unbatched (2-D) `query`, expected `key` and `value` to be 2-D but found {}-D and {}-D tensors respectively"
  _131 = "For unbatched (2-D) `query`, expected `key_padding_mask` to be `None` or 1-D but found {}-D tensor instead"
  _132 = "For unbatched (2-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D but found {}-D tensor instead"
  _133 = "Expected `attn_mask` shape to be {} but got {}"
  _134 = "query should be unbatched 2D or batched 3D tensor but received {}-D query tensor"
  _135 = uninitialized(bool)
  if torch.eq(torch.dim(query), 3):
    if torch.eq(torch.dim(key), 3):
      _136 = torch.eq(torch.dim(value), 3)
    else:
      _136 = False
    if _136:
      pass
    else:
      _137 = torch.format(_127, torch.dim(key), torch.dim(value))
      _138 = torch.add("AssertionError: ", _137)
      ops.prim.RaiseException(_138)
    _139 = torch.__isnot__(key_padding_mask, None)
    if _139:
      key_padding_mask18 = unchecked_cast(Tensor, key_padding_mask)
      _140 = torch.eq(torch.dim(key_padding_mask18), 2)
      if _140:
        pass
      else:
        _141 = torch.format(_128, torch.dim(key_padding_mask18))
        _142 = torch.add("AssertionError: ", _141)
        ops.prim.RaiseException(_142)
    else:
      pass
    if torch.__isnot__(attn_mask, None):
      attn_mask20 = unchecked_cast(Tensor, attn_mask)
      _143 = torch.dim(attn_mask20)
      if torch.__contains__([2, 3], _143):
        pass
      else:
        _144 = torch.format(_129, torch.dim(attn_mask20))
        _145 = torch.add("AssertionError: ", _144)
        ops.prim.RaiseException(_145)
    else:
      pass
    is_batched = True
  else:
    if torch.eq(torch.dim(query), 2):
      if torch.eq(torch.dim(key), 2):
        _146 = torch.eq(torch.dim(value), 2)
      else:
        _146 = False
      if _146:
        pass
      else:
        _147 = torch.format(_130, torch.dim(key), torch.dim(value))
        _148 = torch.add("AssertionError: ", _147)
        ops.prim.RaiseException(_148)
      _149 = torch.__isnot__(key_padding_mask, None)
      if _149:
        key_padding_mask19 = unchecked_cast(Tensor, key_padding_mask)
        _150 = torch.eq(torch.dim(key_padding_mask19), 1)
        if _150:
          pass
        else:
          _151 = torch.format(_131, torch.dim(key_padding_mask19))
          _152 = torch.add("AssertionError: ", _151)
          ops.prim.RaiseException(_152)
      else:
        pass
      if torch.__isnot__(attn_mask, None):
        attn_mask21 = unchecked_cast(Tensor, attn_mask)
        _153 = torch.dim(attn_mask21)
        _154 = torch.__contains__([2, 3], _153)
        if _154:
          pass
        else:
          _155 = torch.format(_132, torch.dim(attn_mask21))
          _156 = torch.add("AssertionError: ", _155)
          ops.prim.RaiseException(_156)
        _157 = torch.eq(torch.dim(attn_mask21), 3)
        if _157:
          _158 = (torch.size(query))[0]
          _159 = (torch.size(key))[0]
          expected_shape = (num_heads, _158, _159)
          _160 = torch.eq(torch.size(attn_mask21), [num_heads, _158, _159])
          if _160:
            pass
          else:
            _161 = torch.format(_133, expected_shape, torch.size(attn_mask21))
            _162 = torch.add("AssertionError: ", _161)
            ops.prim.RaiseException(_162)
        else:
          pass
      else:
        pass
      is_batched0 = False
    else:
      _163 = torch.format(_134, torch.dim(query))
      _164 = torch.add("AssertionError: ", _163)
      ops.prim.RaiseException(_164)
      is_batched0 = _135
    is_batched = is_batched0
  return is_batched
def _in_projection_packed(q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor]=None) -> List[Tensor]:
  E = torch.size(q, -1)
  if torch.__is__(k, v):
    if torch.__is__(q, k):
      _167 = torch.chunk(torch.linear(q, w, b), 3, -1)
      _166 = _167
    else:
      _168 = torch.split(w, [E, torch.mul(E, 2)])
      w_q, w_kv, = _168
      if torch.__is__(b, None):
        b_q, b_kv = None, None
      else:
        b0 = unchecked_cast(Tensor, b)
        _169 = torch.split(b0, [E, torch.mul(E, 2)])
        b_q1, b_kv0, = _169
        b_q, b_kv = b_q1, b_kv0
      _170 = torch.linear(q, w_q, b_q)
      _171 = torch.chunk(torch.linear(k, w_kv, b_kv), 2, -1)
      _166 = torch.add([_170], _171)
    _165 = _166
  else:
    w_q0, w_k, w_v, = torch.chunk(w, 3)
    if torch.__is__(b, None):
      b_q2, b_k, b_v = None, None, None
    else:
      b1 = unchecked_cast(Tensor, b)
      b_q3, b_k1, b_v1, = torch.chunk(b1, 3)
      b_q2, b_k, b_v = b_q3, b_k1, b_v1
    _172 = [torch.linear(q, w_q0, b_q2), torch.linear(k, w_k, b_k), torch.linear(v, w_v, b_v)]
    _165 = _172
  return _165
def _in_projection(q: Tensor,
    k: Tensor,
    v: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    b_q: Optional[Tensor]=None,
    b_k: Optional[Tensor]=None,
    b_v: Optional[Tensor]=None) -> Tuple[Tensor, Tensor, Tensor]:
  _173 = "expecting query weights shape of {}, but got {}"
  _174 = "expecting key weights shape of {}, but got {}"
  _175 = "expecting value weights shape of {}, but got {}"
  _176 = "expecting query bias shape of {}, but got {}"
  _177 = "expecting key bias shape of {}, but got {}"
  _178 = "expecting value bias shape of {}, but got {}"
  _179 = uninitialized(Optional[Tensor])
  _180 = uninitialized(Optional[Tensor])
  _181 = uninitialized(Optional[Tensor])
  Eq = torch.size(q, -1)
  Ek = torch.size(k, -1)
  Ev = torch.size(v, -1)
  if torch.eq(torch.size(w_q), [Eq, Eq]):
    pass
  else:
    _182 = torch.format(_173, (Eq, Eq), torch.size(w_q))
    ops.prim.RaiseException(torch.add("AssertionError: ", _182))
  if torch.eq(torch.size(w_k), [Eq, Ek]):
    pass
  else:
    _183 = torch.format(_174, (Eq, Ek), torch.size(w_k))
    ops.prim.RaiseException(torch.add("AssertionError: ", _183))
  if torch.eq(torch.size(w_v), [Eq, Ev]):
    pass
  else:
    _184 = torch.format(_175, (Eq, Ev), torch.size(w_v))
    ops.prim.RaiseException(torch.add("AssertionError: ", _184))
  if torch.__is__(b_q, None):
    _185, b_q4 = True, b_q
  else:
    b_q5 = unchecked_cast(Tensor, b_q)
    _185, b_q4 = torch.eq(torch.size(b_q5), [Eq]), b_q5
  if _185:
    b_q6 : Optional[Tensor] = b_q4
  else:
    b_q7 = unchecked_cast(Tensor, b_q4)
    _186 = torch.format(_176, (Eq,), torch.size(b_q7))
    ops.prim.RaiseException(torch.add("AssertionError: ", _186))
    b_q6 = _181
  if torch.__is__(b_k, None):
    _187, b_k2 = True, b_k
  else:
    b_k3 = unchecked_cast(Tensor, b_k)
    _187, b_k2 = torch.eq(torch.size(b_k3), [Eq]), b_k3
  if _187:
    b_k4 : Optional[Tensor] = b_k2
  else:
    b_k5 = unchecked_cast(Tensor, b_k2)
    _188 = torch.format(_177, (Eq,), torch.size(b_k5))
    ops.prim.RaiseException(torch.add("AssertionError: ", _188))
    b_k4 = _180
  if torch.__is__(b_v, None):
    _189, b_v2 = True, b_v
  else:
    b_v3 = unchecked_cast(Tensor, b_v)
    _189, b_v2 = torch.eq(torch.size(b_v3), [Eq]), b_v3
  if _189:
    b_v4 : Optional[Tensor] = b_v2
  else:
    b_v5 = unchecked_cast(Tensor, b_v2)
    _190 = torch.format(_178, (Eq,), torch.size(b_v5))
    ops.prim.RaiseException(torch.add("AssertionError: ", _190))
    b_v4 = _179
  _191 = (torch.linear(q, w_q, b_q6), torch.linear(k, w_k, b_k4), torch.linear(v, w_v, b_v4))
  return _191
def _scaled_dot_product_attention(q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor]=None,
    dropout_p: float=0.) -> Tuple[Tensor, Tensor]:
  B, Nt, E, = torch.size(q)
  q3 = torch.div(q, torch.sqrt(E))
  if torch.__isnot__(attn_mask, None):
    attn_mask22 = unchecked_cast(Tensor, attn_mask)
    attn0 = torch.baddbmm(attn_mask22, q3, torch.transpose(k, -2, -1))
    attn = attn0
  else:
    attn1 = torch.bmm(q3, torch.transpose(k, -2, -1))
    attn = attn1
  attn2 = __torch__.torch.nn.functional.softmax(attn, -1, 3, None, )
  if torch.gt(dropout_p, 0.):
    attn4 = __torch__.torch.nn.functional.dropout(attn2, dropout_p, True, False, )
    attn3 = attn4
  else:
    attn3 = attn2
  output = torch.bmm(attn3, v)
  return (output, attn3)
def softmax(input: Tensor,
    dim: Optional[int]=None,
    _stacklevel: int=3,
    dtype: Optional[int]=None) -> Tensor:
  _192 = __torch__.torch.nn.functional._get_softmax_dim
  if torch.__is__(dim, None):
    dim1 = _192("softmax", torch.dim(input), _stacklevel, )
    dim0 = dim1
  else:
    dim0 = unchecked_cast(int, dim)
  if torch.__is__(dtype, None):
    ret = torch.softmax(input, dim0)
  else:
    dtype0 = unchecked_cast(int, dtype)
    ret = torch.softmax(input, dim0, dtype0)
  return ret
def _get_softmax_dim(name: str,
    ndim: int,
    stacklevel: int) -> int:
  _193 = "Implicit dimension choice for {} has been deprecated. Change the call to include dim=X as an argument."
  torch.warn(torch.format(_193, name), stacklevel)
  if torch.eq(ndim, 0):
    _194 = True
  else:
    _194 = torch.eq(ndim, 1)
  if _194:
    _195 = True
  else:
    _195 = torch.eq(ndim, 3)
  if _195:
    ret = 0
  else:
    ret = 1
  return ret
