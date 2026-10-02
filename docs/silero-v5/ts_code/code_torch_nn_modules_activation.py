class MultiheadAttention(Module):
  __parameters__ = ["in_proj_weight", "q_proj_weight", "k_proj_weight", "v_proj_weight", "in_proj_bias", ]
  __buffers__ = []
  in_proj_weight : Tensor
  q_proj_weight : NoneType
  k_proj_weight : NoneType
  v_proj_weight : NoneType
  in_proj_bias : Tensor
  training : bool
  _is_full_backward_hook : NoneType
  embed_dim : int
  kdim : int
  vdim : int
  _qkv_same_embed_dim : bool
  num_heads : int
  dropout : float
  head_dim : int
  bias_k : Optional[Tensor]
  bias_v : Optional[Tensor]
  add_zero_attn : bool
  out_proj : __torch__.torch.nn.modules.linear.NonDynamicallyQuantizableLinear
  batch_first : Final[bool] = True
  def forward(self: __torch__.torch.nn.modules.activation.MultiheadAttention,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    key_padding_mask: Optional[Tensor]=None,
    need_weights: bool=True,
    attn_mask: Optional[Tensor]=None,
    average_attn_weights: bool=True) -> Tuple[Tensor, Optional[Tensor]]:
    _0 = "input not batched; expected query.dim() of 3 but got {}"
    why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
    _1 = "dtypes of query ({}) and self.in_proj_bias ({}) don\'t match"
    _2 = "dtypes of query ({}) and self.in_proj_weight ({}) don\'t match"
    why_not_fast_path0 = "key_padding_mask is not supported with NestedTensor input"
    why_not_fast_path1 = "some Tensor argument is neither CUDA nor CPU"
    why_not_fast_path2 = "grad is enabled and at least one of query or the input/output projection weights or biases requires_grad"
    _3 = "The fast path was not hit because {}"
    _4 = "MultiheadAttention does not support NestedTensor outside of its fast path. "
    _5 = __torch__.torch.nn.functional.multi_head_attention_forward
    _6 = uninitialized(Tuple[Tensor, Tensor])
    is_batched = torch.eq(torch.dim(query), 3)
    if torch.__not__(is_batched):
      why_not_fast_path4 = torch.format(_0, torch.dim(query))
      why_not_fast_path3, key_padding_mask0, attn_mask0 = why_not_fast_path4, key_padding_mask, attn_mask
    else:
      if torch.__isnot__(query, key):
        _7 = True
      else:
        _7 = torch.__isnot__(key, value)
      if _7:
        why_not_fast_path5, key_padding_mask1, attn_mask1 = why_not_fast_path, key_padding_mask, attn_mask
      else:
        _8 = ops.prim.dtype(query)
        in_proj_bias = self.in_proj_bias
        _9 = torch.ne(_8, ops.prim.dtype(in_proj_bias))
        if _9:
          _10 = ops.prim.dtype(query)
          in_proj_bias0 = self.in_proj_bias
          why_not_fast_path7 = torch.format(_1, _10, ops.prim.dtype(in_proj_bias0))
          why_not_fast_path6, key_padding_mask2, attn_mask2 = why_not_fast_path7, key_padding_mask, attn_mask
        else:
          _11 = ops.prim.dtype(query)
          in_proj_weight = self.in_proj_weight
          _12 = ops.prim.dtype(in_proj_weight)
          if torch.ne(_11, _12):
            _13 = ops.prim.dtype(query)
            in_proj_weight0 = self.in_proj_weight
            _14 = ops.prim.dtype(in_proj_weight0)
            why_not_fast_path8, key_padding_mask3, attn_mask3 = torch.format(_2, _13, _14), key_padding_mask, attn_mask
          else:
            training = self.training
            if training:
              why_not_fast_path9, key_padding_mask4, attn_mask4 = "training is enabled", key_padding_mask, attn_mask
            else:
              bias_k = self.bias_k
              _15 = torch.__isnot__(bias_k, None)
              if _15:
                why_not_fast_path10, key_padding_mask5, attn_mask5 = "self.bias_k was not None", key_padding_mask, attn_mask
              else:
                bias_v = self.bias_v
                _16 = torch.__isnot__(bias_v, None)
                if _16:
                  why_not_fast_path11, key_padding_mask6, attn_mask6 = "self.bias_v was not None", key_padding_mask, attn_mask
                else:
                  dropout = self.dropout
                  if bool(dropout):
                    dropout0 = self.dropout
                    why_not_fast_path13 = torch.format("dropout was {}, required zero", dropout0)
                    why_not_fast_path12, key_padding_mask7, attn_mask7 = why_not_fast_path13, key_padding_mask, attn_mask
                  else:
                    add_zero_attn = self.add_zero_attn
                    if add_zero_attn:
                      why_not_fast_path14, key_padding_mask8, attn_mask8 = "add_zero_attn was enabled", key_padding_mask, attn_mask
                    else:
                      _qkv_same_embed_dim = self._qkv_same_embed_dim
                      _17 = torch.__not__(_qkv_same_embed_dim)
                      if _17:
                        why_not_fast_path15, key_padding_mask9, attn_mask9 = "_qkv_same_embed_dim was not True", key_padding_mask, attn_mask
                      else:
                        _18 = torch.__isnot__(attn_mask, None)
                        if _18:
                          attn_mask11 = unchecked_cast(Tensor, attn_mask)
                          why_not_fast_path16, key_padding_mask10, attn_mask10 = "attn_mask was not None", key_padding_mask, attn_mask11
                        else:
                          _19 = ops.prim.is_nested(query)
                          if _19:
                            _21 = torch.__isnot__(key_padding_mask, None)
                            _20 = _21
                          else:
                            _20 = False
                          if _20:
                            key_padding_mask12 = unchecked_cast(Tensor, key_padding_mask)
                            why_not_fast_path17, key_padding_mask11 = why_not_fast_path0, key_padding_mask12
                          else:
                            why_not_fast_path17, key_padding_mask11 = "", key_padding_mask
                          why_not_fast_path16, key_padding_mask10, attn_mask10 = why_not_fast_path17, key_padding_mask11, attn_mask
                        why_not_fast_path15, key_padding_mask9, attn_mask9 = why_not_fast_path16, key_padding_mask10, attn_mask10
                      why_not_fast_path14, key_padding_mask8, attn_mask8 = why_not_fast_path15, key_padding_mask9, attn_mask9
                    why_not_fast_path12, key_padding_mask7, attn_mask7 = why_not_fast_path14, key_padding_mask8, attn_mask8
                  why_not_fast_path11, key_padding_mask6, attn_mask6 = why_not_fast_path12, key_padding_mask7, attn_mask7
                why_not_fast_path10, key_padding_mask5, attn_mask5 = why_not_fast_path11, key_padding_mask6, attn_mask6
              why_not_fast_path9, key_padding_mask4, attn_mask4 = why_not_fast_path10, key_padding_mask5, attn_mask5
            why_not_fast_path8, key_padding_mask3, attn_mask3 = why_not_fast_path9, key_padding_mask4, attn_mask4
          why_not_fast_path6, key_padding_mask2, attn_mask2 = why_not_fast_path8, key_padding_mask3, attn_mask3
        why_not_fast_path5, key_padding_mask1, attn_mask1 = why_not_fast_path6, key_padding_mask2, attn_mask2
      why_not_fast_path3, key_padding_mask0, attn_mask0 = why_not_fast_path5, key_padding_mask1, attn_mask1
    _22 = torch.gt(torch.len(why_not_fast_path3), 0)
    if torch.__not__(_22):
      in_proj_weight1 = self.in_proj_weight
      in_proj_bias1 = self.in_proj_bias
      out_proj = self.out_proj
      weight = out_proj.weight
      out_proj0 = self.out_proj
      bias = out_proj0.bias
      _25 = annotate(List[bool], [])
      if ops.prim.is_cuda(query):
        _26 = True
      else:
        _27 = str(ops.prim.device(query))
        _28 = torch.find(_27, "cpu", 0, torch.len(_27))
        _26 = torch.ne(_28, -1)
      _29 = torch.append(_25, _26)
      if ops.prim.is_cuda(key):
        _30 = True
      else:
        _31 = str(ops.prim.device(key))
        _32 = torch.find(_31, "cpu", 0, torch.len(_31))
        _30 = torch.ne(_32, -1)
      _33 = torch.append(_25, _30)
      if ops.prim.is_cuda(value):
        _34 = True
      else:
        _35 = str(ops.prim.device(value))
        _36 = torch.find(_35, "cpu", 0, torch.len(_35))
        _34 = torch.ne(_36, -1)
      _37 = torch.append(_25, _34)
      if ops.prim.is_cuda(in_proj_weight1):
        _38 = True
      else:
        _39 = ops.prim.device(in_proj_weight1)
        _40 = str(_39)
        _41 = torch.find(_40, "cpu", 0, torch.len(_40))
        _38 = torch.ne(_41, -1)
      _42 = torch.append(_25, _38)
      if ops.prim.is_cuda(in_proj_bias1):
        _43 = True
      else:
        _44 = str(ops.prim.device(in_proj_bias1))
        _45 = torch.find(_44, "cpu", 0, torch.len(_44))
        _43 = torch.ne(_45, -1)
      _46 = torch.append(_25, _43)
      if ops.prim.is_cuda(weight):
        _47 = True
      else:
        _48 = str(ops.prim.device(weight))
        _49 = torch.find(_48, "cpu", 0, torch.len(_48))
        _47 = torch.ne(_49, -1)
      _50 = torch.append(_25, _47)
      if ops.prim.is_cuda(bias):
        _51 = True
      else:
        _52 = str(ops.prim.device(bias))
        _53 = torch.find(_52, "cpu", 0, torch.len(_52))
        _51 = torch.ne(_53, -1)
      _54 = torch.append(_25, _51)
      if torch.__not__(torch.all(_25)):
        why_not_fast_path19 = why_not_fast_path1
      else:
        if torch.is_grad_enabled():
          _56 = annotate(List[bool], [])
          _57 = torch.append(_56, ops.prim.requires_grad(query))
          _58 = torch.append(_56, ops.prim.requires_grad(key))
          _59 = torch.append(_56, ops.prim.requires_grad(value))
          _60 = ops.prim.requires_grad(in_proj_weight1)
          _61 = torch.append(_56, _60)
          _62 = ops.prim.requires_grad(in_proj_bias1)
          _63 = torch.append(_56, _62)
          _64 = ops.prim.requires_grad(weight)
          _65 = torch.append(_56, _64)
          _66 = torch.append(_56, ops.prim.requires_grad(bias))
          _55 = torch.any(_56)
        else:
          _55 = False
        if _55:
          why_not_fast_path20 = why_not_fast_path2
        else:
          why_not_fast_path20 = why_not_fast_path3
        why_not_fast_path19 = why_not_fast_path20
      _67 = torch.gt(torch.len(why_not_fast_path19), 0)
      if torch.__not__(_67):
        embed_dim = self.embed_dim
        num_heads = self.num_heads
        in_proj_weight2 = self.in_proj_weight
        in_proj_bias2 = self.in_proj_bias
        out_proj1 = self.out_proj
        weight0 = out_proj1.weight
        out_proj2 = self.out_proj
        bias0 = out_proj2.bias
        _70 = torch.__isnot__(key_padding_mask0, None)
        if _70:
          key_padding_mask16 = unchecked_cast(Tensor, key_padding_mask0)
          _71, key_padding_mask15 = key_padding_mask16, key_padding_mask16
        else:
          _71, key_padding_mask15 = attn_mask0, key_padding_mask0
        _72, _73 = torch._native_multi_head_attention(query, key, value, embed_dim, num_heads, in_proj_weight2, in_proj_bias2, weight0, bias0, _71, need_weights, average_attn_weights)
        _68, _69, key_padding_mask14 = True, (_72, _73), key_padding_mask15
      else:
        _68, _69, key_padding_mask14 = False, _6, key_padding_mask0
      _23, _24, why_not_fast_path18, key_padding_mask13 = _68, _69, why_not_fast_path19, key_padding_mask14
    else:
      _23, _24, why_not_fast_path18, key_padding_mask13 = False, _6, why_not_fast_path3, key_padding_mask0
    if _23:
      _74 = _24
    else:
      if ops.prim.is_nested(query):
        _75 = True
      else:
        _75 = ops.prim.is_nested(key)
      if _75:
        any_nested = True
      else:
        any_nested = ops.prim.is_nested(value)
      if torch.__not__(any_nested):
        pass
      else:
        _76 = torch.format(_3, why_not_fast_path18)
        _77 = torch.add("AssertionError: ", torch.add(_4, _76))
        ops.prim.RaiseException(_77)
      if is_batched:
        if torch.__is__(key, value):
          if torch.__is__(query, key):
            _78 = torch.transpose(query, 1, 0)
            query2, key2, value2 = _78, _78, _78
          else:
            _79 = annotate(List[Tensor], [])
            _80 = torch.transpose(query, 1, 0)
            _81 = torch.append(_79, _80)
            _82 = torch.append(_79, torch.transpose(key, 1, 0))
            query3, key3, = _79
            query2, key2, value2 = query3, key3, key3
          query1, key1, value1 = query2, key2, value2
        else:
          _83 = annotate(List[Tensor], [])
          _84 = torch.append(_83, torch.transpose(query, 1, 0))
          _85 = torch.append(_83, torch.transpose(key, 1, 0))
          _86 = torch.append(_83, torch.transpose(value, 1, 0))
          query4, key4, value3, = _83
          query1, key1, value1 = query4, key4, value3
        query0, key0, value0 = query1, key1, value1
      else:
        query0, key0, value0 = query, key, value
      _qkv_same_embed_dim0 = self._qkv_same_embed_dim
      _87 = torch.__not__(_qkv_same_embed_dim0)
      if _87:
        embed_dim0 = self.embed_dim
        num_heads0 = self.num_heads
        in_proj_weight3 = self.in_proj_weight
        in_proj_bias3 = self.in_proj_bias
        bias_k0 = self.bias_k
        bias_v0 = self.bias_v
        add_zero_attn0 = self.add_zero_attn
        dropout1 = self.dropout
        out_proj3 = self.out_proj
        weight1 = out_proj3.weight
        out_proj4 = self.out_proj
        bias1 = out_proj4.bias
        training0 = self.training
        q_proj_weight = self.q_proj_weight
        k_proj_weight = self.k_proj_weight
        v_proj_weight = self.v_proj_weight
        _88 = _5(query0, key0, value0, embed_dim0, num_heads0, in_proj_weight3, in_proj_bias3, bias_k0, bias_v0, add_zero_attn0, dropout1, weight1, bias1, training0, key_padding_mask13, need_weights, attn_mask0, True, q_proj_weight, k_proj_weight, v_proj_weight, None, None, average_attn_weights, )
        attn_output0, attn_output_weights0, = _88
        attn_output, attn_output_weights = attn_output0, attn_output_weights0
      else:
        embed_dim1 = self.embed_dim
        num_heads1 = self.num_heads
        in_proj_weight4 = self.in_proj_weight
        in_proj_bias4 = self.in_proj_bias
        bias_k1 = self.bias_k
        bias_v1 = self.bias_v
        add_zero_attn1 = self.add_zero_attn
        dropout2 = self.dropout
        out_proj5 = self.out_proj
        weight2 = out_proj5.weight
        out_proj6 = self.out_proj
        bias2 = out_proj6.bias
        training1 = self.training
        _89 = _5(query0, key0, value0, embed_dim1, num_heads1, in_proj_weight4, in_proj_bias4, bias_k1, bias_v1, add_zero_attn1, dropout2, weight2, bias2, training1, key_padding_mask13, need_weights, attn_mask0, False, None, None, None, None, None, average_attn_weights, )
        attn_output1, attn_output_weights1, = _89
        attn_output, attn_output_weights = attn_output1, attn_output_weights1
      if is_batched:
        _91 = torch.transpose(attn_output, 1, 0)
        _90 = (_91, attn_output_weights)
      else:
        _92 = (attn_output, attn_output_weights)
        _90 = _92
      _74 = _90
    return _74
class ReLU(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : NoneType
  inplace : Final[bool] = False
  def forward(self: __torch__.torch.nn.modules.activation.ReLU,
    input: Tensor) -> Tensor:
    _93 = __torch__.torch.nn.functional.relu(input, False, )
    return _93
class GELU(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : NoneType
  approximate : Final[str] = "none"
  def forward(self: __torch__.torch.nn.modules.activation.GELU,
    input: Tensor) -> Tensor:
    return torch.gelu(input)
