class ModuleList(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : NoneType
  __annotations__["0"] = __torch__.tacotron2.fastpitch_layers.FFTBlock
  def __len__(self: __torch__.torch.nn.modules.container.___torch_mangle_3.ModuleList) -> int:
    return 1
