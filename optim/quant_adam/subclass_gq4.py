"""
   Simulating cdf quantization
   by: Albert Catalan
"""

import math

import torch
from torch import Tensor
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.utils import TORCH_VERSION_AT_LEAST_2_5, TorchAOBaseTensor
from .qmaps import normal_quantization

aten = torch.ops.aten
c10d_functional = torch.ops.c10d_functional
_c10d_functional = torch.ops._c10d_functional

# DTYPE = torch.float8_e4m3fn
DTYPE = torch.float32 # We hack it :)
QBITS = 4

def quantize_fp8(input: Tensor, block_size: int):
    shape = input.shape
    input = input.view(-1, block_size)
    scale = input.abs().amax(-1).clip(1e-12) / torch.finfo(DTYPE).max
    codes = input.to(DTYPE).view(-1)
    return codes.view(shape), scale

# NOTE: FP8 sign bit is redundant for unsigned optim state.
# we may investigate how to use it to increase range/precision for unsigned optim state.
# https://arxiv.org/abs/2409.12517 uses FP8 E5M2 for 2nd Adam buffer
class OptimStateGQ4(TorchAOBaseTensor):
    tensor_attrs = ["codes", "scale"]

    @staticmethod
    def __new__(cls, codes: Tensor, scale: Tensor):
        return Tensor._make_wrapper_subclass(cls, codes.shape, device=codes.device)

    def __init__(self, codes: Tensor, scale: Tensor):
        """Create quantized FP8 optimizer state.

        Args
            codes: quantized FP8 E4M3FN data. Has the same shape as the original float tensor.
            scale: scale data for block-wise quantization.

        NOTE: To get block-wise scale, the original float tensor is first reshape to (-1, block_size).
        Thus, the last dimension of the original float tensor is not necessarily divisible by block size.
        Given `codes` and `scale`, `block_size` is calculated as `codes.numel() // scale.numel()`.
        """
        assert codes.dtype is DTYPE
        assert scale.ndim == 1
        self.codes = codes
        self.scale = scale
        self.block_size = codes.numel() // scale.numel()

    def __tensor_flatten__(self):
        return self.tensor_attrs, []

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None
    ):
        return cls(
            *[tensor_data_dict[name] for name in cls.tensor_attrs], *tensor_attributes
        )

    def dequantize(self, output_dtype=None):
        self._fake_quantize()
        float_data = self.codes.float()
        if output_dtype is not None:
            float_data = float_data.to(output_dtype)
        return float_data

    def _fake_quantize(self):
        _, _, tmp = normal_quantization(self.codes, QBITS, self.block_size)
        self.codes = tmp.view(self.codes.size())

    @classmethod
    def zeros(cls, shape, block_size: int = 256, device=None):
        codes = torch.zeros(shape, dtype=DTYPE, device=device)
        scale = torch.zeros(codes.numel() // block_size, device=device)
        return cls(codes, scale)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(block_size={self.block_size}, "
            f"shape={tuple(self.shape)}, device={self.device}, requires_grad={self.requires_grad})"
        )


@OptimStateGQ4.implements(aten.copy_.default)
def _(func, types, args, kwargs):
    dst = args[0]
    src = args[1]

    if isinstance(dst, OptimStateGQ4) and isinstance(src, OptimStateGQ4):
        assert dst.block_size == src.block_size
        dst.codes.copy_(src.codes)
        dst.scale.copy_(src.scale)

    elif isinstance(dst, OptimStateGQ4):
        codes, scale = quantize_fp8(src, dst.block_size)
        dst.codes.copy_(codes)
        dst.scale.copy_(scale)

    else:
        dst.copy_(src.dequantize())

    return dst


@OptimStateGQ4.implements(aten._to_copy.default)
def _(func, types, args, kwargs):
    # ignore dtype
    device = kwargs.get("device", None)
    out = OptimStateGQ4(
        args[0].codes.to(device=device),
        args[0].scale.to(device=device),
    )
    return return_and_correct_aliasing(func, args, kwargs, out)

@OptimStateGQ4.implements(aten.lerp.Scalar)
def _(func, types, args, kwargs):
    args = [x.dequantize() if isinstance(x, OptimStateGQ4) else x for x in args]
    return func(*args, **kwargs)

# Needed for the quant error compute
@OptimStateGQ4.implements(aten.sub.Tensor)
def _(func, types, args, kwargs):
    args = [x.dequantize() if isinstance(x, OptimStateGQ4) else x for x in args]
    return func(*args, **kwargs)

# this is needed for DTensor.from_local()
@OptimStateGQ4.implements(aten.view.default)
def _(func, types, args, kwargs):
    x, shape = args
    return OptimStateGQ4(x.codes.view(shape), x.scale)

@OptimStateGQ4.implements(
    [
        # required by DTensor.full_tensor()
        c10d_functional.all_gather_into_tensor.default,
        _c10d_functional.all_gather_into_tensor.default,
        c10d_functional.wait_tensor.default,
        _c10d_functional.wait_tensor.default,
        # required by torch.distributed.checkpoint.save
        aten.detach.default,
    ]
)
def _(func, types, args, kwargs):
    x = args[0]
    if not isinstance(x, OptimStateGQ4):
        raise ValueError(f"expecting a OptimStateGQ4 but found {type(x)}")

    # assume tensors from all ranks have the same signedness
    return OptimStateGQ4(
        func(x.codes, *args[1:], **kwargs),
        func(x.scale, *args[1:], **kwargs),
    )

# required by torch.distributed.checkpoint.save
# note that we don't actually implement pin memory for this tensor subclass
# (pin_memory argument is ignored in aten._to_copy)
@OptimStateGQ4.implements(aten.is_pinned.default)
def _(func, types, args, kwargs):
    return args[0].codes.is_pinned() and args[0].scale.is_pinned()

# required by torch.distributed.checkpoint.load when world size changes i.e. re-sharding
@OptimStateGQ4.implements(aten.slice.Tensor)
def _(func, types, args, kwargs):
    x, dim, start, end = args[:4]
    step = args[4] if len(args) > 4 else 1

    # input validation
    if dim != 0:
        raise ValueError("Only support aten.slice along the first dim")
    if step != 1:
        raise ValueError("Only support aten.slice with step=1")

    block_size = x.block_size
    stride = math.prod(x.shape[1:])

    # for 1 increment in x along the first dim,
    # (flattened) scale will increment by stride / block_size
    if (start * stride) % block_size != 0 or (end * stride) % block_size != 0:
        raise ValueError(
            f"Invalid start or end for shape={x.shape} and block_size={block_size}. "
            f"Make sure start and end align with block boundary. "
            f"Received start={start}, end={end}."
        )

    return OptimStateGQ4(
        x.codes[start:end],
        x.scale[start * stride // block_size : end * stride // block_size],
    )

if TORCH_VERSION_AT_LEAST_2_5:
    from torch.serialization import add_safe_globals

    add_safe_globals([OptimStateGQ4])
    
if __name__ == "__main__":
    from qmaps import cdf_quantization, cdf_quantization_legacy
    block_size = 256
    nbits = 4
    w = torch.randn(block_size*4)
    w = w.view(4, block_size)
    # Time both quantization functions
    import time
    
    # Time normal_quantization_fast
    start = time.time()
    for _ in range(1000):
        _ = normal_quantization(w, sym=False, n_bits=nbits)
    fast_time = (time.time() - start) / 1000
    legacy_time=0.999
    
    # Time normal_quantization
    start = time.time()
    for _ in range(1000):
        _ = cdf_quantization_legacy(w, n_bits=nbits, block_size=block_size)
    cdf_time = (time.time() - start) / 1000
    
    # Time normal_quantization
    start = time.time()
    for _ in range(1000):
        _ = cdf_quantization(w, n_bits=nbits, block_size=block_size)
    cdf_fast_time = (time.time() - start) / 1000
    
    # Time normal_quantization
    start = time.time()
    for _ in range(1000):
        _ = quantize_fp8(w, block_size=block_size)
    fp8_time = (time.time() - start) / 1000
    
    print(f"Gaussian version average time: {legacy_time*1000:.3f} ms")
    print(f"Gaussian Fast version average time: {fast_time*1000:.3f} ms")
    print(f"CDF version average time: {cdf_time*1000:.3f} ms")
    print(f"CDF Fast average time: {cdf_fast_time*1000:.3f} ms")    
    print(f"fp8_time Fast version average time: {fp8_time*1000:.3f} ms")
    print(f"Speedup: {legacy_time/fast_time:.2f}x {cdf_time/cdf_fast_time:.2f}x ")