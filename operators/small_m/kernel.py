import torch
import triton
import triton.language as tl


@triton.jit
def _k12_gemm_kernel(
    A_ptr, B_ptr, Y_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    y = acc.to(tl.float16)
    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, y, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class ModelNew(torch.nn.Module):
    def __init__(self, **static_params):
        super().__init__()

    def forward(self, *inputs) -> tuple[torch.Tensor, ...]:
        A, Bmat = inputs
        M, K = A.shape
        _, N = Bmat.shape
        y = torch.empty((M, N), device=A.device, dtype=torch.float16)
        self._launch(A, Bmat, y, M, N, K)
        return (y,)

    def _launch(self, A, Bmat, y, M, N, K):
        # Single launch: 2D grid (M-blocks, N-blocks), BLOCK_M=16 >= 16 so
        # tl.dot's min-dim constraint is satisfied even though only M<=8 rows
        # are live. Both inputs stream contiguously (A along K, B along N).
        assert Bmat.stride(1) == 1
        assert A.stride(1) == 1

        def grid(meta):
            return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
        _k12_gemm_kernel[grid](
            A, Bmat, y,
            M, N, K,
            A.stride(0), A.stride(1),
            Bmat.stride(0), Bmat.stride(1),
            y.stride(0), y.stride(1),
            BLOCK_M=16, BLOCK_N=128, BLOCK_K=64,
        )
