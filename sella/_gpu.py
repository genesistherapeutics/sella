"""Centralized GPU helpers for Sella.

Provides numpy-in / numpy-out wrappers for eigh and QR that route through
torch+CUDA when a usable GPU is present, with a shared `_has_torch` state
and independently measured size thresholds that gate upload overhead.

GPU usage is opt-out via SELLA_DISABLE_GPU=1. Eigh and QR default to 500
and 700 rows, respectively; SELLA_GPU_MIN_DIM overrides both, and their
SELLA_GPU_EIGH_MIN_DIM / SELLA_GPU_QR_MIN_DIM overrides take precedence.
On a torch CUDA OOM the call
falls back to CPU and records the failure so subsequent calls skip the GPU
attempt for that size; this keeps smaller GPUs from thrashing.
"""

import os
from contextlib import nullcontext
from threading import BoundedSemaphore
import numpy as np
from scipy.linalg import eigh as _cpu_eigh
from . import _numba
from scipy.linalg import qr as _cpu_qr

try:
    import torch
    _has_torch = torch.cuda.is_available()
except Exception:
    torch = None
    _has_torch = False

if os.environ.get("SELLA_DISABLE_GPU", "").lower() in ("1", "true", "yes"):
    _has_torch = False

try:
    _GPU_MIN_DIM = int(os.environ.get("SELLA_GPU_MIN_DIM", "200"))
except ValueError:
    _GPU_MIN_DIM = 200

try:
    _GPU_MATMUL_MIN_DIM = int(os.environ.get("SELLA_GPU_MATMUL_MIN_DIM", "1000"))
except ValueError:
    _GPU_MATMUL_MIN_DIM = 1000

# Separate crossovers measured against our nopython CPU LAPACK paths.
# SELLA_GPU_MIN_DIM remains a shared explicit override for compatibility.
def _threshold(name, default):
    try:
        return int(os.environ.get(name, os.environ.get("SELLA_GPU_MIN_DIM", str(default))))
    except ValueError:
        return default


_GPU_EIGH_MIN_DIM = _threshold("SELLA_GPU_EIGH_MIN_DIM", 500)
_GPU_QR_MIN_DIM = _threshold("SELLA_GPU_QR_MIN_DIM", 700)

# After a CUDA OOM at dimension N, refuse subsequent GPU offload for shapes
# >= N. Keeps a single failure from cascading and lets the CPU path take over
# cleanly on small GPUs.
_oom_floor = None

# Process-wide budget shared by independent optimizer workers. Configure before
# launching workers; generic Sella users retain double precision and no cap.
_hessian_eigh_dtype = np.dtype('float64')
_hessian_eigh_slots = None
_hessian_eigh_min_dim = 200
_UNSET = object()


def configure_linalg(use_gpu=None, hessian_eigh_dtype=None,
                     hessian_eigh_max_concurrent=_UNSET, hessian_eigh_min_dim=None):
    """Update specified solver settings before starting optimizers."""
    global _has_torch, _hessian_eigh_dtype, _hessian_eigh_slots, _hessian_eigh_min_dim
    dtype = (_hessian_eigh_dtype if hessian_eigh_dtype is None
             else np.dtype(hessian_eigh_dtype))
    min_dim = (_hessian_eigh_min_dim if hessian_eigh_min_dim is None
               else hessian_eigh_min_dim)
    if dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise ValueError('Hessian eigensolves require float32 or float64')
    if (hessian_eigh_max_concurrent is not _UNSET
            and hessian_eigh_max_concurrent is not None
            and hessian_eigh_max_concurrent < 1):
        raise ValueError('Hessian eigensolve concurrency must be positive')
    if min_dim < 0:
        raise ValueError('Hessian eigensolve minimum dimension must be nonnegative')
    gpu_enabled = _has_torch if use_gpu is None else bool(use_gpu)
    if dtype == np.dtype('float32') and gpu_enabled:
        raise ValueError('Float32 Hessian eigensolves require use_gpu=False')
    slots = _hessian_eigh_slots
    if hessian_eigh_max_concurrent is not _UNSET:
        slots = (BoundedSemaphore(hessian_eigh_max_concurrent)
                 if hessian_eigh_max_concurrent is not None else None)
    if use_gpu is not None:
        _has_torch = bool(use_gpu and torch is not None and torch.cuda.is_available())
    _hessian_eigh_dtype = dtype
    _hessian_eigh_min_dim = min_dim
    _hessian_eigh_slots = slots


def hessian_eigh_budget(dimension):
    """Hold one slot through a Hessian solve, synchronization and any fallback."""
    if _hessian_eigh_slots is None or dimension < _hessian_eigh_min_dim:
        return nullcontext()
    return _hessian_eigh_slots


def _hessian_eigh_unlimited(matrix, A_gpu=None):
    """Solve under a budget already held by the caller; do not acquire twice."""
    if _hessian_eigh_dtype == np.dtype('float32') and _numba.ENABLED and matrix.size:
        try:
            values, vectors = _numba.eigh(np.ascontiguousarray(matrix, dtype=np.float32))
            if not np.isfinite(values).all() or not np.isfinite(vectors).all():
                raise np.linalg.LinAlgError('Nonfinite float32 eigenpairs')
            return values.astype(np.float64), vectors.astype(np.float64)
        except np.linalg.LinAlgError:
            pass
    return gpu_eigh(matrix, A_gpu=A_gpu)


def hessian_eigh(matrix, A_gpu=None):
    """Solve a Hessian with float64 outputs, retrying failed FP32 solves in FP64."""
    with hessian_eigh_budget(matrix.shape[0]):
        return _hessian_eigh_unlimited(matrix, A_gpu=A_gpu)


def _gpu_ok(n, min_dim=None):
    return _has_torch and n >= (_GPU_MIN_DIM if min_dim is None else min_dim) and (
        _oom_floor is None or n < _oom_floor
    )


def _gpu_matmul_ok(n):
    return _has_torch and n >= _GPU_MATMUL_MIN_DIM and (
        _oom_floor is None or n < _oom_floor
    )


def _record_oom(n):
    global _oom_floor
    if _oom_floor is None or n < _oom_floor:
        _oom_floor = n
    if torch is not None:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def to_gpu(A):
    """Upload a numpy array to GPU as a contiguous float64 tensor.

    Returns the torch tensor, or None if no GPU is available / upload fails.
    Caller is responsible for size-gating with `_gpu_ok` if desired.
    """
    if not _has_torch:
        return None
    try:
        return torch.from_numpy(np.ascontiguousarray(A)).cuda()
    except (RuntimeError, MemoryError):
        _record_oom(A.shape[0] if A.ndim >= 1 else 0)
        return None


def gpu_eigh(A, A_gpu=None):
    """Eigendecomposition. GPU when beneficial, CPU otherwise.

    If A_gpu (a torch tensor on CUDA) is supplied, the upload is skipped.
    """
    n = A.shape[0]
    if n == 0:
        # scipy.linalg.eigh rejects 0x0 input (LAPACK dsyevr requires il<=n),
        # whereas np.linalg.eigh returns empties. Match numpy so callers with
        # a fully-constrained (empty) subspace don't crash.
        return (np.empty(0, dtype=np.float64),
                np.empty((0, 0), dtype=np.float64))
    if A_gpu is not None or _gpu_ok(n, _GPU_EIGH_MIN_DIM):
        try:
            At = A_gpu if A_gpu is not None else to_gpu(A)
            if At is not None:
                evals_t, evecs_t = torch.linalg.eigh(At)
                return evals_t.cpu().numpy(), evecs_t.cpu().numpy()
        except (RuntimeError, MemoryError):
            _record_oom(n)
    return _numba.eigh(np.ascontiguousarray(A)) if _numba.ENABLED else _cpu_eigh(A)


def gpu_eigh_t(A_gpu):
    """Eigendecomposition that returns torch tensors (no .cpu() download).

    Caller has already uploaded; returns (evals_t, evecs_t) on GPU, or
    (None, None) on OOM so caller can fall back.
    """
    try:
        return torch.linalg.eigh(A_gpu)
    except (RuntimeError, MemoryError):
        _record_oom(A_gpu.shape[0])
        return None, None


def gpu_qr(A):
    """Economy QR. GPU when beneficial, CPU otherwise."""
    n = A.shape[0]
    if _gpu_ok(n, _GPU_QR_MIN_DIM):
        try:
            At = to_gpu(A)
            if At is not None:
                Q_t, R_t = torch.linalg.qr(At, mode='reduced')
                return Q_t.cpu().numpy(), R_t.cpu().numpy()
        except (RuntimeError, MemoryError):
            _record_oom(n)
    if n == 0:
        return np.linalg.qr(A, mode='reduced')
    if _numba.ENABLED and A.shape[1] > 0:
        return _numba.qr(np.ascontiguousarray(A))
    return _cpu_qr(A, mode='economic', pivoting=False, check_finite=False)


def gpu_qr_with_pinv(A, rank_rtol=1e-6):
    """Economy QR plus full-rank pseudoinverse solve on GPU.

    Returns ``(Q, R, Binv_or_None)`` as numpy arrays. ``Binv_or_None`` is
    populated only when the reduced QR factor is square and passes the same
    relative diagonal rank check used by ``InternalPES``. Returning Q/R even
    when the solve is skipped lets callers reuse the GPU QR and run their
    rank-deficient fallback without refactorizing on CPU.
    """
    n = A.shape[0]
    if not _gpu_ok(n, _GPU_QR_MIN_DIM):
        return None

    try:
        At = to_gpu(A)
        if At is None:
            return None
        Q_t, R_t = torch.linalg.qr(At, mode='reduced')
    except (RuntimeError, MemoryError):
        _record_oom(n)
        return None

    Binv = None
    if R_t.numel() and R_t.shape[0] == R_t.shape[1]:
        try:
            rdiag = torch.abs(torch.diagonal(R_t))
            rmax = torch.max(rdiag)
            full_rank = (
                bool((rmax > 0).item())
                and bool((torch.min(rdiag) >= rank_rtol * rmax).item())
            )
            if full_rank:
                Binv = torch.linalg.solve_triangular(
                    R_t, Q_t.transpose(-2, -1), upper=True
                ).cpu().numpy()
        except (RuntimeError, MemoryError):
            _record_oom(n)
            Binv = None

    try:
        return Q_t.cpu().numpy(), R_t.cpu().numpy(), Binv
    except (RuntimeError, MemoryError):
        _record_oom(n)
        return None


def gpu_solve_triangular(A, B, upper=True):
    """Triangular solve on GPU when beneficial, else None.

    Solves ``A @ X = B`` and returns ``X`` as a numpy array. This is used for
    the full-rank internal-coordinate pseudoinverse after QR, where ``A`` is a
    square triangular factor and ``B`` has many right-hand sides. The upload
    and download overhead is still much smaller than the CPU solve for the
    large Jacobians seen in molecular-crystal runs.
    """
    n = A.shape[0]
    if _gpu_ok(n):
        try:
            At = to_gpu(A)
            Bt = to_gpu(B)
            if At is not None and Bt is not None:
                X_t = torch.linalg.solve_triangular(At, Bt, upper=upper)
                return X_t.cpu().numpy()
        except (RuntimeError, MemoryError):
            _record_oom(n)
    return None


def gpu_left_matmul(A, B, A_gpu=None):
    """Compute ``A @ B`` on GPU when beneficial, returning a numpy array.

    ``A_gpu`` may be a cached CUDA tensor for repeated multiplies by the same
    left factor. This is used by the frozen-Binv ODE path, where a large
    pseudoinverse multiplies several tiny RHS matrices.
    """
    n = (
        max(A.shape)
        if A is not None
        else (A_gpu.shape[0] if A_gpu is not None else 0)
    )
    if A_gpu is not None or _gpu_matmul_ok(n):
        try:
            At = A_gpu if A_gpu is not None else to_gpu(A)
            Bt = to_gpu(B)
            if At is not None and Bt is not None:
                return (At @ Bt).cpu().numpy()
        except (RuntimeError, MemoryError):
            _record_oom(n)
    return None


def gpu_left_factor(A):
    """Upload a reusable left matmul factor when it is large enough."""
    n = max(A.shape)
    if not _gpu_matmul_ok(n):
        return None
    return to_gpu(A)


def gpu_svd(M, full_matrices=True):
    """Economy/full SVD on GPU when beneficial, else None.

    Returns (U, S, Vh) as numpy arrays (same convention as numpy.linalg.svd),
    or ``None`` when the GPU path is unavailable/too small/failed so the caller
    can run its own CPU path (e.g. the gesdd->gesvd robust fallback). Note
    cuSOLVER can itself raise on non-convergence (torch.linalg.LinAlgError, a
    RuntimeError subclass) -- that is caught here and reported as None so the
    caller's robust CPU driver takes over.
    """
    n = M.shape[0]
    if _gpu_ok(n):
        try:
            Mt = to_gpu(M)
            if Mt is not None:
                U_t, S_t, Vh_t = torch.linalg.svd(Mt, full_matrices=full_matrices)
                return U_t.cpu().numpy(), S_t.cpu().numpy(), Vh_t.cpu().numpy()
        except (RuntimeError, MemoryError):
            _record_oom(n)
    return None


def gpu_pinv(B, rcond=1e-15):
    """Moore-Penrose pseudoinverse on GPU when beneficial, else None.

    Computed via GPU SVD with the same relative cutoff (rcond * largest sv) as
    numpy.linalg.pinv, so results match the CPU path to machine precision in
    fp64. Returns a numpy array, or ``None`` when the GPU path is
    unavailable/too small/failed (caller falls back to CPU).
    """
    n = B.shape[0]
    if _gpu_ok(n):
        try:
            Bt = to_gpu(B)
            if Bt is not None:
                U_t, S_t, Vh_t = torch.linalg.svd(Bt, full_matrices=False)
                cutoff = rcond * S_t[0] if S_t.numel() else S_t.new_zeros(())
                Sinv = torch.where(S_t > cutoff, 1.0 / S_t,
                                   torch.zeros_like(S_t))
                P_t = (Vh_t.transpose(-2, -1) * Sinv) @ U_t.transpose(-2, -1)
                return P_t.cpu().numpy()
        except (RuntimeError, MemoryError):
            _record_oom(n)
    return None


def gpu_project(H, U, H_gpu=None):
    """Compute U.T @ H @ U on GPU when beneficial.

    Returns a numpy array. If H_gpu is supplied (cached), the H upload is
    skipped — this is the main motivation for the helper, since the same H
    is read by both the BFGS eigh and the projection in the same step.
    """
    n = H_gpu.shape[0] if H_gpu is not None else H.shape[0]
    if H_gpu is not None or _gpu_ok(n):
        try:
            Ht = H_gpu if H_gpu is not None else to_gpu(H)
            Ut = to_gpu(U)
            if Ht is not None and Ut is not None:
                # Project on device, download only the (k, k) result.
                R_t = Ut.T @ Ht @ Ut
                return R_t.cpu().numpy()
        except (RuntimeError, MemoryError):
            _record_oom(n)
    if H is None:
        H = H_gpu.cpu().numpy()
    return U.T @ H @ U
