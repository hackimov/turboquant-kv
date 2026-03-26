import ast
import math
import threading
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Callable, Any, Iterable, Union

import warnings

import torch
import torch.nn.functional as F

CodebookKind = Literal["paper", "ternary"]

CentroidsCacheKey = Tuple[CodebookKind, int, int]


def _levels_from_mse_bits(mse_bits: int) -> int:
    """
    MSE-stage uses ``K = 2^(mse_bits)`` levels for ``mse_bits > 0`` and a degenerate single centroid
    for ``mse_bits <= 0``.
    """
    return 1 if mse_bits <= 0 else 2**mse_bits


def _centroid_levels_for(bits: float, codebook: CodebookKind) -> int:
    """
    How many MSE centroids are required for the given (possibly fractional) total `bits`.

    TurboQuant splits total bitwidth `b` into:
      - MSE-stage: `b - 1` bits
      - QJL-stage: 1 bit (always present)

    For fractional MSE-stage bitwidth we implement "outlier channel allocation" as a mixture:
    a fraction of channels use the higher integer MSE bitwidth.
    """
    if codebook == "ternary":
        return 3

    mse_bits = float(bits) - 1.0
    # Integer bitwidth => integer MSE bitwidth.
    if abs(mse_bits - round(mse_bits)) < 1e-6:
        return _levels_from_mse_bits(int(round(mse_bits)))

    # Fractional case: mse_bits = low + frac, with low integer and frac in (0,1).
    low = int(math.floor(mse_bits))
    return _levels_from_mse_bits(low) + _levels_from_mse_bits(low + 1)


class TurboQuantProd:
    """
    Reference implementation of TurboQuant (Algorithm 2 in the paper):
    - Quantmse: rotation by Π and scalar MSE-optimized quantization per coordinate (Algorithm 1)
    - QJL on residual: sign(S·r) and dequantization via sqrt(pi/2)/d * gamma * S^T * qjl (Definition 1)

    For ``codebook="paper"`` and ``bits >= 4`` (``mse_bits >= 3``), scalar centroids come from a
    class-level Max–Lloyd solve on a Beta-like density; results are cached (CPU float64) under
    ``(codebook, head_dim, mse_bits)``. Use :meth:`preload_centroids` before constructing many
    quantizers, and :meth:`save_centroids_cache` / :meth:`load_centroids_cache` to persist the
    cache across processes.
    """

    # Numerical Lloyd centroids for ``codebook="paper"``, ``mse_bits >= 3`` only (float64 CPU).
    # Keys: ``(codebook, head_dim, mse_bits)``. Legacy two-tuple ``(head_dim, mse_bits)`` is still
    # read for backward compatibility and migrated on write.
    _CENTROIDS_CACHE: Dict[Tuple[Any, ...], torch.Tensor] = {}
    _cache_lock = threading.Lock()

    @classmethod
    def _paper_centroids_cache_get(cls, head_dim: int, mse_bits: int) -> Optional[torch.Tensor]:
        key3: CentroidsCacheKey = ("paper", head_dim, mse_bits)
        with cls._cache_lock:
            hit = cls._CENTROIDS_CACHE.get(key3)
            if hit is not None:
                return hit
            legacy = cls._CENTROIDS_CACHE.get((head_dim, mse_bits))
            if legacy is not None:
                cls._CENTROIDS_CACHE[key3] = legacy
                del cls._CENTROIDS_CACHE[(head_dim, mse_bits)]
                return legacy
        return None

    @classmethod
    def _paper_centroids_cache_set(cls, head_dim: int, mse_bits: int, value: torch.Tensor) -> None:
        key3: CentroidsCacheKey = ("paper", head_dim, mse_bits)
        with cls._cache_lock:
            cls._CENTROIDS_CACHE[key3] = value
            cls._CENTROIDS_CACHE.pop((head_dim, mse_bits), None)

    def __init__(
        self,
        bits: float = 3,
        head_dim: int = 128,
        device: Optional[str] = None,
        seed: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        use_torch_compile: bool = False,
        *,
        codebook: CodebookKind = "paper",
        Pi: Optional[torch.Tensor] = None,
        S: Optional[torch.Tensor] = None,
        centroids: Optional[torch.Tensor] = None,
    ):
        self.bits = float(bits)
        self.head_dim = int(head_dim)
        if codebook not in ("paper", "ternary"):
            raise ValueError('codebook must be "paper" or "ternary"')
        self.codebook: CodebookKind = codebook
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            dev_s = str(device).lower()
            if dev_s in ("metal", "mlx"):
                dev_s = "mps"
            self.device = dev_s
        if "cuda" in str(self.device).lower() and not torch.cuda.is_available():
            warnings.warn("CUDA is not available; falling back device to 'cpu'.")
            self.device = "cpu"
        if str(self.device).lower() == "mps" and not torch.backends.mps.is_available():
            warnings.warn("MPS/Metal is not available; falling back device to 'cpu'.")
            self.device = "cpu"
        self.dtype = dtype

        if self.head_dim <= 1:
            raise ValueError("head_dim must be > 1")
        if self.bits < 1:
            raise ValueError("bits must be >= 1")

        self._centroid_levels = _centroid_levels_for(self.bits, self.codebook)

        use_fixed = Pi is not None or S is not None or centroids is not None
        if use_fixed and (Pi is None or S is None):
            raise ValueError("Pi and S must both be provided when loading fixed matrices (centroids optional).")
        if Pi is not None and S is not None:
            if Pi.shape != (self.head_dim, self.head_dim) or S.shape != (self.head_dim, self.head_dim):
                raise ValueError(f"Pi and S must be [{self.head_dim}, {self.head_dim}]")

        # Make Π and S deterministic when seed is provided (ignored if Pi/S passed in).
        if not use_fixed and seed is not None:
            torch.manual_seed(seed)
            if "cuda" in str(self.device).lower() and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Algorithm 2: total bit-width is b; MSE stage uses (b-1) bits.
        mse_bits = float(self.bits) - 1.0
        mse_bits_low = int(math.floor(mse_bits + 1e-12))
        mse_bits_frac = float(mse_bits - float(mse_bits_low))
        if abs(mse_bits_frac) < 1e-8:
            mse_bits_frac = 0.0
        if mse_bits_frac < 0.0 or mse_bits_frac >= 1.0:
            # Defensive: should never happen for b in [1, +inf).
            mse_bits_frac = 0.0

        self._mse_bits_low = mse_bits_low
        self._mse_bits_frac = mse_bits_frac
        self._mse_bits_high = mse_bits_low if mse_bits_frac == 0.0 else mse_bits_low + 1
        self._use_fractional_mse_bits = self._mse_bits_high != self._mse_bits_low

        self._centroids_low_levels = _levels_from_mse_bits(self._mse_bits_low)
        self._centroids_high_levels = (
            0 if not self._use_fractional_mse_bits else _levels_from_mse_bits(self._mse_bits_high)
        )

        if self.codebook == "ternary" and self._use_fractional_mse_bits:
            raise ValueError('fractional bits are not supported for codebook="ternary"')

        # Optionally compile hot paths. If compilation fails, we fall back to eager mode.
        self._quantize_components_impl: Callable[..., Any] = self._quantize_components
        self._dequantprod_unit_impl: Callable[..., Any] = self._dequantprod_unit
        if use_torch_compile and hasattr(torch, "compile"):
            try:
                self._quantize_components_impl = torch.compile(self._quantize_components, mode="default")  # type: ignore[attr-defined]
                self._dequantprod_unit_impl = torch.compile(self._dequantprod_unit, mode="default")  # type: ignore[attr-defined]
            except Exception:
                # Compilation is best-effort; TurboQuant is still usable without it.
                self._quantize_components_impl = self._quantize_components
                self._dequantprod_unit_impl = self._dequantprod_unit

        if Pi is not None and S is not None:
            self.Pi = Pi.to(device=self.device, dtype=self.dtype)
            self.S = S.to(device=self.device, dtype=self.dtype)
        else:
            # Π: random orthogonal matrix via QR decomposition.
            self.Pi = self._generate_orthogonal_matrix().to(dtype=self.dtype)

            # S: random i.i.d. Gaussian matrix (entries ~ N(0,1)).
            self.S = torch.randn(self.head_dim, self.head_dim, device=self.device, dtype=torch.float32).to(
                dtype=self.dtype
            )

        # Codebook centroids for Quantmse: paper uses 2^(b-1) levels; ternary uses 3 (~1.58 bit / level).
        if centroids is not None:
            self._centroids = centroids.to(device=self.device, dtype=self.dtype)
            if self._centroids.ndim != 1:
                raise ValueError("centroids must be 1-D")
            if self._centroids.numel() != self._centroid_levels:
                raise ValueError(
                    f"centroids length {self._centroids.numel()} != expected {self._centroid_levels} "
                    f"for codebook={self.codebook!r}, bits={self.bits}"
                )
        elif self.codebook == "ternary":
            self._centroids = self._build_ternary_centroids(self.head_dim).to(self.device, dtype=self.dtype)
        else:
            if not self._use_fractional_mse_bits:
                self._centroids = self._build_centroids(self.head_dim, self._mse_bits_low).to(
                    self.device, dtype=self.dtype
                )
            else:
                low_c = self._build_centroids(self.head_dim, self._mse_bits_low).to(self.device, dtype=self.dtype)
                high_c = self._build_centroids(self.head_dim, self._mse_bits_high).to(
                    self.device, dtype=self.dtype
                )
                self._centroids = torch.cat([low_c, high_c], dim=0)
        self._mse_levels = int(self._centroids.numel())

        # sqrt(pi/2)/d factor used in Algorithm 2 dequantization.
        self._qjl_factor = math.sqrt(math.pi / 2.0) / float(self.head_dim)

    def _generate_orthogonal_matrix(self) -> torch.Tensor:
        g = torch.randn(self.head_dim, self.head_dim, device=self.device, dtype=torch.float32)
        q, _ = torch.linalg.qr(g)
        return q

    @classmethod
    def _build_centroids(cls, head_dim: int, mse_bits: int) -> torch.Tensor:
        """
        Build MSE-optimal scalar quantization centroids for Algorithm 1.
        For mse_bits in {0,1,2} we use the closed-form/approx values from the paper.
        For mse_bits >= 3 we numerically solve the 1D continuous k-means (Max-Lloyd style)
        for the Beta-like density f_X(x) on [-1, 1].
        """
        if mse_bits <= 0:
            return torch.tensor([0.0], dtype=torch.float64)

        d = float(head_dim)

        if mse_bits == 1:
            # Paper example: {±sqrt(2/pi) / sqrt(d)}.
            val = math.sqrt(2.0 / math.pi) / math.sqrt(d)
            return torch.tensor([-val, val], dtype=torch.float64)

        if mse_bits == 2:
            # Paper example: {±0.453/sqrt(d), ±1.51/sqrt(d)}.
            v1 = 0.453 / math.sqrt(d)
            v2 = 1.51 / math.sqrt(d)
            # Ascending order.
            return torch.tensor([-v2, -v1, v1, v2], dtype=torch.float64)

        # Numerical Max-Lloyd for mse_bits >= 3.
        k = 2 ** mse_bits
        cached = cls._paper_centroids_cache_get(head_dim, mse_bits)
        if cached is not None:
            return cached

        # Use an unnormalized density: g(x) = (1 - x^2)^((d-3)/2). Normalization cancels
        # in the conditional expectation update.
        # Integration grid.
        grid_n = 20001
        x_grid = torch.linspace(-1.0, 1.0, grid_n, dtype=torch.float64)
        exp = (d - 3.0) / 2.0
        one_minus_sq = 1.0 - x_grid * x_grid
        g = torch.pow(torch.clamp(one_minus_sq, min=0.0), exp)  # [grid_n]

        # Initialize centroids uniformly.
        c = torch.linspace(-1.0, 1.0, k, dtype=torch.float64)

        # Helper to update centroids from cluster boundaries.
        def update_centroids(c_in: torch.Tensor) -> torch.Tensor:
            c_in = torch.sort(c_in).values
            # Boundaries are -1, midpoints, +1.
            mids = 0.5 * (c_in[:-1] + c_in[1:])
            bounds = torch.cat([torch.tensor([-1.0], dtype=torch.float64), mids, torch.tensor([1.0], dtype=torch.float64)])
            c_new = torch.empty_like(c_in)
            for i in range(k):
                a = bounds[i].item()
                b = bounds[i + 1].item()
                if i == 0:
                    mask = (x_grid >= a) & (x_grid <= b)
                else:
                    mask = (x_grid > a) & (x_grid <= b)
                denom = g[mask].sum()
                if denom.item() == 0.0:
                    c_new[i] = c_in[i]
                else:
                    c_new[i] = (x_grid[mask] * g[mask]).sum() / denom
            return c_new

        # Iterate until convergence.
        for _ in range(50):
            c_new = update_centroids(c)
            delta = torch.max(torch.abs(c_new - c)).item()
            c = c_new
            if delta < 1e-8:
                break

        # Ensure ascending order.
        c_final = torch.sort(c).values
        c_cpu = c_final.detach().cpu()

        again = cls._paper_centroids_cache_get(head_dim, mse_bits)
        if again is not None:
            return again

        cls._paper_centroids_cache_set(head_dim, mse_bits, c_cpu)
        return c_final

    @classmethod
    def _build_ternary_centroids(cls, head_dim: int) -> torch.Tensor:
        """
        Default symmetric ternary codebook {-t, 0, +t} in rotated coordinates (same scale as paper's 2-level code).
        """
        d = float(head_dim)
        t = math.sqrt(2.0 / math.pi) / math.sqrt(d)
        return torch.tensor([-t, 0.0, t], dtype=torch.float64)

    @classmethod
    def _normalize_codebooks(cls, codebook: Union[CodebookKind, Iterable[CodebookKind]]) -> List[CodebookKind]:
        if isinstance(codebook, str):
            if codebook not in ("paper", "ternary"):
                raise ValueError('codebook must be "paper" or "ternary"')
            return [codebook]
        out: List[CodebookKind] = []
        for cb in codebook:
            if cb not in ("paper", "ternary"):
                raise ValueError('each codebook must be "paper" or "ternary"')
            out.append(cb)
        return out

    @classmethod
    def preload_centroids(
        cls,
        head_dims: Iterable[int] = (64, 128, 256),
        bits: Iterable[int] = (2, 3, 4),
        *,
        codebook: Union[CodebookKind, Iterable[CodebookKind]] = "paper",
    ) -> int:
        """
        Warm the class-level centroid cache so the first ``TurboQuantProd(...)`` with matching
        ``head_dim`` / ``bits`` avoids heavy Lloyd iterations (relevant for ``codebook="paper"``
        and ``bits >= 4``, i.e. ``mse_bits >= 3``).

        Ternary codebooks use closed-form centroids — they do not use this class cache.

        Parameters
        ----------
        head_dims, bits:
            Cartesian product of pairs to warm up for ``codebook="paper"``.
        codebook:
            ``"paper"``, ``"ternary"``, or an iterable of kinds. Only ``"paper"`` triggers
            :meth:`_build_centroids` (and thus numerical cache entries).

        Returns
        -------
        int
            How many times :meth:`_build_centroids` was invoked for ``codebook="paper"`` (each
            call is cheap for ``mse_bits < 3`` and hits the cache for repeated ``(head_dim, mse_bits)``).
        """
        codebooks = cls._normalize_codebooks(codebook)
        bits_l = [int(b) for b in bits]
        head_dims_l = [int(hd) for hd in head_dims]
        n_build = 0
        for cb in codebooks:
            if cb != "paper":
                continue
            for hd in head_dims_l:
                if hd <= 1:
                    raise ValueError("head_dim must be > 1")
                for b in bits_l:
                    if b < 1:
                        raise ValueError("bits must be >= 1")
                    mse_bits = b - 1
                    if mse_bits >= 0:
                        cls._build_centroids(hd, mse_bits)
                        n_build += 1
        return n_build

    @classmethod
    def clear_centroids_cache(cls) -> None:
        """Remove all entries from the paper numerical centroid cache (e.g. before tests or to free RAM)."""
        with cls._cache_lock:
            cls._CENTROIDS_CACHE.clear()

    @classmethod
    def centroids_cache_len(cls) -> int:
        """Number of cached tensors (after legacy-key migration, keys are 3-tuples)."""
        with cls._cache_lock:
            return len(cls._CENTROIDS_CACHE)

    @classmethod
    def centroids_cache_keys(cls) -> List[CentroidsCacheKey]:
        """
        Snapshot of cache keys ``(codebook, head_dim, mse_bits)`` for paper numerical entries.
        Legacy two-tuples are normalized on access and should not appear after warm-up.
        """
        with cls._cache_lock:
            raw = list(cls._CENTROIDS_CACHE.keys())
        keys: List[CentroidsCacheKey] = []
        for k in raw:
            if len(k) == 3 and k[0] in ("paper", "ternary"):
                keys.append((k[0], int(k[1]), int(k[2])))
            elif len(k) == 2:
                keys.append(("paper", int(k[0]), int(k[1])))
        keys.sort()
        return keys

    @classmethod
    def save_centroids_cache(cls, path: Union[str, Path]) -> int:
        """
        Persist the centroid cache to ``path`` via ``torch.save`` (CPU tensors).

        Returns the number of entries written.
        """
        p = Path(path)
        with cls._cache_lock:
            payload = {repr(k): v.clone() for k, v in cls._CENTROIDS_CACHE.items()}
            n = len(payload)
        torch.save({"version": 1, "entries": payload}, p)
        return n

    @classmethod
    def load_centroids_cache(
        cls,
        path: Union[str, Path],
        *,
        merge: bool = True,
        map_location: Union[str, torch.device] = "cpu",
    ) -> int:
        """
        Load cache from :meth:`save_centroids_cache`. If ``merge=False``, replaces the entire cache.

        Returns the number of entries merged or installed.
        """
        p = Path(path)
        try:
            blob = torch.load(p, map_location=map_location, weights_only=False)
        except TypeError:
            blob = torch.load(p, map_location=map_location)
        if not isinstance(blob, dict) or "entries" not in blob:
            raise ValueError("not a turboquant centroids cache file")
        entries = blob["entries"]
        if not isinstance(entries, dict):
            raise ValueError("corrupt centroids cache: entries must be a dict")

        def parse_key(s: str) -> Tuple[Any, ...]:
            t = ast.literal_eval(s)
            if isinstance(t, tuple) and len(t) == 2:
                return ("paper", int(t[0]), int(t[1]))
            if isinstance(t, tuple) and len(t) == 3:
                cb = t[0]
                if cb not in ("paper", "ternary"):
                    raise ValueError(f"bad codebook in key {s!r}")
                return (cb, int(t[1]), int(t[2]))
            raise ValueError(f"bad cache key {s!r}")

        n = 0
        with cls._cache_lock:
            if not merge:
                cls._CENTROIDS_CACHE.clear()
            for ks, tensor in entries.items():
                key = parse_key(str(ks))
                cls._CENTROIDS_CACHE[key] = tensor.clone().detach().cpu()
                n += 1
        return n

    def _quantmse(self, x_unit: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Algorithm 1:
          y <- Π · x
          idx_j <- arg min_k |y_j - c_k|
          output idx

        Here x_unit is row-major [N, d], and we use y = x_unit @ Π^T.
        Returns:
          idx: [N, d] int64
          x_tilde_unit: [N, d] float32 reconstructed vector on unit sphere.
        """
        # y = Π x (column), row-major equivalence: y_row = x_row @ Π^T
        y = x_unit @ self.Pi.T  # [N, d]
        c = self._centroids.to(device=y.device, dtype=y.dtype)  # [K]

        # Compute argmin_k |y_j - c_k| for each coordinate j.
        # y[..., None] -> [N, d, 1], c[None,None,:] -> [1,1,K]
        # Use float32 distances for stable argmin even when dtype is fp16/bf16.
        dist = torch.abs(y.float().unsqueeze(-1) - c.float().view(1, 1, -1))
        idx = torch.argmin(dist, dim=-1).to(torch.int64)  # [N, d] in [0, K-1]

        # DeQuantmse: y_tilde_j = c_idxj ; x_tilde = Π^T y_tilde (column)
        # row-major: x_tilde_row = y_tilde_row @ Π
        y_tilde = c[idx]  # [N, d]
        x_tilde_unit = y_tilde @ self.Pi  # [N, d]
        return idx, x_tilde_unit

    def _quantmse_fractional(
        self, x_unit: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fractional `bits` implementation via outlier channel allocation.

        We allocate a fraction of coordinates with higher integer MSE bitwidth, and all other
        coordinates use the lower integer MSE bitwidth. The decoder infers which centroid set
        was used from `idx` ranges (no extra mask stored).
        """
        # y = Π x_unit, row-major: [N, d]
        y = x_unit @ self.Pi.T
        n, d = y.shape

        # Fraction of coordinates assigned to the higher MSE bitwidth.
        frac = float(self._mse_bits_frac)
        k_out = int(round(frac * d))
        k_out = max(0, min(d, k_out))

        # Slice centroid bank as [low, high].
        k_low = int(self._centroids_low_levels)
        c_low = self._centroids[:k_low]
        c_high = self._centroids[k_low:]

        # Argmin for low and high independently; selection happens by idx-range.
        if k_low == 1:
            idx_low = torch.zeros((n, d), device=y.device, dtype=torch.int64)
        else:
            dist_low = torch.abs(y.float().unsqueeze(-1) - c_low.float().view(1, 1, -1))  # [N,d,k_low]
            idx_low = torch.argmin(dist_low, dim=-1).to(torch.int64)

        k_high = int(c_high.numel())
        if k_high == 1:
            idx_high = torch.zeros((n, d), device=y.device, dtype=torch.int64)
        else:
            dist_high = torch.abs(y.float().unsqueeze(-1) - c_high.float().view(1, 1, -1))  # [N,d,k_high]
            idx_high = torch.argmin(dist_high, dim=-1).to(torch.int64)

        if k_out == 0:
            idx = idx_low
        else:
            abs_y = torch.abs(y)
            topk = torch.topk(abs_y, k_out, dim=-1, largest=True, sorted=False).indices  # [N,k_out]
            mask = torch.zeros((n, d), device=y.device, dtype=torch.bool)
            mask.scatter_(dim=-1, index=topk, value=True)
            idx = idx_low.clone()
            idx[mask] = idx_high[mask] + k_low

        y_tilde = self._centroids[idx]  # [N, d]
        x_tilde_unit = y_tilde @ self.Pi  # [N, d]
        return idx, x_tilde_unit

    def _dequantprod_unit(self, idx: torch.Tensor, qjl_sign: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """
        Algorithm 2 DeQuantprod on unit vectors:
          x_tilde_mse <- DeQuantmse(idx)
          x_tilde_qjl <- sqrt(pi/2)/d * gamma * S^T · qjl
          output x_tilde = x_tilde_mse + x_tilde_qjl
        """
        c = self._centroids.to(idx.device, dtype=self.dtype)  # [K]

        # x_tilde_mse_unit = DeQuantmse(idx) on unit sphere.
        # y_tilde = c[idx]; x_tilde_unit = y_tilde @ Π
        y_tilde = c[idx]  # [N, d]
        x_tilde_unit = y_tilde @ self.Pi  # [N, d]

        # qjl_sign is +/-1 per coordinate.
        # row-major: (S^T qjl_col)^T == qjl_row @ S
        x_tilde_qjl_unit = self._qjl_factor * gamma * (qjl_sign @ self.S)  # [N, d]
        return x_tilde_unit + x_tilde_qjl_unit

    def _quantize_components(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize only the components needed for DeQuantprod:
          - idx (Quantmse indices)
          - x_norm = ||x||_2
          - qjl_sign = sign(S · r)
          - gamma = ||r||_2

        This avoids building the full `quantized` vector, which can be useful for KV cache packing.
        """
        orig_shape = x.shape
        x = x.reshape(-1, self.head_dim).to(self.device, dtype=self.dtype)

        x_norm = torch.linalg.norm(x.float(), dim=-1, keepdim=True).clamp(min=1e-8)  # [N,1] float32
        x_norm_d = x_norm.to(dtype=self.dtype)
        x_unit = (x / x_norm_d)  # [N, d] in self.dtype

        if self._use_fractional_mse_bits:
            idx, x_tilde_mse_unit = self._quantmse_fractional(x_unit)
        else:
            idx, x_tilde_mse_unit = self._quantmse(x_unit)

        r_unit = x_unit - x_tilde_mse_unit
        gamma = torch.linalg.norm(r_unit.float(), dim=-1, keepdim=True).clamp(min=0.0)  # float32
        gamma_d = gamma.to(dtype=self.dtype)

        u = r_unit @ self.S.T  # [N,d] in self.dtype
        ones = torch.ones_like(u)
        qjl_sign = torch.where(u >= 0, ones, -ones)  # +/-1 in self.dtype

        return (
            idx.view(orig_shape[:-1] + (self.head_dim,)),
            x_norm_d.view(orig_shape[:-1] + (1,)),
            qjl_sign.view(orig_shape[:-1] + (self.head_dim,)),
            gamma_d.view(orig_shape[:-1] + (1,)),
        )

    def quantize(self, x: torch.Tensor):
        """
        Quantize arbitrary x by:
          - normalizing to unit sphere (paper assumes x in S^{d-1})
          - Algorithm 2 produces reconstructed unit vector
          - scale reconstruction back by ||x||_2

        Returns tuple:
          (quantized, idx, x_norm, qjl_sign, gamma)
        """
        with torch.no_grad():
            idx, x_norm, qjl_sign, gamma = self._quantize_components_impl(x)
            # Reconstruct full quantized vector only when requested.
            idx_flat = idx.reshape(-1, self.head_dim)
            qjl_sign_flat = qjl_sign.reshape(-1, self.head_dim)
            gamma_flat = gamma.reshape(-1, 1)
            x_tilde_unit = self._dequantprod_unit_impl(idx_flat, qjl_sign_flat, gamma_flat)
            quantized = (x_tilde_unit * x_norm.reshape(-1, 1)).reshape(idx.shape)  # [*, d]

            orig_shape = x.shape

            return (
                quantized.view(orig_shape),
                idx,
                x_norm,
                qjl_sign,
                gamma,
            )

    def dequantize(
        self,
        idx: torch.Tensor,
        x_norm: torch.Tensor,
        qjl_sign: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Dequantize from compressed representation:
          - idx from Quantmse
          - x_norm from input normalization
          - qjl_sign and gamma from QJL residual stage
        """
        with torch.no_grad():
            idx = idx.to(self.device)
            x_norm = (
                x_norm.to(self.device, dtype=self.dtype).float()
                if self.dtype == torch.float32
                else x_norm.to(self.device, dtype=self.dtype)
            )
            qjl_sign = qjl_sign.to(self.device, dtype=self.dtype)
            gamma = gamma.to(self.device, dtype=self.dtype)

            x_tilde_unit = self._dequantprod_unit_impl(idx, qjl_sign, gamma)  # [N, d]
            return (x_tilde_unit * x_norm).view(idx.shape[:-1] + (self.head_dim,))

    def compress(self, k: torch.Tensor, v: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Convenience API: compress KV into a dictionary that can be stored/transmitted.
        """
        return self.quantize_kv(k, v, return_compressed=True)

    def decompress(self, compressed_kv: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience API: reconstruct (k, v) from `compress()` output.
        """
        k = self.dequantize(
            compressed_kv["k_idx"],
            compressed_kv["k_norm"],
            compressed_kv["k_sign"],
            compressed_kv["k_gamma"],
        )
        v = self.dequantize(
            compressed_kv["v_idx"],
            compressed_kv["v_norm"],
            compressed_kv["v_sign"],
            compressed_kv["v_gamma"],
        )
        return k, v

    def quantize_kv_cache(self, k: torch.Tensor, v: torch.Tensor, return_compressed: bool = True) -> Dict[str, torch.Tensor]:
        """
        Transformers-style helper (KV cache) — currently equivalent to `quantize_kv`.

        In a full transformers integration this method can be adapted to paged KV formats.
        """
        return self.quantize_kv(k, v, return_compressed=return_compressed)

    def decompress_kv_cache(self, compressed_kv: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse of `quantize_kv_cache` for compressed dictionary format.
        """
        return self.decompress(compressed_kv)

    def quantized_attention_scores_triton(
        self,
        q: torch.Tensor,
        kv_dict_k: Dict[str, torch.Tensor],
        *,
        num_kv_heads: Optional[int] = None,
        causal: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute attention scores using Triton with compressed TurboQuant KV cache for keys.

        Parameters
        ----------
        q: torch.Tensor
            Query tensor with shape [B, H_q, M, head_dim]
        kv_dict_k: dict
            Compressed KV dict for keys (output of quantize_kv(..., return_compressed=True))
            containing required keys: k_idx, k_norm, k_sign, k_gamma.
        num_kv_heads: int, optional
            Number of KV heads ``H_kv`` (GQA/MQA). Default: same as ``H_q``.
        causal: bool
            If True, mask positions where key index ``n > query index`` ``m`` (self-attention
            with aligned positions ``0..N-1`` and ``M == N``).
        attention_mask: torch.Tensor, optional
            Additive mask (bool or float), shapes ``[M,N]``, ``[B,M,N]``, or ``[B,H,M,N]``;
            applied before ``causal``. Boolean ``False`` means mask out (``-inf``).
        """
        from .kernels import turboquant_attention

        if q.ndim != 4:
            raise ValueError("q must have shape [B, H, M, head_dim]")

        if any(k not in kv_dict_k for k in ("k_idx", "k_norm", "k_sign", "k_gamma")):
            raise ValueError("kv_dict_k must include k_idx, k_norm, k_sign, k_gamma")

        if q.shape[-1] != self.head_dim:
            raise ValueError(f"q last dim must be head_dim={self.head_dim}")

        with torch.no_grad():
            # Precompute projections used by the fused formula:
            #   q_pi = q @ Pi.T
            #   q_s  = q @ S.T
            q_pi = torch.matmul(q, self.Pi.T)
            q_s = torch.matmul(q, self.S.T)

            return turboquant_attention(
                q_pi,
                q_s,
                kv_dict_k,
                centroids=self._centroids,
                qjl_factor=float(self._qjl_factor),
                num_kv_heads=num_kv_heads,
                causal=causal,
                attention_mask=attention_mask,
            )

    def quantized_attention_fused_triton(
        self,
        q: torch.Tensor,
        kv_dict: Dict[str, torch.Tensor],
        *,
        num_kv_heads: Optional[int] = None,
        causal: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fused attention: ``softmax(Q K^T / sqrt(d)) V`` with TurboQuant-compressed K and V (dense layout).

        ``kv_dict`` must be the output of ``quantize_kv(..., return_compressed=True)`` (includes v_*).
        Returns tensor ``[B, H_q, M, head_dim]`` in float32.

        For GQA/MQA, pass ``num_kv_heads=H_kv``; ``k_*``/``v_*`` must use shape ``[B, H_kv, N, ...]``.

        ``attention_mask`` — same semantics as ``quantized_attention_scores_triton`` (last dim ``N``).
        """
        from .kernels.fused_attention import turboquant_fused_attention_dense

        if q.ndim != 4:
            raise ValueError("q must have shape [B, H, M, head_dim]")
        if q.shape[-1] != self.head_dim:
            raise ValueError(f"q last dim must be head_dim={self.head_dim}")
        required = ("k_idx", "k_norm", "k_sign", "k_gamma", "v_idx", "v_norm", "v_sign", "v_gamma")
        if any(k not in kv_dict for k in required):
            raise ValueError(f"kv_dict must include {required}")

        with torch.no_grad():
            q_pi = torch.matmul(q, self.Pi.T)
            q_s = torch.matmul(q, self.S.T)
            return turboquant_fused_attention_dense(
                q_pi,
                q_s,
                kv_dict,
                centroids=self._centroids,
                qjl_factor=float(self._qjl_factor),
                pi=self.Pi,
                s=self.S,
                num_kv_heads=num_kv_heads,
                causal=causal,
                attention_mask=attention_mask,
            )

    def quantized_attention_fused_torch(
        self,
        q: torch.Tensor,
        kv_dict: Dict[str, torch.Tensor],
        *,
        num_kv_heads: Optional[int] = None,
        causal: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Portable fused attention path (CUDA/CPU/MPS) via dequantized KV + SDPA.

        This path exists primarily for backends where Triton is unavailable (for example Apple Metal/MPS).
        """
        if q.ndim != 4:
            raise ValueError("q must have shape [B, H, M, head_dim]")
        if q.shape[-1] != self.head_dim:
            raise ValueError(f"q last dim must be head_dim={self.head_dim}")
        required = ("k_idx", "k_norm", "k_sign", "k_gamma", "v_idx", "v_norm", "v_sign", "v_gamma")
        if any(k not in kv_dict for k in required):
            raise ValueError(f"kv_dict must include {required}")

        with torch.no_grad():
            k, v = self.decompress(kv_dict)
            dev = q.device
            qf = q.to(device=dev, dtype=torch.float32)
            kf = k.to(device=dev, dtype=torch.float32)
            vf = v.to(device=dev, dtype=torch.float32)

            B, H, M, _ = qf.shape
            H_kv = int(kf.shape[1])
            N = int(kf.shape[2])
            expected_h_kv = int(num_kv_heads) if num_kv_heads is not None else H
            if H_kv != expected_h_kv:
                raise ValueError(f"KV heads mismatch: got {H_kv}, expected num_kv_heads={expected_h_kv}")
            if H % H_kv != 0:
                raise ValueError(f"num query heads ({H}) must be divisible by num_kv_heads ({H_kv})")
            if H_kv != H:
                rep = H // H_kv
                kf = kf.repeat_interleave(rep, dim=1)
                vf = vf.repeat_interleave(rep, dim=1)

            mask_t: Optional[torch.Tensor] = None
            if attention_mask is not None:
                from .kernels.attention_mask import broadcast_additive_attn_mask

                mask_t = broadcast_additive_attn_mask(attention_mask, B, H, M, N, device=dev)
            if causal:
                ar_m = torch.arange(M, device=dev, dtype=torch.long).view(1, 1, M, 1)
                ar_n = torch.arange(N, device=dev, dtype=torch.long).view(1, 1, 1, N)
                c_mask = torch.where(
                    ar_n <= ar_m,
                    torch.zeros((), device=dev, dtype=torch.float32),
                    torch.full((), float("-inf"), device=dev, dtype=torch.float32),
                )
                mask_t = c_mask if mask_t is None else (mask_t + c_mask)

            out = F.scaled_dot_product_attention(
                qf,
                kf,
                vf,
                attn_mask=mask_t,
                dropout_p=0.0,
                is_causal=False,
            )
            return out.to(torch.float32)

    def quantized_attention_fused_metal(
        self,
        q: torch.Tensor,
        kv_dict: Dict[str, torch.Tensor],
        *,
        num_kv_heads: Optional[int] = None,
        causal: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apple Metal path (MPS) for fused attention.
        """
        if q.device.type != "mps":
            raise ValueError("quantized_attention_fused_metal expects q on an MPS device")
        return self.quantized_attention_fused_torch(
            q,
            kv_dict,
            num_kv_heads=num_kv_heads,
            causal=causal,
            attention_mask=attention_mask,
        )

    def quantized_attention_fused_auto(
        self,
        q: torch.Tensor,
        kv_dict: Dict[str, torch.Tensor],
        *,
        num_kv_heads: Optional[int] = None,
        causal: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Select fused attention backend automatically:
        - Triton on CUDA when available;
        - fallback SDPA path otherwise (including Apple Metal/MPS).
        """
        if q.is_cuda:
            try:
                return self.quantized_attention_fused_triton(
                    q,
                    kv_dict,
                    num_kv_heads=num_kv_heads,
                    causal=causal,
                    attention_mask=attention_mask,
                )
            except ModuleNotFoundError as e:
                if e.name != "triton":
                    raise
        return self.quantized_attention_fused_torch(
            q,
            kv_dict,
            num_kv_heads=num_kv_heads,
            causal=causal,
            attention_mask=attention_mask,
        )

    def quantized_attention_fused_triton_paged(
        self,
        q: torch.Tensor,
        paged_kv: Dict[str, torch.Tensor],
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        block_size: int,
        max_seq_len: int,
        *,
        num_kv_heads: Optional[int] = None,
        causal: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Same as ``quantized_attention_fused_triton`` but KV in paged physical layout.

        ``paged_kv`` uses keys ``*_phys`` as produced by ``turboquant.kernels.fused_attention.pack_dense_kv_to_paged``.
        Physical tensors use head dimension ``H_kv`` (see ``num_kv_heads``).
        ``attention_mask`` last key dimension must equal ``max_seq_len``.
        """
        from .kernels.fused_attention import turboquant_fused_attention_paged

        if q.ndim != 4:
            raise ValueError("q must have shape [B, H, M, head_dim]")
        if q.shape[-1] != self.head_dim:
            raise ValueError(f"q last dim must be head_dim={self.head_dim}")

        with torch.no_grad():
            q_pi = torch.matmul(q, self.Pi.T)
            q_s = torch.matmul(q, self.S.T)
            return turboquant_fused_attention_paged(
                q_pi,
                q_s,
                paged_kv,
                block_tables,
                context_lens,
                block_size,
                max_seq_len,
                centroids=self._centroids,
                qjl_factor=float(self._qjl_factor),
                pi=self.Pi,
                s=self.S,
                num_kv_heads=num_kv_heads,
                causal=causal,
                attention_mask=attention_mask,
            )

    def quantize_kv(self, k: torch.Tensor, v: torch.Tensor, return_compressed: bool = True) -> Dict:
        """
        Quantize KV cache.

        If `return_compressed=True`, avoids reconstructing full quantized tensors and returns only:
        indices + norms + sign/gamma (suitable for cache packing).

        If `return_compressed=False`, additionally returns `k_quant`/`v_quant`.
        """
        with torch.no_grad():
            if return_compressed:
                k_idx, k_norm, k_sign, k_gamma = self._quantize_components_impl(k)
                v_idx, v_norm, v_sign, v_gamma = self._quantize_components_impl(v)
                return {
                    "k_idx": k_idx,
                    "k_norm": k_norm,
                    "k_sign": k_sign,
                    "k_gamma": k_gamma,
                    "v_idx": v_idx,
                    "v_norm": v_norm,
                    "v_sign": v_sign,
                    "v_gamma": v_gamma,
                }

            kq, k_idx, k_norm, k_sign, k_gamma = self.quantize(k)
            vq, v_idx, v_norm, v_sign, v_gamma = self.quantize(v)
            return {
                "k_quant": kq,
                "k_idx": k_idx,
                "k_norm": k_norm,
                "k_sign": k_sign,
                "k_gamma": k_gamma,
                "v_quant": vq,
                "v_idx": v_idx,
                "v_norm": v_norm,
                "v_sign": v_sign,
                "v_gamma": v_gamma,
            }


def concat_compressed_kv(
    left: Optional[Dict[str, torch.Tensor]],
    right: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Concatenate two ``quantize_kv(..., return_compressed=True)`` dicts along sequence dim (dim 2).

    Used for incremental KV: quantize only the new tokens, then concat to the existing cache without
    re-quantizing the full history. Numerically this differs slightly from ``quantize_kv(cat(K))`` on
    the full float cache.
    """
    if left is None:
        return {k: v.clone() for k, v in right.items()}
    out: Dict[str, torch.Tensor] = {}
    for k in left:
        if k not in right:
            raise KeyError(f"right dict missing key {k!r}")
        out[k] = torch.cat([left[k], right[k]], dim=2)
    return out


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = TurboQuantProd(bits=3, head_dim=128, device=device)
    print("Lightweight TurboQuantProd initialized")