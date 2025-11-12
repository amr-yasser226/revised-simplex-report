#!/usr/bin/env python3
"""
A compact, readable Revised Simplex implementation (LU-based) intended
for educational / small-scale testing. It intentionally rebuilds the LU
factorization after each pivot for simplicity and numerical robustness.

API:
    RevisedSimplex(A, b, c, B_init, tol=1e-12, sense='max')
    -> .solve(max_iters=1000) -> dict with keys:
       - status: 'optimal' / 'unbounded' / 'max_iters' / 'error'
       - x: solution vector (when available)
       - objective: float (when available)
       - iterations: number of pivots performed
       - rebuilds: number of LU rebuilds
       - B_idx: final basis indices (list)
"""
from __future__ import annotations
from typing import List, Sequence, Dict, Any, Optional
import numpy as np

try:
    from scipy.linalg import lu_factor, lu_solve
except Exception as e:
    raise ImportError("scipy.linalg.lu_factor and lu_solve are required") from e


class RevisedSimplex:
    def __init__(
        self,
        A: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        B_init: Sequence[int],
        tol: float = 1e-12,
        sense: str = "max",
    ):
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)
        c = np.asarray(c, dtype=float)
        m, n = A.shape
        if b.shape[0] != m:
            raise ValueError("Incompatible dimensions: b must have length m")
        if c.shape[0] != n:
            raise ValueError("Incompatible dimensions: c must have length n")
        if len(B_init) != m:
            raise ValueError("B_init must have length m (one basic index per row)")

        self.A = A
        self.b = b
        self.c = c
        self.m = m
        self.n = n
        self.tol = float(tol)
        if sense not in ("max", "min"):
            raise ValueError("sense must be 'max' or 'min'")
        self.sense = sense

        # index lists
        self.B_idx: List[int] = list(int(i) for i in B_init)
        self.N_idx: List[int] = [j for j in range(n) if j not in self.B_idx]

        # state
        self.iterations = 0
        self.rebuilds = 0
        self._rebuild_basis()

    # -------------------------
    def _rebuild_basis(self) -> None:
        """Rebuild LU factorization of the current basis and compute x_B."""
        B = self.A[:, self.B_idx]
        # LU factorization (with partial pivoting)
        self.lu_and_piv = lu_factor(B)
        self.rebuilds += 1
        # Solve B x_B = b
        self.xB = lu_solve(self.lu_and_piv, self.b)

        # basis cost
        self.cB = self.c[self.B_idx]

    def _reduced_costs_and_dual(self):
        """Return (red, y) where red is reduced costs for nonbasics and y is dual."""
        # Solve B^T y = cB  -> use trans=1 in lu_solve
        y = lu_solve(self.lu_and_piv, self.cB, trans=1)
        AN = self.A[:, self.N_idx]
        cN = self.c[self.N_idx]
        red = cN - AN.T.dot(y)
        return red, y

    def _choose_entering(self, red: np.ndarray) -> Optional[int]:
        """Return local index into N_idx of the entering variable, or None if optimal."""
        if self.sense == "max":
            idx = int(np.argmax(red))
            if red[idx] <= self.tol:
                return None
            return idx
        else:  # minimization
            idx = int(np.argmin(red))
            if red[idx] >= -self.tol:
                return None
            return idx

    def _ratio_test(self, d: np.ndarray) -> Optional[int]:
        """Given direction d = B^{-1} a_q, return local leaving index into B_idx or None if unbounded."""
        pos = d > self.tol
        if not np.any(pos):
            return None
        ratios = np.full(d.shape, np.inf)
        ratios[pos] = self.xB[pos] / d[pos]
        # pick smallest ratio; ties: pick first (can improve to Bland later)
        leave_local = int(np.argmin(ratios))
        return leave_local

    def iterate_once(self):
        """Perform a single pivot iteration. Return status and bookkeeping info."""
        red, y = self._reduced_costs_and_dual()
        entering_local = self._choose_entering(red)
        if entering_local is None:
            return "optimal", None, None

        q = self.N_idx[entering_local]
        a_q = self.A[:, q]
        d = lu_solve(self.lu_and_piv, a_q)  # solve B d = a_q

        # unbounded?
        leave_local = self._ratio_test(d)
        if leave_local is None:
            return "unbounded", q, d

        # pivot: replace leaving basis by entering
        p = self.B_idx[leave_local]
        self.B_idx[leave_local] = q
        self.N_idx[entering_local] = p

        # rebuild basis data (simple strategy)
        self._rebuild_basis()
        self.iterations += 1
        return "continue", p, q

    def solve(self, max_iters: int = 1000) -> Dict[str, Any]:
        """Run simplex until termination or max_iters. Returns result dict."""
        try:
            for _ in range(max_iters):
                status, p, q = self.iterate_once()
                if status == "optimal":
                    # produce full x
                    x = np.zeros(self.n, dtype=float)
                    for i, bi in enumerate(self.B_idx):
                        x[bi] = float(self.xB[i])
                    obj = float(self.c.dot(x))
                    return {
                        "status": "optimal",
                        "x": x,
                        "objective": obj,
                        "iterations": int(self.iterations),
                        "rebuilds": int(self.rebuilds),
                        "B_idx": list(self.B_idx),
                    }
                if status == "unbounded":
                    return {
                        "status": "unbounded",
                        "entering": int(p) if p is not None else None,
                        "iterations": int(self.iterations),
                        "rebuilds": int(self.rebuilds),
                    }
            # reached max iters
            return {"status": "max_iters", "iterations": int(self.iterations), "rebuilds": int(self.rebuilds)}
        except Exception as e:
            return {"status": "error", "error": str(e), "iterations": int(self.iterations), "rebuilds": int(self.rebuilds)}
