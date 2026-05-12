from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import tree_util

from ..core.ops import TrialOps
from ..core.system import System


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PtuccsdTrial:
    """
    Unrestricted first-order PT-UCCSD trial in the UHF MO basis.

    The Hamiltonian/walker basis follows the existing UHF convention: alpha
    orbitals define the one-particle basis, and ``mo_coeff_b`` rotates vectors
    from the alpha basis into the beta MO basis.  The cluster amplitudes are raw
    UCCSD amplitudes in internal ``(i, a, j, b)`` layout; no disconnected
    ``T1*T1`` terms are included.
    """

    mo_coeff_a: jax.Array
    mo_coeff_b: jax.Array
    t1a: jax.Array
    t1b: jax.Array
    t2aa: jax.Array
    t2ab: jax.Array
    t2bb: jax.Array

    def __post_init__(self) -> None:
        if not hasattr(self.t1a, "ndim"):
            return
        if self.t1a.ndim != 2 or self.t1b.ndim != 2:
            raise ValueError("PtuccsdTrial.t1a/t1b must be rank 2.")
        if self.t2aa.ndim != 4 or self.t2ab.ndim != 4 or self.t2bb.ndim != 4:
            raise ValueError("PtuccsdTrial.t2aa/t2ab/t2bb must be rank 4.")
        noa, nva = self.t1a.shape
        nob, nvb = self.t1b.shape
        if self.t2aa.shape != (noa, nva, noa, nva):
            raise ValueError(
                "PtuccsdTrial.t2aa must have shape (nocc_a, nvir_a, nocc_a, nvir_a); "
                f"got {self.t2aa.shape}."
            )
        if self.t2ab.shape != (noa, nva, nob, nvb):
            raise ValueError(
                "PtuccsdTrial.t2ab must have shape (nocc_a, nvir_a, nocc_b, nvir_b); "
                f"got {self.t2ab.shape}."
            )
        if self.t2bb.shape != (nob, nvb, nob, nvb):
            raise ValueError(
                "PtuccsdTrial.t2bb must have shape (nocc_b, nvir_b, nocc_b, nvir_b); "
                f"got {self.t2bb.shape}."
            )
        if self.mo_coeff_a.shape != (noa + nva, noa + nva):
            raise ValueError(
                "PtuccsdTrial.mo_coeff_a must be the full alpha-basis orbital matrix "
                f"with shape {(noa + nva, noa + nva)}; got {self.mo_coeff_a.shape}."
            )
        if self.mo_coeff_b.shape != (nob + nvb, nob + nvb):
            raise ValueError(
                "PtuccsdTrial.mo_coeff_b must be the full beta orbital rotation "
                f"with shape {(nob + nvb, nob + nvb)}; got {self.mo_coeff_b.shape}."
            )
        if self.mo_coeff_a.shape[0] != self.mo_coeff_b.shape[0]:
            raise ValueError("Alpha and beta orbital spaces must have the same size.")

    @property
    def norb(self) -> int:
        return int(self.mo_coeff_a.shape[0])

    @property
    def nocc(self) -> tuple[int, int]:
        return (int(self.t1a.shape[0]), int(self.t1b.shape[0]))

    @property
    def nvir(self) -> tuple[int, int]:
        return (int(self.t1a.shape[1]), int(self.t1b.shape[1]))

    def tree_flatten(self):
        return (
            self.mo_coeff_a,
            self.mo_coeff_b,
            self.t1a,
            self.t1b,
            self.t2aa,
            self.t2ab,
            self.t2bb,
        ), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        mo_coeff_a, mo_coeff_b, t1a, t1b, t2aa, t2ab, t2bb = children
        return cls(
            mo_coeff_a=mo_coeff_a,
            mo_coeff_b=mo_coeff_b,
            t1a=t1a,
            t1b=t1b,
            t2aa=t2aa,
            t2ab=t2ab,
            t2bb=t2bb,
        )


def _det(m: jax.Array) -> jax.Array:
    return jnp.linalg.det(m)


def _t2_from_pyscf_layout(t2: jax.Array) -> jax.Array:
    return jnp.asarray(t2).transpose(0, 2, 1, 3)


def get_rdm1(trial_data: PtuccsdTrial) -> jax.Array:
    norb = trial_data.norb
    noa, nob = trial_data.nocc
    occ_a = jnp.arange(norb) < noa
    dm_a = jnp.diag(occ_a)
    cb_occ = trial_data.mo_coeff_b[:, :nob]
    dm_b = cb_occ @ cb_occ.conj().T
    return jnp.stack([dm_a, dm_b], axis=0)


def _half_green_from_overlap_matrix(w: jax.Array, overlap_mat: jax.Array) -> jax.Array:
    return jnp.linalg.solve(overlap_mat.T, w.T)


def greens_unrestricted(
    walker: tuple[jax.Array, jax.Array], trial_data: PtuccsdTrial
) -> tuple[jax.Array, jax.Array]:
    wa, wb = walker
    noa, nob = trial_data.nocc
    wb_beta = trial_data.mo_coeff_b.conj().T @ wb
    green_a = _half_green_from_overlap_matrix(wa, wa[:noa, :])
    green_b = _half_green_from_overlap_matrix(wb_beta, wb_beta[:nob, :])
    return green_a, green_b


def theta_from_greens(
    green_a: jax.Array, green_b: jax.Array, trial_data: PtuccsdTrial
) -> jax.Array:
    noa, nob = trial_data.nocc
    ga = green_a[:, noa:]
    gb = green_b[:, nob:]
    theta1 = jnp.einsum("ia,ia->", trial_data.t1a, ga, optimize="optimal")
    theta1 += jnp.einsum("ia,ia->", trial_data.t1b, gb, optimize="optimal")
    theta2aa = 0.5 * jnp.einsum("iajb,ia,jb->", trial_data.t2aa, ga, ga, optimize="optimal")
    theta2ab = jnp.einsum("iajb,ia,jb->", trial_data.t2ab, ga, gb, optimize="optimal")
    theta2bb = 0.5 * jnp.einsum("iajb,ia,jb->", trial_data.t2bb, gb, gb, optimize="optimal")
    return theta1 + theta2aa + theta2ab + theta2bb


def theta_pt_u(walker: tuple[jax.Array, jax.Array], trial_data: PtuccsdTrial) -> jax.Array:
    green_a, green_b = greens_unrestricted(walker, trial_data)
    return theta_from_greens(green_a, green_b, trial_data)


def reference_overlap_u(walker: tuple[jax.Array, jax.Array], trial_data: PtuccsdTrial) -> jax.Array:
    wa, wb = walker
    noa, nob = trial_data.nocc
    wb_beta = trial_data.mo_coeff_b.conj().T @ wb
    return _det(wa[:noa, :]) * _det(wb_beta[:nob, :])


def overlap_u(walker: tuple[jax.Array, jax.Array], trial_data: PtuccsdTrial) -> jax.Array:
    return reference_overlap_u(walker, trial_data) * jnp.exp(theta_pt_u(walker, trial_data))


def overlap_r(walker: jax.Array, trial_data: PtuccsdTrial) -> jax.Array:
    noa, nob = trial_data.nocc
    return overlap_u((walker[:, :noa], walker[:, :nob]), trial_data)


def make_ptuccsd_trial_ops(sys: System) -> TrialOps:
    wk = sys.walker_kind.lower()
    if wk == "restricted":
        overlap_fn = overlap_r
    elif wk == "unrestricted":
        overlap_fn = overlap_u
    else:
        raise ValueError(
            f"PT-UCCSD trial currently supports restricted/unrestricted walkers, got: {sys.walker_kind}"
        )
    return TrialOps(overlap=overlap_fn, get_rdm1=get_rdm1)


def make_ptuccsd_trial_data(data: dict, sys: System) -> PtuccsdTrial:
    t1a = jnp.asarray(data["t1a"])
    t1b = jnp.asarray(data["t1b"])
    layout = str(data.get("t2_layout", "pyscf")).lower()
    if layout in {"pyscf", "ijab"}:
        t2aa = _t2_from_pyscf_layout(data["t2aa"])
        t2ab = _t2_from_pyscf_layout(data["t2ab"])
        t2bb = _t2_from_pyscf_layout(data["t2bb"])
    elif layout == "iajb":
        t2aa = jnp.asarray(data["t2aa"])
        t2ab = jnp.asarray(data["t2ab"])
        t2bb = jnp.asarray(data["t2bb"])
    else:
        raise ValueError(f"Unknown PT-UCCSD t2_layout: {layout!r}")

    mo_coeff_a = jnp.asarray(data.get("mo_coeff_a", data.get("mo_a", jnp.eye(sys.norb))))
    mo_coeff_b_raw = data.get("mo_coeff_b", data.get("mo_b"))
    if mo_coeff_b_raw is None:
        raise KeyError("PT-UCCSD trial data requires 'mo_coeff_b' or 'mo_b'.")
    mo_coeff_b = jnp.asarray(mo_coeff_b_raw)
    return PtuccsdTrial(
        mo_coeff_a=mo_coeff_a,
        mo_coeff_b=mo_coeff_b,
        t1a=t1a,
        t1b=t1b,
        t2aa=t2aa,
        t2ab=t2ab,
        t2bb=t2bb,
    )
