from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import tree_util

from ..core.ops import TrialOps
from ..core.system import System


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PtuccsdThoulessTrial:
    """
    Unrestricted first-order PT-UCCSD trial with T1 absorbed into the reference.

    ``mo_t_a`` is the alpha occupied Thouless determinant in the alpha MO basis.
    ``mo_t_b`` is the beta occupied Thouless determinant in the beta MO basis.
    ``mo_coeff_b`` rotates alpha-basis walker orbitals into the beta MO basis.
    The doubles amplitudes are raw UCCSD amplitudes in internal ``(i,a,j,b)``
    layout; no disconnected ``T1*T1`` terms are included.
    """

    mo_t_a: jax.Array
    mo_t_b: jax.Array
    mo_coeff_b: jax.Array
    t2aa: jax.Array
    t2ab: jax.Array
    t2bb: jax.Array

    def __post_init__(self) -> None:
        if not hasattr(self.mo_t_a, "ndim"):
            return
        if self.mo_t_a.ndim != 2 or self.mo_t_b.ndim != 2:
            raise ValueError("PtuccsdThoulessTrial.mo_t_a/mo_t_b must be rank 2.")
        if self.t2aa.ndim != 4 or self.t2ab.ndim != 4 or self.t2bb.ndim != 4:
            raise ValueError("PtuccsdThoulessTrial.t2aa/t2ab/t2bb must be rank 4.")

        noa, nva = self.t2aa.shape[:2]
        nob, nvb = self.t2bb.shape[:2]
        norb = noa + nva
        if self.t2aa.shape != (noa, nva, noa, nva):
            raise ValueError(
                "PtuccsdThoulessTrial.t2aa must have shape "
                f"(nocc_a, nvir_a, nocc_a, nvir_a); got {self.t2aa.shape}."
            )
        if self.t2ab.shape != (noa, nva, nob, nvb):
            raise ValueError(
                "PtuccsdThoulessTrial.t2ab must have shape "
                f"(nocc_a, nvir_a, nocc_b, nvir_b); got {self.t2ab.shape}."
            )
        if self.t2bb.shape != (nob, nvb, nob, nvb):
            raise ValueError(
                "PtuccsdThoulessTrial.t2bb must have shape "
                f"(nocc_b, nvir_b, nocc_b, nvir_b); got {self.t2bb.shape}."
            )
        if nob + nvb != norb:
            raise ValueError("Alpha and beta orbital spaces must have the same size.")
        if self.mo_t_a.shape != (norb, noa):
            raise ValueError(
                "PtuccsdThoulessTrial.mo_t_a must have shape "
                f"{(norb, noa)}; got {self.mo_t_a.shape}."
            )
        if self.mo_t_b.shape != (norb, nob):
            raise ValueError(
                "PtuccsdThoulessTrial.mo_t_b must have shape "
                f"{(norb, nob)}; got {self.mo_t_b.shape}."
            )
        if self.mo_coeff_b.shape != (norb, norb):
            raise ValueError(
                "PtuccsdThoulessTrial.mo_coeff_b must be the full beta orbital "
                f"rotation with shape {(norb, norb)}; got {self.mo_coeff_b.shape}."
            )

    @property
    def norb(self) -> int:
        return int(self.mo_t_a.shape[0])

    @property
    def nocc(self) -> tuple[int, int]:
        return (int(self.mo_t_a.shape[1]), int(self.mo_t_b.shape[1]))

    @property
    def nvir(self) -> tuple[int, int]:
        noa, nob = self.nocc
        norb = self.norb
        return (int(norb - noa), int(norb - nob))

    def tree_flatten(self):
        return (
            self.mo_t_a,
            self.mo_t_b,
            self.mo_coeff_b,
            self.t2aa,
            self.t2ab,
            self.t2bb,
        ), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        mo_t_a, mo_t_b, mo_coeff_b, t2aa, t2ab, t2bb = children
        return cls(
            mo_t_a=mo_t_a,
            mo_t_b=mo_t_b,
            mo_coeff_b=mo_coeff_b,
            t2aa=t2aa,
            t2ab=t2ab,
            t2bb=t2bb,
        )


def thouless_mo_from_t1(t1: jax.Array) -> jax.Array:
    """Return occupied orbitals for ``exp(T1)|HF>`` in that spin's MO basis."""
    nocc, _ = t1.shape
    eye = jnp.eye(nocc, dtype=t1.dtype)
    return jnp.vstack([eye, t1.T])


def _t2_from_pyscf_layout(t2: jax.Array) -> jax.Array:
    return jnp.asarray(t2).transpose(0, 2, 1, 3)


def _projector(c: jax.Array) -> jax.Array:
    metric = c.conj().T @ c
    return c @ jnp.linalg.solve(metric, c.conj().T)


def get_rdm1(trial_data: PtuccsdThoulessTrial) -> jax.Array:
    dm_a = _projector(trial_data.mo_t_a)
    beta_occ_alpha_basis = trial_data.mo_coeff_b @ trial_data.mo_t_b
    dm_b = _projector(beta_occ_alpha_basis)
    return jnp.stack([dm_a, dm_b], axis=0)


def _half_green_from_overlap_matrix(w: jax.Array, overlap_mat: jax.Array) -> jax.Array:
    return jnp.linalg.solve(overlap_mat.T, w.T)


def _spin_green(walker_spin: jax.Array, mo_t: jax.Array) -> jax.Array:
    overlap_mat = mo_t.conj().T @ walker_spin
    half_green = _half_green_from_overlap_matrix(walker_spin, overlap_mat)
    return mo_t.conj() @ half_green


def greens_unrestricted(
    walker: tuple[jax.Array, jax.Array], trial_data: PtuccsdThoulessTrial
) -> tuple[jax.Array, jax.Array]:
    wa, wb = walker
    wb_beta = trial_data.mo_coeff_b.conj().T @ wb
    green_a = _spin_green(wa, trial_data.mo_t_a)
    green_b = _spin_green(wb_beta, trial_data.mo_t_b)
    return green_a, green_b


def greenp_from_green(green: jax.Array, nocc: int) -> jax.Array:
    return (green - jnp.eye(green.shape[0], dtype=green.dtype))[:, nocc:]


def theta_t2_from_greens(
    green_a: jax.Array,
    green_b: jax.Array,
    trial_data: PtuccsdThoulessTrial,
) -> jax.Array:
    noa, nob = trial_data.nocc
    ga = green_a[:noa, noa:]
    gb = green_b[:nob, nob:]
    theta2aa = 0.5 * jnp.einsum("iajb,ia,jb->", trial_data.t2aa, ga, ga, optimize="optimal")
    theta2ab = jnp.einsum("iajb,ia,jb->", trial_data.t2ab, ga, gb, optimize="optimal")
    theta2bb = 0.5 * jnp.einsum("iajb,ia,jb->", trial_data.t2bb, gb, gb, optimize="optimal")
    return theta2aa + theta2ab + theta2bb


def theta_t2_u(walker: tuple[jax.Array, jax.Array], trial_data: PtuccsdThoulessTrial) -> jax.Array:
    green_a, green_b = greens_unrestricted(walker, trial_data)
    return theta_t2_from_greens(green_a, green_b, trial_data)


def reference_overlap_u(
    walker: tuple[jax.Array, jax.Array], trial_data: PtuccsdThoulessTrial
) -> jax.Array:
    wa, wb = walker
    wb_beta = trial_data.mo_coeff_b.conj().T @ wb
    overlap_a = trial_data.mo_t_a.conj().T @ wa
    overlap_b = trial_data.mo_t_b.conj().T @ wb_beta
    return jnp.linalg.det(overlap_a) * jnp.linalg.det(overlap_b)


def overlap_u(walker: tuple[jax.Array, jax.Array], trial_data: PtuccsdThoulessTrial) -> jax.Array:
    return reference_overlap_u(walker, trial_data) * jnp.exp(theta_t2_u(walker, trial_data))


def overlap_r(walker: jax.Array, trial_data: PtuccsdThoulessTrial) -> jax.Array:
    noa, nob = trial_data.nocc
    return overlap_u((walker[:, :noa], walker[:, :nob]), trial_data)


def make_ptuccsd_thouless_trial_ops(sys: System) -> TrialOps:
    wk = sys.walker_kind.lower()
    if wk == "restricted":
        overlap_fn = overlap_r
    elif wk == "unrestricted":
        overlap_fn = overlap_u
    else:
        raise ValueError(
            "PT-UCCSD Thouless trial currently supports restricted/unrestricted "
            f"walkers, got: {sys.walker_kind}"
        )
    return TrialOps(overlap=overlap_fn, get_rdm1=get_rdm1)


def make_ptuccsd_thouless_trial_data(data: dict, sys: System) -> PtuccsdThoulessTrial:
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
        raise ValueError(f"Unknown PT-UCCSD Thouless t2_layout: {layout!r}")

    if "mo_t_a" in data:
        mo_t_a = jnp.asarray(data["mo_t_a"])
    else:
        mo_t_a = thouless_mo_from_t1(jnp.asarray(data["t1a"]))

    if "mo_t_b" in data:
        mo_t_b = jnp.asarray(data["mo_t_b"])
    else:
        mo_t_b = thouless_mo_from_t1(jnp.asarray(data["t1b"]))

    mo_coeff_b_raw = data.get("mo_coeff_b", data.get("mo_b"))
    if mo_coeff_b_raw is None:
        raise KeyError("PT-UCCSD Thouless trial data requires 'mo_coeff_b' or 'mo_b'.")
    mo_coeff_b = jnp.asarray(mo_coeff_b_raw)

    return PtuccsdThoulessTrial(
        mo_t_a=mo_t_a,
        mo_t_b=mo_t_b,
        mo_coeff_b=mo_coeff_b,
        t2aa=t2aa,
        t2ab=t2ab,
        t2bb=t2bb,
    )


__all__ = [
    "PtuccsdThoulessTrial",
    "get_rdm1",
    "greenp_from_green",
    "greens_unrestricted",
    "make_ptuccsd_thouless_trial_data",
    "make_ptuccsd_thouless_trial_ops",
    "overlap_r",
    "overlap_u",
    "reference_overlap_u",
    "theta_t2_from_greens",
    "theta_t2_u",
    "thouless_mo_from_t1",
]
