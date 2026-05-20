from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import lax, tree_util

from ..core.ops import MeasOps, k_energy, k_force_bias
from ..core.system import System
from ..ham.chol import HamChol
from ..trial.ccsd import HsCcsdTrial, overlap_hs_r


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class HsCcsdMeasCtx:
    def tree_flatten(self):
        return (), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls()


def build_hsccsd_meas_ctx(ham_data: HamChol, trial_data: HsCcsdTrial) -> HsCcsdMeasCtx:
    del trial_data
    if ham_data.basis != "restricted":
        raise ValueError("HS-CCSD MeasOps currently assumes HamChol.basis == 'restricted'.")
    return HsCcsdMeasCtx()


def _overlap_and_green_restricted(walker: jax.Array, det_coeff: jax.Array):
    ovlp_mat = det_coeff.conj().T @ walker
    half_green = jnp.linalg.solve(ovlp_mat.T, walker.T)
    green = det_coeff.conj() @ half_green
    overlap = jnp.linalg.det(ovlp_mat) ** 2
    return overlap, green


def _weights_and_greens(walker: jax.Array, trial_data: HsCcsdTrial):
    overlaps, greens = jax.vmap(_overlap_and_green_restricted, in_axes=(None, 0))(
        walker, trial_data.det_coeffs
    )
    weights = trial_data.ci_coeffs * overlaps
    denom = jnp.sum(weights)
    return weights, denom, greens


def _energy_from_green_restricted(green: jax.Array, ham_data: HamChol) -> jax.Array:
    e1 = 2.0 * jnp.einsum("ij,ij->", ham_data.h1, green, optimize="optimal")

    zero = jnp.array(0.0, dtype=jnp.result_type(green, ham_data.chol, ham_data.h1))

    def scan_chol(e2_acc: jax.Array, chol_i: jax.Array):
        lg_i = jnp.einsum("ij,ij->", chol_i, green, optimize="optimal")
        gl_i = jnp.einsum("pr,qr->pq", green, chol_i, optimize="optimal")
        e2_i = 2.0 * lg_i * lg_i - jnp.einsum("pq,qp->", gl_i, gl_i, optimize="optimal")
        return e2_acc + e2_i, None

    e2, _ = lax.scan(scan_chol, zero, ham_data.chol)
    return ham_data.h0 + e1 + e2


def force_bias_hsccsd_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: HsCcsdMeasCtx,
    trial_data: HsCcsdTrial,
) -> jax.Array:
    del meas_ctx
    weights, denom, greens = _weights_and_greens(walker, trial_data)
    fb_det = 2.0 * jnp.einsum("gij,sij->sg", ham_data.chol, greens, optimize="optimal")
    numerator = jnp.einsum("s,sg->g", weights, fb_det, optimize="optimal")
    return numerator / denom


def energy_hsccsd_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: HsCcsdMeasCtx,
    trial_data: HsCcsdTrial,
) -> jax.Array:
    del meas_ctx
    weights, denom, greens = _weights_and_greens(walker, trial_data)
    energies = jax.vmap(lambda green: _energy_from_green_restricted(green, ham_data))(greens)
    return jnp.sum(weights * energies) / denom


def make_hsccsd_meas_ops(sys: System) -> MeasOps:
    if sys.nup != sys.ndn:
        raise ValueError("HS-CCSD measurements require nup == ndn.")
    if sys.walker_kind.lower() != "restricted":
        raise ValueError(
            f"HS-CCSD measurements currently support only restricted walkers, got: {sys.walker_kind}"
        )
    return MeasOps(
        overlap=overlap_hs_r,
        build_meas_ctx=build_hsccsd_meas_ctx,
        kernels={k_force_bias: force_bias_hsccsd_rw_rh, k_energy: energy_hsccsd_rw_rh},
    )

