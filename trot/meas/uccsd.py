from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import lax, tree_util

from ..core.ops import MeasOps, k_energy, k_force_bias
from ..core.system import System
from ..ham.chol import HamChol
from ..trial.uccsd import HsUccsdTrial, overlap_r, overlap_u


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class HsUccsdMeasCtx:
    def tree_flatten(self):
        return (), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls()


def build_hsuccsd_meas_ctx(ham_data: HamChol, trial_data: HsUccsdTrial) -> HsUccsdMeasCtx:
    del trial_data
    if ham_data.basis != "restricted":
        raise ValueError("HS-UCCSD MeasOps currently assumes HamChol.basis == 'restricted'.")
    return HsUccsdMeasCtx()


def _overlap_and_green_spin(walker_spin: jax.Array, det_coeff: jax.Array):
    ovlp_mat = det_coeff.conj().T @ walker_spin
    half_green = jnp.linalg.solve(ovlp_mat.T, walker_spin.T)
    green = det_coeff.conj() @ half_green
    overlap = jnp.linalg.det(ovlp_mat)
    return overlap, green


def _weights_and_greens(walker: tuple[jax.Array, jax.Array], trial_data: HsUccsdTrial):
    wa, wb = walker
    overlaps_a, greens_a = jax.vmap(_overlap_and_green_spin, in_axes=(None, 0))(
        wa, trial_data.det_coeffs_a
    )
    overlaps_b, greens_b = jax.vmap(_overlap_and_green_spin, in_axes=(None, 0))(
        wb, trial_data.det_coeffs_b
    )
    weights = trial_data.ci_coeffs * overlaps_a * overlaps_b
    denom = jnp.sum(weights)
    return weights, denom, greens_a, greens_b


def _energy_from_greens_unrestricted(
    green_a: jax.Array, green_b: jax.Array, ham_data: HamChol
) -> jax.Array:
    e1 = jnp.einsum("ij,ij->", ham_data.h1, green_a + green_b, optimize="optimal")

    zero = jnp.array(0.0, dtype=jnp.result_type(green_a, green_b, ham_data.chol, ham_data.h1))

    def scan_chol(e2_acc: jax.Array, chol_i: jax.Array):
        lg_a = jnp.einsum("ij,ij->", chol_i, green_a, optimize="optimal")
        lg_b = jnp.einsum("ij,ij->", chol_i, green_b, optimize="optimal")
        gl_a = jnp.einsum("pr,qr->pq", green_a, chol_i, optimize="optimal")
        gl_b = jnp.einsum("pr,qr->pq", green_b, chol_i, optimize="optimal")
        exch_a = jnp.einsum("pq,qp->", gl_a, gl_a, optimize="optimal")
        exch_b = jnp.einsum("pq,qp->", gl_b, gl_b, optimize="optimal")
        e2_i = 0.5 * ((lg_a + lg_b) * (lg_a + lg_b) - exch_a - exch_b)
        return e2_acc + e2_i, None

    e2, _ = lax.scan(scan_chol, zero, ham_data.chol)
    return ham_data.h0 + e1 + e2


def force_bias_hsuccsd_uw_rh(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamChol,
    meas_ctx: HsUccsdMeasCtx,
    trial_data: HsUccsdTrial,
) -> jax.Array:
    del meas_ctx
    weights, denom, greens_a, greens_b = _weights_and_greens(walker, trial_data)
    fb_det = jnp.einsum("gij,sij->sg", ham_data.chol, greens_a + greens_b, optimize="optimal")
    numerator = jnp.einsum("s,sg->g", weights, fb_det, optimize="optimal")
    return numerator / denom


def force_bias_hsuccsd_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: HsUccsdMeasCtx,
    trial_data: HsUccsdTrial,
) -> jax.Array:
    noa, nob = trial_data.nocc
    return force_bias_hsuccsd_uw_rh(
        (walker[:, :noa], walker[:, :nob]), ham_data, meas_ctx, trial_data
    )


def energy_hsuccsd_uw_rh(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamChol,
    meas_ctx: HsUccsdMeasCtx,
    trial_data: HsUccsdTrial,
) -> jax.Array:
    del meas_ctx
    weights, denom, greens_a, greens_b = _weights_and_greens(walker, trial_data)
    energies = jax.vmap(
        lambda ga, gb: _energy_from_greens_unrestricted(ga, gb, ham_data), in_axes=(0, 0)
    )(greens_a, greens_b)
    return jnp.sum(weights * energies) / denom


def energy_hsuccsd_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: HsUccsdMeasCtx,
    trial_data: HsUccsdTrial,
) -> jax.Array:
    noa, nob = trial_data.nocc
    return energy_hsuccsd_uw_rh((walker[:, :noa], walker[:, :nob]), ham_data, meas_ctx, trial_data)


def make_hsuccsd_meas_ops(sys: System) -> MeasOps:
    wk = sys.walker_kind.lower()
    if wk == "restricted":
        overlap_fn = overlap_r
        kernels = {k_force_bias: force_bias_hsuccsd_rw_rh, k_energy: energy_hsuccsd_rw_rh}
    elif wk == "unrestricted":
        overlap_fn = overlap_u
        kernels = {k_force_bias: force_bias_hsuccsd_uw_rh, k_energy: energy_hsuccsd_uw_rh}
    else:
        raise ValueError(
            f"HS-UCCSD measurements support restricted/unrestricted walkers, got: {sys.walker_kind}"
        )
    return MeasOps(
        overlap=overlap_fn,
        build_meas_ctx=build_hsuccsd_meas_ctx,
        kernels=kernels,
    )

