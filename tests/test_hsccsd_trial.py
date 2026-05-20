from trot import config

config.configure_once()

import jax
import jax.numpy as jnp

from trot.core.ops import k_energy, k_force_bias
from trot.core.system import System
from trot.ham.chol import HamChol
from trot.meas.ccsd import make_hsccsd_meas_ops
from trot.meas.rhf import make_rhf_meas_ops
from trot.meas.uccsd import make_hsuccsd_meas_ops
from trot.meas.uhf import make_uhf_meas_ops
from trot.trial.ccsd import HsCcsdTrial, overlap_hs_r
from trot.trial.rhf import RhfTrial, overlap_r as rhf_overlap_r
from trot.trial.uccsd import HsUccsdTrial, overlap_u as hsuccsd_overlap_u
from trot.trial.uhf import UhfTrial, overlap_u as uhf_overlap_u


def _complex_normal(key, shape):
    kr, ki = jax.random.split(key)
    return jax.random.normal(kr, shape) + 0.3j * jax.random.normal(ki, shape)


def _orth(key, norb, nocc):
    q, _ = jnp.linalg.qr(_complex_normal(key, (norb, nocc)))
    return q[:, :nocc]


def _ham(key, norb, nchol):
    kh, kl = jax.random.split(key)
    h1 = _complex_normal(kh, (norb, norb))
    h1 = 0.5 * (h1 + h1.conj().T)
    chol = _complex_normal(kl, (nchol, norb, norb))
    chol = 0.5 * (chol + jnp.swapaxes(chol.conj(), -1, -2))
    return HamChol(h0=jnp.asarray(0.7), h1=h1, chol=chol, basis="restricted")


def test_single_hsccsd_det_matches_rhf_kernels():
    keys = jax.random.split(jax.random.PRNGKey(11), 3)
    norb, nocc, nchol = 5, 2, 4
    det = _orth(keys[0], norb, nocc)
    walker = _orth(keys[1], norb, nocc)
    ham = _ham(keys[2], norb, nchol)

    sys = System(norb=norb, nelec=(nocc, nocc), walker_kind="restricted")
    hs_trial = HsCcsdTrial(det_coeffs=det[None], ci_coeffs=jnp.ones((1,), dtype=det.dtype))
    rhf_trial = RhfTrial(det)

    hs_meas = make_hsccsd_meas_ops(sys)
    rhf_meas = make_rhf_meas_ops(sys)
    hs_ctx = hs_meas.build_meas_ctx(ham, hs_trial)
    rhf_ctx = rhf_meas.build_meas_ctx(ham, rhf_trial)

    assert jnp.allclose(overlap_hs_r(walker, hs_trial), rhf_overlap_r(walker, rhf_trial))
    assert jnp.allclose(
        hs_meas.require_kernel(k_force_bias)(walker, ham, hs_ctx, hs_trial),
        rhf_meas.require_kernel(k_force_bias)(walker, ham, rhf_ctx, rhf_trial),
    )
    assert jnp.allclose(
        hs_meas.require_kernel(k_energy)(walker, ham, hs_ctx, hs_trial),
        rhf_meas.require_kernel(k_energy)(walker, ham, rhf_ctx, rhf_trial),
    )


def test_single_hsuccsd_det_matches_uhf_kernels():
    keys = jax.random.split(jax.random.PRNGKey(13), 5)
    norb, nup, ndn, nchol = 5, 2, 1, 4
    det_a = _orth(keys[0], norb, nup)
    det_b = _orth(keys[1], norb, ndn)
    walker = (_orth(keys[2], norb, nup), _orth(keys[3], norb, ndn))
    ham = _ham(keys[4], norb, nchol)

    sys = System(norb=norb, nelec=(nup, ndn), walker_kind="unrestricted")
    hs_trial = HsUccsdTrial(
        det_coeffs_a=det_a[None],
        det_coeffs_b=det_b[None],
        ci_coeffs=jnp.ones((1,), dtype=det_a.dtype),
    )
    uhf_trial = UhfTrial(det_a, det_b)

    hs_meas = make_hsuccsd_meas_ops(sys)
    uhf_meas = make_uhf_meas_ops(sys)
    hs_ctx = hs_meas.build_meas_ctx(ham, hs_trial)
    uhf_ctx = uhf_meas.build_meas_ctx(ham, uhf_trial)

    assert jnp.allclose(hsuccsd_overlap_u(walker, hs_trial), uhf_overlap_u(walker, uhf_trial))
    assert jnp.allclose(
        hs_meas.require_kernel(k_force_bias)(walker, ham, hs_ctx, hs_trial),
        uhf_meas.require_kernel(k_force_bias)(walker, ham, uhf_ctx, uhf_trial),
    )
    assert jnp.allclose(
        hs_meas.require_kernel(k_energy)(walker, ham, hs_ctx, hs_trial),
        uhf_meas.require_kernel(k_energy)(walker, ham, uhf_ctx, uhf_trial),
    )

