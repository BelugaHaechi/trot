from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import tree_util

from ..core.ops import MeasOps, k_energy, k_force_bias
from ..core.system import System
from ..ham.chol import HamChol
from ..trial.ptuccsd import PtuccsdTrial, overlap_r, overlap_u

o_pt_components = "pt_components"


@dataclass(frozen=True)
class PtuccsdMeasCfg:
    memory_mode: str = "low"
    mixed_real_dtype: jnp.dtype = jnp.float64
    mixed_complex_dtype: jnp.dtype = jnp.complex128
    mixed_real_dtype_testing: jnp.dtype = jnp.float32
    mixed_complex_dtype_testing: jnp.dtype = jnp.complex64


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PtuccsdMeasCtx:
    h1_b: jax.Array
    chol_b: jax.Array
    rot_chol_a: jax.Array
    rot_chol_b: jax.Array
    l_t1_a: jax.Array
    l_t1_b: jax.Array
    cfg: PtuccsdMeasCfg

    def tree_flatten(self):
        children = (
            self.h1_b,
            self.chol_b,
            self.rot_chol_a,
            self.rot_chol_b,
            self.l_t1_a,
            self.l_t1_b,
        )
        return children, (self.cfg,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        (cfg,) = aux
        h1_b, chol_b, rot_chol_a, rot_chol_b, l_t1_a, l_t1_b = children
        return cls(
            h1_b=h1_b,
            chol_b=chol_b,
            rot_chol_a=rot_chol_a,
            rot_chol_b=rot_chol_b,
            l_t1_a=l_t1_a,
            l_t1_b=l_t1_b,
            cfg=cfg,
        )


def build_ptuccsd_meas_ctx(
    ham_data: HamChol,
    trial_data: PtuccsdTrial,
    cfg: PtuccsdMeasCfg = PtuccsdMeasCfg(),
) -> PtuccsdMeasCtx:
    if ham_data.basis != "restricted":
        raise ValueError("PT-UCCSD MeasOps currently assumes HamChol.basis == 'restricted'.")

    noa, nob = trial_data.nocc
    cb = trial_data.mo_coeff_b
    cbh = cb.conj().T
    h1_sym = 0.5 * (ham_data.h1 + ham_data.h1.T.conj())
    h1_b = cbh @ h1_sym @ cb
    chol_b = jnp.einsum("pi,gij,jq->gpq", cbh, ham_data.chol, cb, optimize="optimal")
    rot_chol_a = ham_data.chol[:, :noa, :]
    rot_chol_b = chol_b[:, :nob, :]
    l_t1_a = jnp.einsum(
        "git,pt->gip",
        ham_data.chol[:, :, noa:],
        trial_data.t1a,
        optimize="optimal",
    )
    l_t1_b = jnp.einsum(
        "git,pt->gip",
        chol_b[:, :, nob:],
        trial_data.t1b,
        optimize="optimal",
    )
    return PtuccsdMeasCtx(
        h1_b=h1_b,
        chol_b=chol_b,
        rot_chol_a=rot_chol_a,
        rot_chol_b=rot_chol_b,
        l_t1_a=l_t1_a,
        l_t1_b=l_t1_b,
        cfg=cfg,
    )


def _half_green_from_overlap_matrix(w: jax.Array, overlap_mat: jax.Array) -> jax.Array:
    return jnp.linalg.solve(overlap_mat.T, w.T)


def _green_blocks(
    walker: tuple[jax.Array, jax.Array],
    trial_data: PtuccsdTrial,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    wa, wb = walker
    noa, nob = trial_data.nocc
    nva, nvb = trial_data.nvir
    wb_beta = trial_data.mo_coeff_b.conj().T @ wb
    green_a = _half_green_from_overlap_matrix(wa, wa[:noa, :])
    green_b = _half_green_from_overlap_matrix(wb_beta, wb_beta[:nob, :])
    green_occ_a = green_a[:, noa:]
    green_occ_b = green_b[:, nob:]
    greenp_a = jnp.vstack((green_occ_a, -jnp.eye(nva, dtype=green_a.dtype)))
    greenp_b = jnp.vstack((green_occ_b, -jnp.eye(nvb, dtype=green_b.dtype)))
    return green_a, green_b, green_occ_a, green_occ_b, greenp_a, greenp_b


def _theta_terms(
    trial_data: PtuccsdTrial,
    green_occ_a: jax.Array,
    green_occ_b: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    t1a, t1b = trial_data.t1a, trial_data.t1b
    t2aa, t2ab, t2bb = trial_data.t2aa, trial_data.t2ab, trial_data.t2bb
    theta1a = jnp.einsum("pt,pt->", t1a, green_occ_a, optimize="optimal")
    theta1b = jnp.einsum("pt,pt->", t1b, green_occ_b, optimize="optimal")
    t2g_a_fb = jnp.einsum("ptqu,pt->qu", t2aa, green_occ_a, optimize="optimal")
    t2g_b_fb = jnp.einsum("ptqu,pt->qu", t2bb, green_occ_b, optimize="optimal")
    theta2a = 0.5 * jnp.einsum("qu,qu->", t2g_a_fb, green_occ_a, optimize="optimal")
    theta2b = 0.5 * jnp.einsum("qu,qu->", t2g_b_fb, green_occ_b, optimize="optimal")
    t2g_ab_a = jnp.einsum("ptqu,qu->pt", t2ab, green_occ_b, optimize="optimal")
    theta2ab = jnp.einsum("pt,pt->", t2g_ab_a, green_occ_a, optimize="optimal")
    theta1 = theta1a + theta1b
    theta2 = theta2a + theta2b + theta2ab
    return theta1, theta2, t2g_a_fb, t2g_b_fb, t2g_ab_a


def force_bias_kernel_uw_rh(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamChol,
    meas_ctx: PtuccsdMeasCtx,
    trial_data: PtuccsdTrial,
) -> jax.Array:
    green_a, green_b, green_occ_a, green_occ_b, greenp_a, greenp_b = _green_blocks(
        walker, trial_data
    )
    t1a, t1b = trial_data.t1a, trial_data.t1b
    t2ab = trial_data.t2ab
    cfg = meas_ctx.cfg

    chol_a = ham_data.chol
    chol_b = meas_ctx.chol_b
    lg_a = jnp.einsum("gpj,pj->g", meas_ctx.rot_chol_a, green_a, optimize="optimal")
    lg_b = jnp.einsum("gpj,pj->g", meas_ctx.rot_chol_b, green_b, optimize="optimal")
    f0 = lg_a + lg_b

    theta1, theta2, t2g_a, t2g_b, t2g_ab_a = _theta_terms(trial_data, green_occ_a, green_occ_b)
    theta = theta1 + theta2

    t1gp_a = jnp.einsum("pt,it->pi", t1a, greenp_a, optimize="optimal")
    t1gp_b = jnp.einsum("pt,it->pi", t1b, greenp_b, optimize="optimal")
    gt1gp_a = jnp.einsum("pj,pi->ij", green_a, t1gp_a, optimize="optimal")
    gt1gp_b = jnp.einsum("pj,pi->ij", green_b, t1gp_b, optimize="optimal")
    fb_1_1 = theta1 * f0
    fb_1_2 = -jnp.einsum(
        "gij,ij->g",
        chol_a.astype(cfg.mixed_real_dtype),
        gt1gp_a.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    ) - jnp.einsum(
        "gij,ij->g",
        chol_b.astype(cfg.mixed_real_dtype),
        gt1gp_b.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )

    t2g_ab_b = jnp.einsum("ptqu,pt->qu", t2ab, green_occ_a, optimize="optimal")
    t2_green_a = (greenp_a @ (t2g_a + t2g_ab_a).T) @ green_a
    t2_green_b = (greenp_b @ (t2g_b + t2g_ab_b).T) @ green_b
    fb_2_1 = theta2 * f0
    fb_2_2 = -jnp.einsum(
        "gij,ij->g",
        chol_a.astype(cfg.mixed_real_dtype),
        t2_green_a.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    ) - jnp.einsum(
        "gij,ij->g",
        chol_b.astype(cfg.mixed_real_dtype),
        t2_green_b.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )

    return f0 + fb_1_1 + fb_1_2 + fb_2_1 + fb_2_2 - f0 * theta


def force_bias_kernel_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtuccsdMeasCtx,
    trial_data: PtuccsdTrial,
) -> jax.Array:
    noa, nob = trial_data.nocc
    return force_bias_kernel_uw_rh(
        (walker[:, :noa], walker[:, :nob]), ham_data, meas_ctx, trial_data
    )


def _energy_components_uw_rh(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamChol,
    meas_ctx: PtuccsdMeasCtx,
    trial_data: PtuccsdTrial,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    green_a, green_b, green_occ_a, green_occ_b, greenp_a, greenp_b = _green_blocks(
        walker, trial_data
    )
    t1a, t1b = trial_data.t1a, trial_data.t1b
    t2aa, t2ab, t2bb = trial_data.t2aa, trial_data.t2ab, trial_data.t2bb
    noa, nob = trial_data.nocc
    cfg = meas_ctx.cfg

    h1_a = 0.5 * (ham_data.h1 + ham_data.h1.T.conj())
    h1_b = meas_ctx.h1_b
    chol_a = ham_data.chol
    chol_b = meas_ctx.chol_b
    rot_chol_a = meas_ctx.rot_chol_a
    rot_chol_b = meas_ctx.rot_chol_b

    hg_a = jnp.einsum("pj,pj->", h1_a[:noa, :], green_a, optimize="optimal")
    hg_b = jnp.einsum("pj,pj->", h1_b[:nob, :], green_b, optimize="optimal")
    hg = hg_a + hg_b
    e1_0 = hg

    t1g_a = jnp.einsum("pt,pt->", t1a, green_occ_a, optimize="optimal")
    t1g_b = jnp.einsum("pt,pt->", t1b, green_occ_b, optimize="optimal")
    theta1 = t1g_a + t1g_b
    gpt1a = greenp_a @ t1a.T
    gpt1b = greenp_b @ t1b.T
    t1_green_a = gpt1a @ green_a
    t1_green_b = gpt1b @ green_b
    e1_1_1 = theta1 * hg
    e1_1_2 = -(
        jnp.einsum("ij,ij->", h1_a, t1_green_a, optimize="optimal")
        + jnp.einsum("ij,ij->", h1_b, t1_green_b, optimize="optimal")
    )
    e1_1 = e1_1_1 + e1_1_2

    t2g_a = 0.25 * jnp.einsum(
        "ptqu,pt->qu",
        t2aa.astype(cfg.mixed_real_dtype),
        green_occ_a.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    t2g_b = 0.25 * jnp.einsum(
        "ptqu,pt->qu",
        t2bb.astype(cfg.mixed_real_dtype),
        green_occ_b.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    t2g_ab_a = jnp.einsum(
        "ptqu,qu->pt",
        t2ab.astype(cfg.mixed_real_dtype),
        green_occ_b.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    t2g_ab_b = jnp.einsum(
        "ptqu,pt->qu",
        t2ab.astype(cfg.mixed_real_dtype),
        green_occ_a.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    theta2a = jnp.einsum("qu,qu->", t2g_a, green_occ_a, optimize="optimal")
    theta2b = jnp.einsum("qu,qu->", t2g_b, green_occ_b, optimize="optimal")
    theta2ab = jnp.einsum("pt,pt->", t2g_ab_a, green_occ_a, optimize="optimal")
    theta2 = 2.0 * (theta2a + theta2b) + theta2ab
    theta = theta1 + theta2

    t2_green_a = (greenp_a @ t2g_a.T) @ green_a
    t2_green_ab_a = (greenp_a @ t2g_ab_a.T) @ green_a
    t2_green_b = (greenp_b @ t2g_b.T) @ green_b
    t2_green_ab_b = (greenp_b @ t2g_ab_b.T) @ green_b
    e1_2_1 = hg * theta2
    e1_2_2_a = -jnp.einsum("ij,ij->", h1_a, 4.0 * t2_green_a + t2_green_ab_a, optimize="optimal")
    e1_2_2_b = -jnp.einsum("ij,ij->", h1_b, 4.0 * t2_green_b + t2_green_ab_b, optimize="optimal")
    e1_2 = e1_2_1 + e1_2_2_a + e1_2_2_b

    lg_a = jnp.einsum("gpj,pj->g", rot_chol_a, green_a, optimize="optimal")
    lg_b = jnp.einsum("gpj,pj->g", rot_chol_b, green_b, optimize="optimal")
    lg = lg_a + lg_b
    e2_0_1 = 0.5 * (lg @ lg)
    lg1_a = jnp.einsum("gpj,qj->gpq", rot_chol_a, green_a, optimize="optimal")
    lg1_b = jnp.einsum("gpj,qj->gpq", rot_chol_b, green_b, optimize="optimal")
    e2_0_2 = -0.5 * (
        jnp.sum(lg1_a * jnp.swapaxes(lg1_a, -1, -2)) + jnp.sum(lg1_b * jnp.swapaxes(lg1_b, -1, -2))
    )
    e2_0 = e2_0_1 + e2_0_2
    electronic_0 = e1_0 + e2_0

    e2_1_1 = e2_0 * theta1
    lt1g_a = jnp.einsum(
        "gij,ij->g",
        chol_a.astype(cfg.mixed_real_dtype),
        t1_green_a.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    lt1g_b = jnp.einsum(
        "gij,ij->g",
        chol_b.astype(cfg.mixed_real_dtype),
        t1_green_b.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    e2_1_2 = -((lt1g_a + lt1g_b) @ lg)
    t1g1_a = t1a @ green_occ_a.T
    t1g1_b = t1b @ green_occ_b.T
    e2_1_3_1 = jnp.einsum("gpq,gqr,rp->", lg1_a, lg1_a, t1g1_a, optimize="optimal")
    e2_1_3_1 += jnp.einsum("gpq,gqr,rp->", lg1_b, lg1_b, t1g1_b, optimize="optimal")
    lt1g_mat_a = jnp.einsum("gip,qi->gpq", meas_ctx.l_t1_a, green_a, optimize="optimal")
    lt1g_mat_b = jnp.einsum("gip,qi->gpq", meas_ctx.l_t1_b, green_b, optimize="optimal")
    e2_1_3_2 = -jnp.einsum("gpq,gqp->", lt1g_mat_a, lg1_a, optimize="optimal")
    e2_1_3_2 -= jnp.einsum("gpq,gqp->", lt1g_mat_b, lg1_b, optimize="optimal")
    e2_1 = e2_1_1 + e2_1_2 + e2_1_3_1 + e2_1_3_2

    e2_2_1 = e2_0 * theta2
    lt2g_a = jnp.einsum(
        "gij,ij->g",
        chol_a.astype(cfg.mixed_real_dtype),
        (8.0 * t2_green_a + 2.0 * t2_green_ab_a).astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    lt2g_b = jnp.einsum(
        "gij,ij->g",
        chol_b.astype(cfg.mixed_real_dtype),
        (8.0 * t2_green_b + 2.0 * t2_green_ab_b).astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    e2_2_2_1 = -0.5 * ((lt2g_a + lt2g_b) @ lg)

    def scan_over_chol(carry, x):
        e222_acc, e23_acc = carry
        chol_a_i, rot_chol_a_i, chol_b_i, rot_chol_b_i = x
        gl_a_i = jnp.einsum("pj,ji->pi", green_a, chol_a_i, optimize="optimal")
        gl_b_i = jnp.einsum("pj,ji->pi", green_b, chol_b_i, optimize="optimal")
        lt2_green_a_i = jnp.einsum(
            "pi,ji->pj",
            rot_chol_a_i,
            8.0 * t2_green_a + 2.0 * t2_green_ab_a,
            optimize="optimal",
        )
        lt2_green_b_i = jnp.einsum(
            "pi,ji->pj",
            rot_chol_b_i,
            8.0 * t2_green_b + 2.0 * t2_green_ab_b,
            optimize="optimal",
        )
        e222_acc += 0.5 * (
            jnp.einsum("pi,pi->", gl_a_i, lt2_green_a_i, optimize="optimal")
            + jnp.einsum("pi,pi->", gl_b_i, lt2_green_b_i, optimize="optimal")
        )
        glgp_a_i = jnp.einsum("pi,it->pt", gl_a_i, greenp_a, optimize="optimal").astype(
            cfg.mixed_complex_dtype_testing
        )
        glgp_b_i = jnp.einsum("pi,it->pt", gl_b_i, greenp_b, optimize="optimal").astype(
            cfg.mixed_complex_dtype_testing
        )
        l2t2_a = 0.5 * jnp.einsum(
            "pt,qu,ptqu->",
            glgp_a_i,
            glgp_a_i,
            t2aa.astype(cfg.mixed_real_dtype_testing),
            optimize="optimal",
        )
        l2t2_b = 0.5 * jnp.einsum(
            "pt,qu,ptqu->",
            glgp_b_i,
            glgp_b_i,
            t2bb.astype(cfg.mixed_real_dtype_testing),
            optimize="optimal",
        )
        l2t2_ab = jnp.einsum(
            "pt,qu,ptqu->",
            glgp_a_i,
            glgp_b_i,
            t2ab.astype(cfg.mixed_real_dtype_testing),
            optimize="optimal",
        )
        e23_acc += l2t2_a + l2t2_b + l2t2_ab
        return (e222_acc, e23_acc), None

    zero = jnp.array(0.0, dtype=cfg.mixed_complex_dtype)
    (e2_2_2_2, e2_2_3), _ = jax.lax.scan(
        scan_over_chol,
        (zero, zero),
        (chol_a, rot_chol_a, chol_b, rot_chol_b),
    )
    e2_2 = e2_2_1 + e2_2_2_1 + e2_2_2_2 + e2_2_3

    h_t = e1_1 + e1_2 + e2_1 + e2_2
    return theta, electronic_0, h_t


def energy_components_uw_rh(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamChol,
    meas_ctx: PtuccsdMeasCtx,
    trial_data: PtuccsdTrial,
) -> jax.Array:
    theta, electronic_0, h_t = _energy_components_uw_rh(walker, ham_data, meas_ctx, trial_data)
    guide_reweight = jnp.exp(-theta)
    return jnp.stack(
        [
            guide_reweight,
            guide_reweight * theta,
            guide_reweight * electronic_0,
            guide_reweight * h_t,
        ]
    )


def energy_components_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtuccsdMeasCtx,
    trial_data: PtuccsdTrial,
) -> jax.Array:
    noa, nob = trial_data.nocc
    return energy_components_uw_rh(
        (walker[:, :noa], walker[:, :nob]), ham_data, meas_ctx, trial_data
    )


def energy_kernel_uw_rh(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamChol,
    meas_ctx: PtuccsdMeasCtx,
    trial_data: PtuccsdTrial,
) -> jax.Array:
    theta, electronic_0, h_t = _energy_components_uw_rh(walker, ham_data, meas_ctx, trial_data)
    return ham_data.h0 + electronic_0 + h_t - electronic_0 * theta


def energy_kernel_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtuccsdMeasCtx,
    trial_data: PtuccsdTrial,
) -> jax.Array:
    noa, nob = trial_data.nocc
    return energy_kernel_uw_rh((walker[:, :noa], walker[:, :nob]), ham_data, meas_ctx, trial_data)


def make_ptuccsd_meas_ops(
    sys: System,
    memory_mode: str = "low",
    mixed_precision: bool = False,
    testing: bool = False,
) -> MeasOps:
    cfg = PtuccsdMeasCfg(
        memory_mode=memory_mode,
        mixed_real_dtype=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed_precision else jnp.complex128,
        mixed_real_dtype_testing=jnp.float64 if testing else jnp.float32,
        mixed_complex_dtype_testing=jnp.complex128 if testing else jnp.complex64,
    )

    wk = sys.walker_kind.lower()
    if wk == "restricted":
        overlap_fn = overlap_r
        kernels = {k_force_bias: force_bias_kernel_rw_rh, k_energy: energy_kernel_rw_rh}
        observables = {o_pt_components: energy_components_rw_rh}
    elif wk == "unrestricted":
        overlap_fn = overlap_u
        kernels = {k_force_bias: force_bias_kernel_uw_rh, k_energy: energy_kernel_uw_rh}
        observables = {o_pt_components: energy_components_uw_rh}
    else:
        raise ValueError(
            f"PT-UCCSD measurements support restricted/unrestricted walkers, got: {sys.walker_kind}"
        )

    return MeasOps(
        overlap=overlap_fn,
        build_meas_ctx=lambda ham_data, trial_data: build_ptuccsd_meas_ctx(
            ham_data, trial_data, cfg
        ),
        kernels=kernels,
        observables=observables,
    )
