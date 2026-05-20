from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import tree_util
from numpy.typing import ArrayLike

from ..core.ops import TrialOps
from ..core.system import System


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class HsUccsdTrial:
    """
    Fixed-sample HS representation of an unrestricted CCSD trial.

    ``det_coeffs_a`` and ``det_coeffs_b`` store the sampled alpha/beta
    determinants in the AFQMC one-particle basis with shapes
    ``(n_dets, norb, nocc_a)`` and ``(n_dets, norb, nocc_b)``.
    """

    det_coeffs_a: jax.Array
    det_coeffs_b: jax.Array
    ci_coeffs: jax.Array

    def __post_init__(self) -> None:
        if not hasattr(self.det_coeffs_a, "ndim"):
            return
        if self.det_coeffs_a.ndim != 3 or self.det_coeffs_b.ndim != 3:
            raise ValueError("HsUccsdTrial determinant arrays must be rank 3.")
        if self.det_coeffs_a.shape[0] != self.det_coeffs_b.shape[0]:
            raise ValueError("Alpha and beta determinant sample counts must match.")
        if self.det_coeffs_a.shape[1] != self.det_coeffs_b.shape[1]:
            raise ValueError("Alpha and beta determinant orbital dimensions must match.")
        if self.ci_coeffs.ndim != 1:
            raise ValueError(
                f"HsUccsdTrial.ci_coeffs must be rank 1; got {self.ci_coeffs.shape}."
            )
        if self.ci_coeffs.shape[0] != self.det_coeffs_a.shape[0]:
            raise ValueError("HsUccsdTrial.ci_coeffs length must match determinant samples.")

    @property
    def ndets(self) -> int:
        return int(self.det_coeffs_a.shape[0])

    @property
    def norb(self) -> int:
        return int(self.det_coeffs_a.shape[1])

    @property
    def nocc(self) -> tuple[int, int]:
        return (int(self.det_coeffs_a.shape[2]), int(self.det_coeffs_b.shape[2]))

    def tree_flatten(self):
        return (self.det_coeffs_a, self.det_coeffs_b, self.ci_coeffs), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        det_coeffs_a, det_coeffs_b, ci_coeffs = children
        return cls(det_coeffs_a=det_coeffs_a, det_coeffs_b=det_coeffs_b, ci_coeffs=ci_coeffs)


def _projector(c: jax.Array) -> jax.Array:
    metric = c.conj().T @ c
    return c @ jnp.linalg.solve(metric, c.conj().T)


def get_rdm1(trial_data: HsUccsdTrial) -> jax.Array:
    weights = jnp.abs(trial_data.ci_coeffs)
    weights = weights / jnp.sum(weights)
    dm_a_s = jax.vmap(_projector)(trial_data.det_coeffs_a)
    dm_b_s = jax.vmap(_projector)(trial_data.det_coeffs_b)
    dm_a = jnp.einsum("s,sij->ij", weights, dm_a_s, optimize="optimal")
    dm_b = jnp.einsum("s,sij->ij", weights, dm_b_s, optimize="optimal")
    dm_a = 0.5 * (dm_a + dm_a.conj().T)
    dm_b = 0.5 * (dm_b + dm_b.conj().T)
    return jnp.stack([dm_a, dm_b], axis=0)


def det_overlaps_u(
    walker: tuple[jax.Array, jax.Array], trial_data: HsUccsdTrial
) -> jax.Array:
    wa, wb = walker
    mats_a = jnp.einsum(
        "sni,nj->sij", trial_data.det_coeffs_a.conj(), wa, optimize="optimal"
    )
    mats_b = jnp.einsum(
        "sni,nj->sij", trial_data.det_coeffs_b.conj(), wb, optimize="optimal"
    )
    return jax.vmap(jnp.linalg.det)(mats_a) * jax.vmap(jnp.linalg.det)(mats_b)


def overlap_u(walker: tuple[jax.Array, jax.Array], trial_data: HsUccsdTrial) -> jax.Array:
    return jnp.sum(trial_data.ci_coeffs * det_overlaps_u(walker, trial_data))


def overlap_r(walker: jax.Array, trial_data: HsUccsdTrial) -> jax.Array:
    noa, nob = trial_data.nocc
    return overlap_u((walker[:, :noa], walker[:, :nob]), trial_data)


def make_hsuccsd_trial_ops(sys: System) -> TrialOps:
    wk = sys.walker_kind.lower()
    if wk == "restricted":
        overlap_fn = overlap_r
    elif wk == "unrestricted":
        overlap_fn = overlap_u
    else:
        raise ValueError(
            f"HS-UCCSD trial currently supports restricted/unrestricted walkers, got: {sys.walker_kind}"
        )
    return TrialOps(overlap=overlap_fn, get_rdm1=get_rdm1)


def make_hsuccsd_trial_data_from_amplitudes(
    trial_coeff: tuple[ArrayLike, ArrayLike],
    t1: ArrayLike,
    t2: ArrayLike,
    *,
    seed: int = 0,
    n_samples: int = 100,
) -> HsUccsdTrial:
    hs_op = build_hs_op(t2)
    key = jax.random.PRNGKey(int(seed))
    det_coeffs_a, det_coeffs_b = init_walkers(trial_coeff, t1, hs_op, key, int(n_samples))
    ci_coeffs = jnp.full((int(n_samples),), 1.0 / float(n_samples), dtype=det_coeffs_a.dtype)
    return HsUccsdTrial(
        det_coeffs_a=det_coeffs_a,
        det_coeffs_b=det_coeffs_b,
        ci_coeffs=ci_coeffs,
    )


def make_hsuccsd_trial_data(data: dict, sys: System | None = None) -> HsUccsdTrial:
    del sys
    if "det_coeffs_a" in data and "det_coeffs_b" in data:
        det_coeffs_a = jnp.asarray(data["det_coeffs_a"])
        det_coeffs_b = jnp.asarray(data["det_coeffs_b"])
        ci_raw = data.get("ci_coeffs")
        if ci_raw is None:
            ci_coeffs = jnp.full(
                (det_coeffs_a.shape[0],),
                1.0 / float(det_coeffs_a.shape[0]),
                dtype=det_coeffs_a.dtype,
            )
        else:
            ci_coeffs = jnp.asarray(ci_raw, dtype=det_coeffs_a.dtype)
        return HsUccsdTrial(
            det_coeffs_a=det_coeffs_a,
            det_coeffs_b=det_coeffs_b,
            ci_coeffs=ci_coeffs,
        )

    trial_coeff_a = data.get("trial_coeff_a", data.get("mo_coeff_a", data.get("mo_a")))
    trial_coeff_b = data.get("trial_coeff_b", data.get("mo_coeff_b", data.get("mo_b")))
    if trial_coeff_a is None or trial_coeff_b is None:
        raise KeyError(
            "HS-UCCSD trial data requires sampled determinant arrays or alpha/beta trial coeffs."
        )
    return make_hsuccsd_trial_data_from_amplitudes(
        (trial_coeff_a, trial_coeff_b),
        (data["t1a"], data["t1b"]),
        (data["t2aa"], data["t2ab"], data["t2bb"]),
        seed=int(data.get("seed", 0)),
        n_samples=int(data.get("n_samples", 100)),
    )


def build_hs_op(t2: ArrayLike) -> tuple[jax.Array, jax.Array]:
    """
    Builds the Cholesky decomposition of UCCSD T2 amplitudes,
    T2 = LL^T.

    Input:
    t2: UCCSD T2 amplitudes

    Output:
    (La, Lb): alpha and beta parts of the Cholesky vectors
    """
    t2aa, t2ab, t2bb = t2

    t2aa = jnp.asarray(t2aa)
    t2ab = jnp.asarray(t2ab)
    t2bb = jnp.asarray(t2bb)

    nOa, nOb, nVa, nVb = t2ab.shape
    n = nOa + nVa
    assert n == nOb + nVb

    # Number of excitations
    nex_a = nOa * nVa
    nex_b = nOb * nVb

    assert t2aa.shape == (nOa, nOa, nVa, nVa)
    assert t2ab.shape == (nOa, nOb, nVa, nVb)
    assert t2bb.shape == (nOb, nOb, nVb, nVb)

    # t2(i,j,a,b) -> t2(ai,bj)
    t2aa = jnp.einsum("ijab->aibj", t2aa)
    t2ab = jnp.einsum("ijab->aibj", t2ab)
    t2bb = jnp.einsum("ijab->aibj", t2bb)

    t2aa = t2aa.reshape(nex_a, nex_a)
    t2ab = t2ab.reshape(nex_a, nex_b)
    t2bb = t2bb.reshape(nex_b, nex_b)

    # Symmetric t2 =
    # t2aa/2 t2ab
    # t2ab^T t2bb
    t2 = jnp.zeros((nex_a + nex_b, nex_a + nex_b))
    t2 = jax.lax.dynamic_update_slice(t2, 0.5 * t2aa, (0, 0))
    t2 = jax.lax.dynamic_update_slice(t2, t2ab.T, (nex_a, 0))
    t2 = jax.lax.dynamic_update_slice(t2, t2ab, (0, nex_a))
    t2 = jax.lax.dynamic_update_slice(t2, 0.5 * t2bb, (nex_a, nex_a))

    # t2 = LL^T
    e_val, e_vec = jnp.linalg.eigh(t2)
    L = e_vec @ jnp.diag(jnp.sqrt(e_val + 0.0j))
    assert abs(jnp.linalg.norm(t2 - L @ L.T)) < 1e-12

    # alpha/beta operators for HS
    # Summation on the left to have a list of operators
    La = jnp.array(L[:nex_a, :])
    Lb = jnp.array(L[nex_a:, :])
    La = La.T.reshape(nex_a + nex_b, nVa, nOa)
    Lb = Lb.T.reshape(nex_a + nex_b, nVb, nOb)

    return (La, Lb)


def init_walkers(
    trial_coeff: tuple[ArrayLike, ArrayLike],
    t1: ArrayLike,
    hs_op: tuple[jax.Array, jax.Array],
    subkey: jax.Array,
    n_w: int,
) -> tuple[jax.Array, jax.Array]:
    """
    Builds a stochastic representation of the UCCSD wavefunction using the Hubbard-Stratonovich transformation.

    Input:
    trial_coeff: mo coefficients in the alpha mo basis
    t1         : UCCSD T1 amplitudes
    hs_op      : alpha and beta Cholesky vectors from the UCCSD T2 amplitudes
    subkey     : PRNG key
    n_w        : number of walkers

    Output:
    (w_a, w_b): alpha and beta walkers
    """
    t1a, t1b = t1

    t1a = jnp.asarray(t1a)
    t1b = jnp.asarray(t1b)

    nOa, nVa = t1a.shape
    nOb, nVb = t1b.shape
    n = nOa + nVa
    assert n == nOb + nVb

    nex_a = nOa * nVa
    nex_b = nOb * nVb
    nex = nex_a + nex_b

    La, Lb = hs_op
    assert La.shape == (nex_a + nex_b, nVa, nOa)
    assert Lb.shape == (nex_a + nex_b, nVb, nOb)

    Ca, Cb = trial_coeff
    Ca = jnp.asarray(Ca)
    Cb = jnp.asarray(Cb)
    assert Ca.shape == (n, n)
    assert Cb.shape == (n, n)

    Ca_occ, Ca_vir = jnp.split(Ca, [nOa], axis=1)
    Cb_occ, Cb_vir = jnp.split(Cb, [nOb], axis=1)

    # e^T1
    e_t1a = t1a.T + 0.0j
    e_t1b = t1b.T + 0.0j

    ops_a = jnp.array([e_t1a] * n_w)
    ops_b = jnp.array([e_t1b] * n_w)

    fields = jax.random.normal(subkey, shape=(n_w, nex))

    # e^{T1+T2}
    ops_a = ops_a + jnp.einsum("wg,gai->wai", fields, La)
    ops_b = ops_b + jnp.einsum("wg,gai->wai", fields, Lb)

    # Initial walkers
    dm_a = Ca[:, :nOa] @ Ca[:, :nOa].conj().T
    dm_b = Cb[:, :nOb] @ Cb[:, :nOb].conj().T
    nos_a = jnp.linalg.eigh(dm_a)[1][:, ::-1][:, :nOa]
    nos_b = jnp.linalg.eigh(dm_b)[1][:, ::-1][:, :nOb]

    w_a = jnp.array([nos_a + 0.0j] * n_w)
    w_b = jnp.array([nos_b + 0.0j] * n_w)

    id_a = jnp.array([jnp.identity(n) + 0.0j] * n_w)
    id_b = jnp.array([jnp.identity(n) + 0.0j] * n_w)

    # e^{T1+T2} \ket{\phi}
    w_a = (id_a + jnp.einsum("pa,wai,iq -> wpq", Ca_vir, ops_a, Ca_occ.T)) @ w_a
    w_b = (id_b + jnp.einsum("pa,wai,iq -> wpq", Cb_vir, ops_b, Cb_occ.T)) @ w_b

    return (w_a, w_b)


def make_init_prop_state(trial_coeff: tuple[ArrayLike, ArrayLike], t1: ArrayLike, t2: ArrayLike):
    from jax.sharding import Mesh
    from trot import walkers as wk
    from trot.core.ops import MeasOps, TrialOps, k_energy
    from trot.core.system import System
    from trot.ham.chol import HamChol
    from trot.sharding import shard_prop_state
    from trot.prop.types import PropState, QmcParamsBase

    hs_op = build_hs_op(t2)

    def init_prop_state(
        *,
        sys: System,
        ham_data: HamChol,
        trial_ops: TrialOps,
        trial_data: Any,
        meas_ops: MeasOps,
        params: QmcParamsBase,
        initial_walkers: Any | None = None,
        initial_e_estimate: jax.Array | None = None,
        rdm1: jax.Array | None = None,
        mesh: Mesh | None = None,
    ) -> PropState:
        """
        Initialize AFQMC propagation state.
        """
        assert sys.walker_kind == "unrestricted"
        n_walkers = params.n_walkers
        seed = params.seed
        key = jax.random.PRNGKey(int(seed))
        weights = jnp.ones((n_walkers,))

        initial_walkers = init_walkers(trial_coeff, t1, hs_op, key, n_walkers)

        overlaps = wk.vmap_chunked(meas_ops.overlap, n_chunks=params.n_chunks, in_axes=(0, None))(
            initial_walkers, trial_data
        )

        meas_ctx = meas_ops.build_meas_ctx(ham_data, trial_data)
        e_kernel = meas_ops.require_kernel(k_energy)
        e_samples = jnp.real(
            wk.vmap_chunked(e_kernel, n_chunks=params.n_chunks, in_axes=(0, None, None, None))(
                initial_walkers, ham_data, meas_ctx, trial_data
            )
        )
        e_est = jnp.mean(e_samples)

        pop_shift = e_est

        node_encounters = jnp.asarray(0)

        state = PropState(
            walkers=initial_walkers,
            weights=weights,
            overlaps=overlaps,
            rng_key=key,
            pop_control_ene_shift=pop_shift,
            e_estimate=e_est,
            node_encounters=node_encounters,
        )
        return shard_prop_state(state, mesh)

    return init_prop_state
