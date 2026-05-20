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
class HsCcsdTrial:
    """
    Fixed-sample HS representation of a restricted CCSD trial.

    ``det_coeffs`` stores the sampled Thouless determinants with shape
    ``(n_dets, norb, nocc)``.  ``ci_coeffs`` stores the linear coefficients in
    the nonorthogonal determinant expansion; the HS sampler uses equal
    coefficients by default.
    """

    det_coeffs: jax.Array
    ci_coeffs: jax.Array

    def __post_init__(self) -> None:
        if not hasattr(self.det_coeffs, "ndim") or not hasattr(self.ci_coeffs, "ndim"):
            return
        if self.det_coeffs.ndim != 3:
            raise ValueError(
                "HsCcsdTrial.det_coeffs must have shape (n_dets, norb, nocc); "
                f"got {self.det_coeffs.shape}."
            )
        if self.ci_coeffs.ndim != 1:
            raise ValueError(
                f"HsCcsdTrial.ci_coeffs must be rank 1; got {self.ci_coeffs.shape}."
            )
        if self.ci_coeffs.shape[0] != self.det_coeffs.shape[0]:
            raise ValueError(
                "HsCcsdTrial.ci_coeffs length must match det_coeffs.shape[0]; "
                f"got {self.ci_coeffs.shape[0]} and {self.det_coeffs.shape[0]}."
            )

    @property
    def ndets(self) -> int:
        return int(self.det_coeffs.shape[0])

    @property
    def norb(self) -> int:
        return int(self.det_coeffs.shape[1])

    @property
    def nocc(self) -> int:
        return int(self.det_coeffs.shape[2])

    def tree_flatten(self):
        return (self.det_coeffs, self.ci_coeffs), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        det_coeffs, ci_coeffs = children
        return cls(det_coeffs=det_coeffs, ci_coeffs=ci_coeffs)


def _projector(c: jax.Array) -> jax.Array:
    metric = c.conj().T @ c
    return c @ jnp.linalg.solve(metric, c.conj().T)


def get_rdm1(trial_data: HsCcsdTrial) -> jax.Array:
    weights = jnp.abs(trial_data.ci_coeffs)
    weights = weights / jnp.sum(weights)
    dms = jax.vmap(_projector)(trial_data.det_coeffs)
    dm = jnp.einsum("s,sij->ij", weights, dms, optimize="optimal")
    dm = 0.5 * (dm + dm.conj().T)
    return jnp.stack([dm, dm], axis=0)


def det_overlaps_r(walker: jax.Array, trial_data: HsCcsdTrial) -> jax.Array:
    mats = jnp.einsum("sni,nj->sij", trial_data.det_coeffs.conj(), walker, optimize="optimal")
    return jax.vmap(jnp.linalg.det)(mats) ** 2


def overlap_hs_r(walker: jax.Array, trial_data: HsCcsdTrial) -> jax.Array:
    return jnp.sum(trial_data.ci_coeffs * det_overlaps_r(walker, trial_data))


def make_hsccsd_trial_ops(sys: System) -> TrialOps:
    if sys.nup != sys.ndn:
        raise ValueError("HS-CCSD trial requires nup == ndn.")
    if sys.walker_kind.lower() != "restricted":
        raise ValueError(
            f"HS-CCSD trial currently supports only restricted walkers, got: {sys.walker_kind}"
        )
    return TrialOps(overlap=overlap_hs_r, get_rdm1=get_rdm1)


def make_hsccsd_trial_data_from_amplitudes(
    trial_coeff: ArrayLike,
    t1: ArrayLike,
    t2: ArrayLike,
    *,
    seed: int = 0,
    n_samples: int = 100,
) -> HsCcsdTrial:
    hs_op = build_hs_op(t2)
    key = jax.random.PRNGKey(int(seed))
    det_coeffs = init_walkers(trial_coeff, t1, hs_op, key, int(n_samples))
    ci_coeffs = jnp.full((int(n_samples),), 1.0 / float(n_samples), dtype=det_coeffs.dtype)
    return HsCcsdTrial(det_coeffs=det_coeffs, ci_coeffs=ci_coeffs)


def make_hsccsd_trial_data(data: dict, sys: System | None = None) -> HsCcsdTrial:
    del sys
    if "det_coeffs" in data:
        det_coeffs = jnp.asarray(data["det_coeffs"])
        ci_raw = data.get("ci_coeffs")
        if ci_raw is None:
            ci_coeffs = jnp.full(
                (det_coeffs.shape[0],),
                1.0 / float(det_coeffs.shape[0]),
                dtype=det_coeffs.dtype,
            )
        else:
            ci_coeffs = jnp.asarray(ci_raw, dtype=det_coeffs.dtype)
        return HsCcsdTrial(det_coeffs=det_coeffs, ci_coeffs=ci_coeffs)

    trial_coeff = data.get("trial_coeff", data.get("mo"))
    if trial_coeff is None:
        raise KeyError("HS-CCSD trial data requires 'det_coeffs' or 'trial_coeff'/'mo'.")
    return make_hsccsd_trial_data_from_amplitudes(
        trial_coeff,
        data["t1"],
        data["t2"],
        seed=int(data.get("seed", 0)),
        n_samples=int(data.get("n_samples", 100)),
    )


def build_hs_op(t2: ArrayLike) -> jax.Array:
    """
    Builds the Cholesky decomposition of CCSD T2 amplitudes,
    T2 = LL^T.

    Input:
    t2: CCSD T2 amplitudes

    Output:
    L: Cholesky vectors of the CCSD T2 amplitudes
    """
    t2 = jnp.asarray(t2)

    nO, _, nV, _ = t2.shape

    # Number of excitations
    nex = nO * nV

    assert t2.shape == (nO, nO, nV, nV)

    # t2(i,j,a,b) -> t2(ai,bj)
    t2 = jnp.einsum("ijab->aibj", t2)
    t2 = t2.reshape(nex, nex)

    # t2 = LL^T
    e_val, e_vec = jnp.linalg.eigh(t2)
    L = e_vec @ jnp.diag(jnp.sqrt(e_val + 0.0j))
    assert abs(jnp.linalg.norm(t2 - L @ L.T)) < 1e-12

    # Summation on the left to have a list of operators
    L = L.T.reshape(nex, nV, nO)

    return L


def init_walkers(
    trial_coeff: ArrayLike,
    t1: ArrayLike,
    hs_op: jax.Array,
    subkey: jax.Array,
    n_w: int,
) -> jax.Array:
    """
    Builds a stochastic representation of the UCCSD wavefunction using the Hubbard-Stratonovich transformation.

    Input:
    trial_coeff: mo coefficients in the alpha mo basis
    t1         : CCSD T1 amplitudes
    hs_op      : Cholesky vectors of the CCSD T2 amplitudes
    subkey     : PRNG key
    n_w        : number of walkers

    Output:
    w: walkers
    """
    t1 = jnp.asarray(t1)

    nO, nV = t1.shape
    n = nO + nV
    nex = nO * nV

    L = hs_op
    assert L.shape == (nex, nV, nO)

    C = trial_coeff
    C = jnp.asarray(C)
    assert C.shape == (n, n)

    C_occ, C_vir = jnp.split(C, [nO], axis=1)

    # e^T1
    e_t1 = t1.T + 0.0j

    ops = jnp.array([e_t1] * n_w)

    fields = jax.random.normal(subkey, shape=(n_w, nex))

    # e^{T1+T2}
    ops = ops + jnp.einsum("wg,gai->wai", fields, L)

    # Initial walkers
    dm = C[:, :nO] @ C[:, :nO].conj().T
    nos = jnp.linalg.eigh(dm)[1][:, ::-1][:, :nO]

    w = jnp.array([nos + 0.0j] * n_w)

    id_ = jnp.array([jnp.identity(n) + 0.0j] * n_w)

    # e^{T1+T2} \ket{\phi}
    w = (id_ + jnp.einsum("pa,wai,iq -> wpq", C_vir, ops, C_occ.T)) @ w

    return w


def make_init_prop_state(trial_coeff: ArrayLike, t1: ArrayLike, t2: ArrayLike):
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
        assert sys.walker_kind == "restricted"
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
