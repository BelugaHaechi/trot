from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, cast

import numpy as np
import jax.numpy as jnp

if TYPE_CHECKING:
    import jax


def _pick_plateau(
    Bs: np.ndarray,
    SEs: np.ndarray,
    Gs: np.ndarray,
    *,
    min_blocks: int = 20,
    min_rise: float = 0.20,
    flat_tol: float = 0.03,
    k: int = 3,
) -> tuple[int, float, int]:
    assert Bs.size > 0
    Bs, SEs, Gs = map(np.asarray, (Bs, SEs, Gs))
    ok = Gs >= min_blocks
    Bs2, SEs2, Gs2 = Bs[ok], SEs[ok], Gs[ok]
    if Bs2.size == 0:
        return int(Bs[0]), float(SEs[0]), int(Gs[0])
    rise_ok = SEs2 >= (1.0 + min_rise) * SEs2[0]
    for i in range(0, Bs2.size - k):
        if not rise_ok[i]:
            continue
        window = SEs2[i : i + k + 1]
        if np.all(np.abs(np.diff(window)) <= flat_tol * window[:-1]):
            return int(Bs2[i]), float(SEs2[i]), int(Gs2[i])
    finite = np.isfinite(SEs2)
    if not np.any(finite):
        return int(Bs[0]), float(SEs[0]), int(Gs[0])
    jmax = int(np.where(finite, SEs2, -np.inf).argmax())
    thresh = 0.95 * SEs2[jmax]
    candidates = np.where(SEs2 >= thresh)[0]
    j = int(candidates[0]) if candidates.size > 0 else jmax
    return int(Bs2[j]), float(SEs2[j]), int(Gs2[j])


def blocking_analysis_ratio(
    ene: np.ndarray | jax.Array,
    wt: np.ndarray | jax.Array,
    block_grid: Iterable[int] | None = None,
    *,
    min_blocks: int = 20,
    min_rise: float = 0.20,
    flat_tol: float = 0.03,
    k: int = 3,
    bins: int | str = "fd",
    figsize: tuple[float, float] = (12, 4.2),
    title: str | None = None,
    print_q: bool = True,
    plot_q: bool = False,
    exact: float | None = None,
) -> Dict[str, Any]:
    """Blocking analysis for mu = sum(wt*ene)/sum(wt)"""
    ene = np.asarray(ene, float).ravel()
    wt = np.asarray(wt, float).ravel()
    n = ene.size
    assert wt.size == n

    S = wt * ene
    N = wt
    S_tot, N_tot = S.sum(), N.sum()
    mu_full = S_tot / N_tot

    if block_grid is None:
        raw = np.unique(np.rint(np.geomspace(1, max(2, n // min_blocks), 18)).astype(int))
        block_grid = [int(b) for b in raw if b >= 1 and (n // b) >= min_blocks]
        if (n // raw[-1]) >= 5 and raw[-1] not in block_grid:
            block_grid.append(int(raw[-1]))

    Bs_list: list[int] = []
    SEs_list: list[float] = []
    Gs_list: list[int] = []
    LOO_cache: dict[int, tuple[np.ndarray, float, int]] = {}
    for B in block_grid:
        G = n // B
        if G < 5:
            continue
        usable = G * B
        Sg = S[:usable].reshape(G, B).sum(axis=1)
        Ng = N[:usable].reshape(G, B).sum(axis=1)
        St, Nt = Sg.sum(), Ng.sum()

        denom_loo = Nt - Ng
        safe = np.abs(denom_loo) > 1e-18
        mu_loo = np.where(safe, (St - Sg) / denom_loo, St / Nt)

        mu_bar = mu_loo.mean()
        var = (G - 1) / G * np.sum((mu_loo - mu_bar) ** 2)
        se = float(np.sqrt(max(var, 0.0)))

        Bs_list.append(B)
        SEs_list.append(se)
        Gs_list.append(G)
        LOO_cache[B] = (mu_loo, mu_bar, G)

    Bs = np.array(Bs_list, int)
    SEs = np.array(SEs_list, float)
    Gs = np.array(Gs_list, int)
    ci95: tuple[float, float] | None = None
    if Bs.size == 0:
        B_star: int | None = None
        se_star: float | None = None
        G_star: int | None = None
    else:
        B_star, se_star, G_star = _pick_plateau(
            Bs,
            SEs,
            Gs,
            min_blocks=min_blocks,
            min_rise=min_rise,
            flat_tol=flat_tol,
            k=k,
        )
        ci95 = (mu_full - 1.96 * se_star, mu_full + 1.96 * se_star)

    if B_star is None:
        # Blocking analysis not possible
        out = {
            "mu": float(mu_full),
            "block_sizes": None,
            "se_curve": None,
            "n_blocks": None,
            "B_star": None,
            "se_star": None,
            "ci95_star": (None, None),
            "estimator_scale_samples": None,
            "bias": None,
            "z_score": None,
        }
        return out

    se_star = cast(float, se_star)
    ci95 = cast(tuple[float, float], ci95)
    mu_loo, mu_bar, G = LOO_cache[B_star]
    est_samples = mu_full + (G - 1) / np.sqrt(G) * (mu_loo - mu_bar)

    bias = z = None
    if exact is not None and np.isfinite(se_star) and se_star > 0:
        bias = float(mu_full - exact)
        z = float((mu_full - exact) / se_star)

    out = {
        "mu": float(mu_full),
        "block_sizes": Bs,
        "se_curve": SEs,
        "n_blocks": Gs,
        "B_star": int(B_star),
        "se_star": float(se_star),
        "ci95_star": (float(ci95[0]), float(ci95[1])),
        "estimator_scale_samples": est_samples,
        "bias": bias,
        "z_score": z,
    }

    if print_q:
        print(f"mu: {out['mu']:.16g}  SE*: {out['se_star']:.16g}  95% CI: {out['ci95_star']}")
        if out["z_score"] is not None:
            print(f"bias: {out['bias']:.16g}  z: {out['z_score']:.6g}")

        # table: block size vs SE, mark chosen B*
        se0 = float(SEs[0]) if SEs.size else float("nan")
        print("\nBlocking SE curve (ratio LOO):")
        print(f"{'':1s}{'B':>6s} {'G':>6s} {'SE':>14s} {'SE/SE(B=1)':>12s}")
        for B, G, se in zip(Bs, Gs, SEs):
            mark = "*" if int(B) == int(B_star) else " "
            rel = (float(se) / se0) if (se0 > 0 and np.isfinite(se0)) else float("nan")
            print(f"{mark}{int(B):6d} {int(G):6d} {float(se):14.6e} {rel:12.3f}")
        print("")  # trailing newline

    if plot_q:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # SE curve
        ax1.plot(Bs, SEs, marker="o", lw=1.6)
        ax1.axvline(B_star, ls="--", color="k", alpha=0.85, label=f"chosen B = {B_star}")
        if exact is not None:
            ax1.set_title(
                (title or "Blocking SE for ratio estimator")
                + "\n"
                + rf"$\mu$={mu_full:.6f}, SE*={se_star:.3e}, bias={bias:.3e}, z={z:.2f}"
            )
        else:
            ax1.set_title(title or "Blocking SE for ratio estimator")
        ax1.set_xscale("log")
        ax1.set_xlabel("block size B (walkers)")
        ax1.set_ylabel(r"SE[$\mu$]")
        ax1.grid(True, alpha=0.25)
        ax1.legend()

        # estimator-scale histogram
        ax2.hist(est_samples, bins=bins, density=True, alpha=0.6, edgecolor="white")
        xs = np.linspace(mu_full - 6 * se_star, mu_full + 6 * se_star, 400)
        pdf = (1.0 / (se_star * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((xs - mu_full) / se_star) ** 2
        )
        ax2.plot(xs, pdf, lw=2.0, color="#f58518", label="Normal(SE*)")
        ax2.axvline(mu_full, ls="--", color="k", lw=1.2, label=r"$\hat\mu$")
        ax2.axvline(ci95[0], ls=":", color="k", lw=1.2, label="95% CI")
        ax2.axvline(ci95[1], ls=":", color="k", lw=1.2)
        if exact is not None:
            ax2.axvline(exact, ls="--", color="red", lw=1.4, label="exact/target")
        ax2.set_xlabel("estimator-scale (rescaled LOO)")
        ax2.set_ylabel("density")
        ax2.legend()
        fig.tight_layout()

    return out


def reject_outliers(
    data: np.ndarray | jax.Array,
    obs: int,
    m: float = 10.0,
    min_threshold: float = 1e-5,
) -> tuple[Any, Any]:
    target = data[:, obs]
    median_val = np.median(target)
    d = np.abs(target - median_val)
    mdev = np.median(d)
    q1, q3 = np.percentile(target, [25, 75])
    iqr = q3 - q1
    normalized_iqr = iqr / 1.349
    dispersion = max(mdev, normalized_iqr, min_threshold)
    s = d / dispersion
    mask = s < m
    return data[mask], mask


def jackknife_ratios(
    num: np.ndarray,
    denom: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Jackknife mean and standard error for a ratio estimator with array valued numerator.

    Parameters
    ----------
    num : np.ndarray
        Numerator samples, shape (n_samples, *obs_shape).
    denom : np.ndarray
        Denominator samples, shape (n_samples,).

    Returns
    -------
    mean : np.ndarray
        Jackknife estimate of the ratio mean, shape (*obs_shape,).
    sigma : np.ndarray
        Jackknife standard error, shape (*obs_shape,).
    """
    num = np.asarray(num)
    denom = np.asarray(denom).ravel()
    n = num.shape[0]
    assert denom.shape[0] == n

    num_sum = num.sum(axis=0)
    denom_sum = denom.sum()

    # leave one out sums
    loo_num = (num_sum - num) / (n - 1)  # (n, *obs_shape)
    d_shape = (n,) + (1,) * (num.ndim - 1)
    loo_denom = (denom_sum - denom).reshape(d_shape) / (n - 1)  # (n, 1, ...)

    loo_ratio = (loo_num / loo_denom).real  # (n, *obs_shape)
    mean = loo_ratio.mean(axis=0)
    sigma = np.sqrt((n - 1) * np.var(loo_ratio, axis=0))
    return mean, sigma


def rebin_observable(
    obs: np.ndarray,
    weights: np.ndarray,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rebin block-level observable data into larger super-blocks.

    Parameters
    ----------
    obs : np.ndarray
        Per-block weighted-mean observable, shape ``(n_blocks, *obs_shape)``.
    weights : np.ndarray
        Per-block total weights, shape ``(n_blocks,)``.
    block_size : int
        Number of original blocks per super-block.

    Returns
    -------
    num : np.ndarray
        Super-block numerator sums, shape ``(n_groups, *obs_shape)``.
    denom : np.ndarray
        Super-block denominator sums, shape ``(n_groups,)``.
    """
    obs = np.asarray(obs)
    weights = np.asarray(weights).ravel()
    n = obs.shape[0]
    n_groups = n // block_size
    usable = n_groups * block_size

    w = weights[:usable].reshape(n_groups, block_size)
    w_shape = (n_groups, block_size) + (1,) * (obs.ndim - 1)
    o = obs[:usable].reshape((n_groups, block_size) + obs.shape[1:])

    denom = w.sum(axis=1)  # (n_groups,)
    num = (w.reshape(w_shape) * o).sum(axis=1)  # (n_groups, *obs_shape)
    return num, denom


def clean_pt2ccsd(ept_sp, wt_sp, t2_sp, e0_sp, e1_sp, zeta=20):
    # print(f'Clean AFQMC/pt2CCSD Observation...')
    d = jnp.abs(ept_sp - jnp.median(ept_sp))
    d_med = jnp.median(d)
    d_med = jnp.where(d_med == 0, 1e-10, d_med)
    z = d / d_med
    mask = z < zeta
    print(
        f"Remove outlier blocks zeta {z[~mask]} \n"
        f"                    energy {ept_sp[~mask]} \n"
        f"                    weight {wt_sp.real[~mask]} "
    )

    wt_clean = wt_sp[mask]
    t2_clean = t2_sp[mask]
    e0_clean = e0_sp[mask]
    e1_clean = e1_sp[mask]

    return (wt_clean, t2_clean, e0_clean, e1_clean)


def pt2ccsd_blocking(
    h0, weights, t2_sp, e0_sp, e1_sp, printQ=False, min_blocks=5, plateau_window=2, plateau_tol=0.04
):
    nsample = len(weights)
    max_size = max(1, nsample // min_blocks)

    block_errs = []
    block_means = []
    block_sizes = []

    for block_size in range(1, max_size + 1):
        n_blocks = nsample // block_size
        if n_blocks < min_blocks:
            break

        wt_truncated = weights[: n_blocks * block_size]
        t2_truncated = t2_sp[: n_blocks * block_size]
        e0_truncated = e0_sp[: n_blocks * block_size]
        e1_truncated = e1_sp[: n_blocks * block_size]

        wt_t2 = wt_truncated * t2_truncated
        wt_e0 = wt_truncated * e0_truncated
        wt_e1 = wt_truncated * e1_truncated

        wt = wt_truncated.reshape(n_blocks, block_size)
        wt_t2 = wt_t2.reshape(n_blocks, block_size)
        wt_e0 = wt_e0.reshape(n_blocks, block_size)
        wt_e1 = wt_e1.reshape(n_blocks, block_size)

        block_wt = jnp.sum(wt, axis=1)
        block_t2 = jnp.sum(wt_t2, axis=1) / block_wt
        block_e0 = jnp.sum(wt_e0, axis=1) / block_wt
        block_e1 = jnp.sum(wt_e1, axis=1) / block_wt

        block_energy = (h0 + block_e0 + block_e1 - block_t2 * block_e0).real
        block_mean = jnp.mean(block_energy)
        block_error = jnp.std(block_energy, ddof=1) / jnp.sqrt(n_blocks)

        block_sizes.append(block_size)
        block_means.append(block_mean)
        block_errs.append(block_error)

    # --- Plateau detection ---
    errs = jnp.array(block_errs)
    plateau_idx = None

    if len(errs) >= plateau_window + 1:
        for i in range(1, len(errs) - plateau_window + 1):
            window = errs[i : i + plateau_window]
            rel_changes = jnp.abs(jnp.diff(window) / window[:-1])
            if jnp.all(rel_changes < plateau_tol):
                plateau_idx = i
                break

    if plateau_idx is not None:
        err = jnp.mean(errs[plateau_idx : plateau_idx + plateau_window])
    else:
        err = errs.max()

    # --- Overall energy ---
    wt_avg = jnp.mean(weights)
    t2_avg = jnp.mean(weights * t2_sp) / wt_avg
    e0_avg = jnp.mean(weights * e0_sp) / wt_avg
    e1_avg = jnp.mean(weights * e1_sp) / wt_avg
    energy_avg = h0 + e0_avg + e1_avg - t2_avg * e0_avg

    # --- Printing ---
    if printQ:
        print("Performing Blocking Analysis for AFQMC/pt2CCSD energy...")
        print(f"{'Bsz':>4s}  {'NB':>4s}  {'Nsp':>4s}  {'Energy':>11s}  {'Error':>8s}")

        if plateau_idx is not None:
            print_end = min(len(block_errs), plateau_idx + plateau_window + 3)
        else:
            print_end = len(block_errs)

        for i in range(print_end):
            bs = block_sizes[i]
            nb = nsample // bs
            marker = "  <--" if (plateau_idx is not None and i == plateau_idx) else ""
            print(
                f"{bs:4d}  {nb:4d}  {bs*nb:4d}  {block_means[i]:11.6f}  {block_errs[i]:8.6f}{marker}"
            )

        if plateau_idx is not None:
            print(f"Plateau found at block size {block_sizes[plateau_idx]}, error = {err.real:.6f}")
        else:
            print(f"No plateau found, using max error = {err.real:.6f}")

    return energy_avg.real, err.real


def ptccsd_ensemble_blocking(
    h0,
    weights,
    components,
    printQ=False,
    min_blocks=5,
    min_rise=0.20,
    flat_tol=0.03,
    k=3,
    screen_outliers=True,
    outlier_m=12.0,
    min_screen_blocks=20,
    max_reject_fraction=0.02,
    denom_floor_rel=1.0e-12,
):
    """
    Blocking analysis for ensemble-level first-order PT-CCSD components.

    For PT-guided propagation, ``components`` must have columns
    ``[r, r * theta, r * electronic_0, r * h_t]`` where
    ``r = exp(-theta)`` is the inverse PT guide factor.  Each row is a
    block-level AFQMC-weighted mean.  The estimator is

        h0 + <electronic_0> + <h_t> - <theta> * <electronic_0>

    with the averages taken using the corrected weights ``weights * r``.  The
    uncertainty is estimated by leave-one-superblock-out jackknife over weighted
    component sums.

    Legacy ``[theta, electronic_0, h_t]`` components are still accepted; those
    use the old HF-guide-style block weighting and are not the formal PT-guide
    projected estimator.

    A conservative block-level screen removes nonfinite blocks, effectively
    zero-denominator blocks, and at most ``max_reject_fraction`` of otherwise
    finite blocks whose corrected block energy is more than ``outlier_m`` robust
    scales from the median.
    """
    weights = np.asarray(weights).ravel()
    components = np.asarray(components)
    if components.ndim != 2 or components.shape[1] not in (3, 4):
        raise ValueError(
            "components must have shape (n_blocks, 4) with columns "
            "[r, r * theta, r * electronic_0, r * h_t], or legacy shape "
            "(n_blocks, 3) with columns [theta, electronic_0, h_t]."
        )
    if components.shape[0] != weights.shape[0]:
        raise ValueError(
            f"weights length {weights.shape[0]} does not match components length "
            f"{components.shape[0]}."
        )

    use_guide_reweight = components.shape[1] == 4

    def block_energy_for_screen(
        weights_in: np.ndarray, components_in: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        weighted_in = weights_in[:, None] * components_in
        if use_guide_reweight:
            denom_in = weighted_in[:, 0]
            with np.errstate(divide="ignore", invalid="ignore"):
                avg_in = weighted_in[:, 1:] / denom_in[:, None]
        else:
            denom_in = weights_in
            avg_in = components_in
        theta_in = avg_in[:, 0]
        e0_in = avg_in[:, 1]
        ht_in = avg_in[:, 2]
        energy_in = (h0 + e0_in + ht_in - theta_in * e0_in).real
        return denom_in, energy_in

    denom_blocks, energy_blocks = block_energy_for_screen(weights, components)
    finite_mask = np.isfinite(weights)
    finite_mask &= np.all(np.isfinite(components), axis=1)
    finite_mask &= np.isfinite(denom_blocks)
    finite_mask &= np.isfinite(energy_blocks)

    denom_abs = np.abs(denom_blocks)
    denom_positive = denom_abs[np.isfinite(denom_abs) & (denom_abs > 0.0)]
    if denom_positive.size > 0:
        denom_floor = max(1.0e-300, float(denom_floor_rel) * float(np.median(denom_positive)))
    else:
        denom_floor = 1.0e-300
    finite_mask &= denom_abs > denom_floor

    robust_reject = np.zeros_like(finite_mask, dtype=bool)
    if screen_outliers:
        good_idx = np.flatnonzero(finite_mask)
        good_energy = energy_blocks[good_idx]
        if good_energy.size >= int(min_screen_blocks):
            center = np.median(good_energy)
            abs_dev = np.abs(good_energy - center)
            mad_scale = 1.4826 * np.median(abs_dev)
            q1, q3 = np.percentile(good_energy, [25.0, 75.0])
            iqr_scale = (q3 - q1) / 1.349
            min_scale = max(1.0e-12, 1.0e-12 * abs(float(center)))
            scale = max(float(mad_scale), float(iqr_scale), min_scale)
            candidates = np.flatnonzero(abs_dev > float(outlier_m) * scale)
            min_keep = max(int(min_blocks), 2)
            max_reject = int(np.floor(float(max_reject_fraction) * good_energy.size))
            max_reject = max(0, min(max_reject, good_energy.size - min_keep))
            if candidates.size > 0 and max_reject > 0:
                order = np.argsort(abs_dev[candidates])[::-1]
                rejected_local = candidates[order[:max_reject]]
                robust_reject[good_idx[rejected_local]] = True

    keep_mask = finite_mask & ~robust_reject
    n_rejected_bad = int(weights.shape[0] - np.count_nonzero(finite_mask))
    n_rejected_robust = int(np.count_nonzero(robust_reject))
    if np.any(~keep_mask):
        weights = weights[keep_mask]
        components = components[keep_mask]

    if n_rejected_bad or n_rejected_robust:
        print(
            "PT-CCSD ensemble outlier screen rejected "
            f"{n_rejected_bad + n_rejected_robust} blocks "
            f"({n_rejected_bad} invalid, {n_rejected_robust} robust outliers)."
        )

    n = weights.shape[0]
    if n == 0:
        raise ValueError("No valid PT-CCSD blocks remain after outlier screening.")
    max_size = max(1, n // min_blocks)
    raw = np.unique(np.rint(np.geomspace(1, max(2, max_size), 18)).astype(int))
    block_grid = [int(b) for b in raw if b >= 1 and (n // b) >= min_blocks]

    weighted_components = weights[:, None] * components
    if use_guide_reweight:
        total_c = weighted_components.sum(axis=0)
        total_w = total_c[0]
        full_avg = total_c[1:] / total_w
    else:
        total_w = weights.sum()
        total_c = weighted_components.sum(axis=0)
        full_avg = total_c / total_w
    theta_avg, e0_avg, ht_avg = full_avg
    energy_avg = h0 + e0_avg + ht_avg - theta_avg * e0_avg

    block_sizes = []
    block_errs = []
    n_blocks_curve = []
    for block_size in block_grid:
        n_groups = n // block_size
        if n_groups < 2:
            continue
        usable = n_groups * block_size
        w_group = weights[:usable].reshape(n_groups, block_size).sum(axis=1)
        c_group = (
            weighted_components[:usable]
            .reshape(n_groups, block_size, components.shape[1])
            .sum(axis=1)
        )

        c_total = c_group.sum(axis=0)

        if use_guide_reweight:
            denom = c_total[0] - c_group[:, 0]
            safe = np.abs(denom) > 1.0e-18
            c_loo = c_total[None, :] - c_group
            avg_loo = np.where(
                safe[:, None],
                c_loo[:, 1:] / denom[:, None],
                c_total[None, 1:] / c_total[0],
            )
        else:
            w_total = w_group.sum()
            denom = w_total - w_group
            safe = np.abs(denom) > 1.0e-18
            c_loo = c_total[None, :] - c_group
            avg_loo = np.where(safe[:, None], c_loo / denom[:, None], c_total[None, :] / w_total)

        theta_loo = avg_loo[:, 0]
        e0_loo = avg_loo[:, 1]
        ht_loo = avg_loo[:, 2]
        energy_loo = (h0 + e0_loo + ht_loo - theta_loo * e0_loo).real

        loo_bar = energy_loo.mean()
        var = (n_groups - 1) / n_groups * np.sum((energy_loo - loo_bar) ** 2)
        block_sizes.append(block_size)
        block_errs.append(float(np.sqrt(max(var, 0.0))))
        n_blocks_curve.append(n_groups)

    if not block_errs:
        err = np.nan
        block_star = None
    else:
        bs = np.asarray(block_sizes, dtype=int)
        errs = np.asarray(block_errs, dtype=float)
        groups = np.asarray(n_blocks_curve, dtype=int)
        block_star, err, _ = _pick_plateau(
            bs,
            errs,
            groups,
            min_blocks=min_blocks,
            min_rise=min_rise,
            flat_tol=flat_tol,
            k=k,
        )

    if printQ:
        print("Performing Blocking Analysis for ensemble PT-CCSD energy...")
        print(f"{'Bsz':>4s}  {'NB':>4s}  {'Energy':>11s}  {'Error':>8s}")
        for block_size, n_groups, block_err in zip(block_sizes, n_blocks_curve, block_errs):
            marker = "  <--" if block_star is not None and block_size == block_star else ""
            print(
                f"{block_size:4d}  {n_groups:4d}  "
                f"{energy_avg.real:11.6f}  {block_err:8.6f}{marker}"
            )

    return jnp.asarray(energy_avg.real), jnp.asarray(err)
