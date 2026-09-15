#!/usr/bin/env python

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yt
from yt.frontends.boxlib.api import CastroDataset
from scipy.stats import pearsonr, spearmanr

from correlation import collect_deviates, normalize_speeds, get_T_profile, find_x_for_T

yt.set_log_level(40)

slimline = "-----------------------------------------------------------------------------"
boldline = "============================================================================="

def mean_speed(plotfiles):
    """ Calculates the mean shock speed given plotfiles"""

    # use only the last quarter of plotfiles as this ensures
    # a stable burning for
    quarter = len(plotfiles) // 4
    plotfiles = plotfiles[-quarter:]

    v, dt, = [], []

    for n, p in enumerate(plotfiles):
        time, x, T = get_T_profile(p)
        xpos = find_x_for_T(x, T)

        if n == 0:
            xpos_old = xpos
            time_old = time
        else:
            # difference with the previous file to find the det speed
            # note: the corresponding time is centered in the interval
            v.append((xpos - xpos_old)/(time - time_old))
            dt.append(time - time_old)

            xpos_old = xpos
            time_old = time

    v = np.array(v)
    dt = np.array(dt)

    mean = np.average(v, weights=dt)

    # return mean speed, final x position, final time
    return mean, xpos, time

def read_plotfiles(plotfiles):
    """ This function breaks off the expensive part of
        integrating mass fractions, i.e., reading plotfiles,
        so that files are not read at every integration call
    """
    v_mean, x_final, t_final = mean_speed(plotfiles)
    ds = CastroDataset(plotfiles[-1])

    return {"v_mean": v_mean,
            "x_final": x_final,
            "t_final": t_final,
            "ds": ds}


def integrate(ds, v_mean, x_final, t_final, t_frac):
    """ Integrate the masses of Ni56 and Ni58 over the final
        `t_frac` of the time domain of a run.
    """

    #Set integration bounds
    x_hi = x_final
    x_lo = x_final - v_mean * t_final * t_frac

    #define region
    left_edge = ds.domain_left_edge.copy()
    right_edge = ds.domain_right_edge.copy()

    left_edge[0] = x_lo
    right_edge[0] = x_hi

    region = ds.box(left_edge, right_edge)

    #pull out relevant fields
    cell_volume = np.array(region['cell_volume'])
    rho_Ni56 = np.array(region['rho_Ni56'])
    rho_Ni58 = np.array(region['rho_Ni58'])

    #calculate ratio
    M_Ni56 = np.sum(rho_Ni56 * cell_volume)
    M_Ni58 = np.sum(rho_Ni58 * cell_volume)

    return M_Ni56, M_Ni58, M_Ni56 / M_Ni58


def correlation_w_deviates(runs, read_runs, frac, do_plot, do_fit):
    p_corrs, s_corrs = {}, {}

    #define a dict to store mass ratios
    mass_ratios = {}
    for prefix, values in read_runs.items():
        #let num=0 correspond to the median
        ds = values["ds"]
        v_mean = values["v_mean"]
        x_final = values["x_final"]
        t_final = values["t_final"]

        if prefix == "median":
            _, _, mass_ratios[0] = integrate(ds, v_mean, x_final, t_final, frac)
        else:
            _, _, mass_ratios[int(prefix)] = integrate(ds, v_mean, x_final, t_final, frac)

    # read in deviates
    # Note that collect_deviates as written in correlation.py does not
    # account for the median case, we need to artifically introduce it
    deviates = collect_deviates(runs)
    for rate in deviates.keys():
        deviates[rate][0] = 0.0

    # we use dictionaries instead of lists, as some indices might be
    # skipped over due to failed runs. But we still need a sorted way to
    # iterate over these dicts
    nums = sorted(mass_ratios.keys())

    #Final set up a similar correlation analysis as in correlation.py
    y = np.array([mass_ratios[n] for n in nums])

    print("Correlating mass ratios (Ni_56 / Ni_58) to rate deviates ...")
    print(f"Mass ratios are calculated over final {frac*100:.2f}% of the time domain")
    print(slimline)
    print(f"{'rate':>30} | {'Pearson (r, p)':>18} | {'Spearman (ρ, p)':>18}")
    print(slimline)

    for rate in deviates.keys():
        x = np.array([deviates[rate][n] for n in nums])

        p_corr, p_pval = pearsonr(x, y)
        s_corr, s_pval = spearmanr(x, y)

        p_corrs[rate] = (p_corr, p_pval)
        s_corrs[rate] = (s_corr, s_pval)

        p_str = f"({p_corr:.4f} , {p_pval:.4f})"
        s_str = f"({s_corr:.4f} , {s_pval:.4f})"
        print(f"{rate:>30} | {p_str:>18} | {s_str:>18}")

    if do_plot:
        plot_mass_v_deviates(do_plot, do_fit, mass_ratios, deviates, p_corrs)
        print(slimline)

    print()
    return mass_ratios

def correlation_w_speed(runs, mass_ratios, do_plot):
    print("Correlating shock speed with M(Ni56)/M(Ni58)")

    nums = sorted(mass_ratios.keys())

    # The normalize_speeds() method in correlation.py, is missing
    # the median case, add it artificially
    speeds = normalize_speeds(runs, output=False)
    speeds[0] = 0.0

    x = np.array([speeds[n] for n in nums])
    y = np.array([mass_ratios[n] for n in nums])

    p_corr, p_pval = pearsonr(x, y)
    s_corr, s_pval = spearmanr(x, y)

    print(f"Pearson: r = {p_corr:.4f}, p = {p_pval:.4f}")
    print(f"Spearman: ρ = {s_corr:.4f}, p = {s_pval:.4f}")
    print()

    if do_plot:
        fig, ax = plt.subplots()
        ax.scatter(x, y)
        ax.set_xlabel("Normalized shock speed")
        ax.set_ylabel("M(Ni56)/M(Ni58)")
        ax.set_title(f"Ni mass ratios vs Shock speed")
        fig.savefig("mass_v_speed.png", bbox_inches="tight")
        plt.close()
        print("Created mass_v_speed.png")

    print(slimline)
    print()

def mass_ratios_over_time(read_runs):
    print("Varying integration bounds to determine Ni mass ratios...")
    print()

    fig, (ax_ratio, ax_mass) = plt.subplots(1, 2, figsize=(12, 5))

    for prefix, values in read_runs.items():
        #Determine mass ratios for different t_fracs
        t_fracs = np.linspace(0.05, 0.85, num=20)

        M_Ni56 = np.zeros_like(t_fracs)
        M_Ni58 = np.zeros_like(t_fracs)
        ratios = np.zeros_like(t_fracs)

        for i, frac in enumerate(t_fracs):
            ds = values["ds"]
            v_mean = values["v_mean"]
            x_final = values["x_final"]
            t_final = values["t_final"]

            M_Ni56[i], M_Ni58[i], ratios[i] = integrate(ds, v_mean, x_final, 
                                                        t_final, frac)

        if prefix == "median":
            ax_ratio.plot(t_fracs, ratios, label="median", color='k', zorder=1)
            ax_mass.plot(t_fracs, M_Ni56, label="M_Ni56", color='darkblue', zorder=1)
            ax_mass.plot(t_fracs, M_Ni58, label="M_Ni58", color='darkred', zorder=1)

        else:
            ax_ratio.plot(t_fracs, ratios, color='r', alpha=0.5, zorder=0)
            ax_mass.plot(t_fracs, M_Ni56, color='b', alpha=0.5, zorder=0)
            ax_mass.plot(t_fracs, M_Ni58, color='r', alpha=0.5, zorder=0)

    ax_ratio.set_xlabel("Fraction of final time domain")
    ax_ratio.set_ylabel("Ni56 / Ni58")
    ax_ratio.set_title(f"Ni mass ratios vs Integration bounds")
    ax_ratio.legend()

    ax_mass.set_xlabel("Fraction of final time domain")
    ax_mass.set_ylabel("Mass")
    ax_mass.set_title(f"Ni isotope masses vs Integration bounds")
    ax_mass.legend()

    fig.savefig("mass_v_integration.png", bbox_inches="tight")
    plt.close()

    print("Created mass_v_integration.png")
    print(slimline)

def plot_mass_v_deviates(n_rates, do_fit, mass_ratios, deviates, corrs):

    ranked = sorted(corrs.items(), key=lambda kv: abs(kv[1][0]), reverse=True)

    nums = sorted(mass_ratios.keys())
    y = np.array([mass_ratios[n] for n in nums])

    fig, ax = plt.subplots()

    for rate, (corr, p_val) in ranked[:n_rates]:
        x = np.array([deviates[rate][n] for n in nums])
        ax.scatter(x, y, label=f"{rate} (r={corr:.3f}, p={p_val:.3f})",
                   alpha=0.7)

    if do_fit:
        rate = ranked[0][0]
        x = np.array([deviates[rate][n] for n in nums])
        m, b = np.polyfit(x, y, deg=1)

        x_lin = np.linspace(-3, 3)
        ax.plot(x_lin, m*x_lin + b, label=f"Linear fit for {rate}")

    ax.set_xlabel("deviate")
    ax.set_ylabel("Ni56 / Ni58")
    ax.set_title(f"Ni mass ratios vs deviates for top {n_rates} correlated rates")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        fontsize="small",
        frameon=True,
    )

    fig.savefig("mass_ratios.png", bbox_inches="tight")
    plt.close(fig)
    print("Created mass_ratios.png")

if __name__ == "__main__":

    p = argparse.ArgumentParser()

    p.add_argument("directory", type=str,
                   help="directories holding runs")
    p.add_argument("--time_frac", type=float, default=0.2,
                   help="The fraction of time domain that should be integrated")
    p.add_argument("--do_plot", type=int, default=0,
                   help="Plot and fit n-most correlated rates")
    p.add_argument("--do_fit", action="store_true",
                   help="Does a linear fit on most correlated rate")

    args = p.parse_args()

    dir_path = Path(args.directory)
    if not dir_path.is_dir():
        raise ValueError(f"{dir_path} is not a directory")

    if not (0.01 <= args.time_frac <= 1.0):
        raise ValueError("The fraction of time domain to be integrated must be" \
        "a floating point number between 0.01 and 1.0")

    runs = {}

    for run_dir in dir_path.glob("run_*"):
        #Ensure that none of the log files enter the dict
        if not run_dir.is_dir():
            continue

        prefix = run_dir.name.split("_")[1]
        plotfiles = sorted(run_dir.glob("det_x_plt*"))

        runs[prefix] = plotfiles

    read_runs = {}
    for prefix, plotfiles in runs.items():
        read_runs[prefix] = read_plotfiles(plotfiles)

    print(boldline)
    print("CORRELATION ANALYSIS of starlib deviates with mass ratios of Ni isotopes")
    print(boldline)
    print(f"Total of {len(runs) - 1} + 1 runs found")
    print(slimline)
    print()

    mass_ratios = correlation_w_deviates(runs, read_runs, args.time_frac,
                                         args.do_plot, args.do_fit)
    correlation_w_speed(runs, mass_ratios, args.do_plot)
    mass_ratios_over_time(read_runs)

    print(boldline)