#!/usr/bin/env python

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yt
from yt.frontends.boxlib.api import CastroDataset
from scipy.stats import pearsonr, spearmanr

from correlation import (collect_deviates, normalize_speeds, parse_info_txt,
                         get_T_profile, find_x_for_T, read_directories)

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


def integrate(ds, v_mean, x_final, t_final, t_frac=None, t_const=None):
    """ Integrate the masses of Ni56 and Ni58 over the final
        `t_frac` of the time domain of a run.
    """
    #Set integration bounds
    x_hi = x_final
    x_lo = x_final
    if t_frac:
        x_lo -= v_mean * t_final * t_frac
    if t_const:
        assert t_final > t_const, "Integration bounds are larger than total time."
        x_lo -= v_mean * t_const
    if t_frac and t_const:
        raise ValueError("both fractional and absolute time domain provided," \
                         "only one is necessary.")

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
    rho_Fe56 = np.array(region['rho_Fe56'])

    #calculate ratio
    M_Ni56 = np.sum(rho_Ni56 * cell_volume)
    M_Ni58 = np.sum(rho_Ni58 * cell_volume)
    M_Fe56 = np.sum(rho_Fe56 * cell_volume)

    return M_Ni56, M_Ni58, M_Fe56, M_Ni56 / M_Ni58, M_Ni56 / M_Fe56


def correlation_w_deviates(runs, read_runs, frac, const, do_plot, do_linear, write_rates):
    Ni_corrs, Fe_corrs = {}, {}

    #define a dict to store mass ratios
    Ni_mass_ratios = {}
    Fe_mass_ratios = {}

    for prefix, values in read_runs.items():
        #let num=0 correspond to the median
        ds = values["ds"]
        v_mean = values["v_mean"]
        x_final = values["x_final"]
        t_final = values["t_final"]

        if prefix == "median":
            _, _, _, Ni_mass_ratios[0], Fe_mass_ratios[0] = integrate(ds, v_mean, x_final,
                                                                       t_final, frac, const)
        else:
            _, _, _, Ni_mass_ratios[int(prefix)], Fe_mass_ratios[int(prefix)]  = integrate(ds, v_mean, x_final,
                                                                                           t_final, frac, const)

    # read in deviates
    # Note that collect_deviates as written in correlation.py does not
    # account for the median case, we need to artifically introduce it
    deviates = collect_deviates(runs)
    for rate in deviates.keys():
        deviates[rate][0] = 0.0

    # we use dictionaries instead of lists, as some indices might be
    # skipped over due to failed runs. But we still need a sorted way to
    # iterate over these dicts
    nums = sorted(Ni_mass_ratios.keys())

    #Final set up a similar correlation analysis as in correlation.py
    y_Ni = np.array([Ni_mass_ratios[n] for n in nums])
    y_Fe = np.array([Fe_mass_ratios[n] for n in nums])

    print("Correlating mass ratios to rate deviates ...")
    if frac:
        print(f"Mass ratios are calculated over final {frac*100:.2f}% of the time domain")
    if const:
        print(f"Mass ratios are calculated over final {const:.2f}s of the time domain")

    print(slimline)
    if do_linear:
        print(f"{'rate':>30} | {'(r,p) for Ni56/Ni58':>18} | {'(r,p) for Ni56/Fe56':>18}")
    else:
        print(f"{'rate':>30} | {'(ρ,p) for Ni56/Ni58':>18} | {'(ρ,p) for Ni56/Fe56':>18}")
    print(slimline)

    for rate in deviates.keys():
        x = np.array([deviates[rate][n] for n in nums])

        if do_linear:
            Ni_corr, Ni_pval = pearsonr(x, y_Ni)
            Fe_corr, Fe_pval = pearsonr(x, y_Fe)
        else:
            Ni_corr, Ni_pval = spearmanr(x, y_Ni)
            Fe_corr, Fe_pval = spearmanr(x, y_Fe)

        Ni_corrs[rate] = (Ni_corr, Ni_pval)
        Fe_corrs[rate] = (Fe_corr, Fe_pval)

        Ni_str = f"({Ni_corr:.4f} , {Ni_pval:.4f})"
        Fe_str = f"({Fe_corr:.4f} , {Fe_pval:.4f})"
        print(f"{rate:>30} | {Ni_str:>18} | {Fe_str:>18}")

    print(slimline)
    if do_plot:
        plot_mass_v_deviates(do_plot, (Ni_mass_ratios, Fe_mass_ratios), deviates, Ni_corrs, Fe_corrs)
        print("Created mass_ratios.png")
        print(slimline)

    #write out top most correlated rates in a .txt file 
    if write_rates:
        Ni_ranked = sorted(Ni_corrs.items(), key=lambda kv: abs(kv[1][0]), reverse=True)
        Fe_ranked = sorted(Fe_corrs.items(), key=lambda kv: abs(kv[1][0]), reverse=True)

        with open("rates.txt", "w", encoding="utf-8") as file:
            file.write(f"Top {write_rates} most correlated rates for M(Ni56)/M(Ni58)\n")
            for rate, (_, _) in Ni_ranked[:write_rates]:
                file.write(f"{rate}\n")

            file.write("\n")
            file.write(f"Top {write_rates} most correlated rates for M(Ni56)/M(Fe56)\n")
            for rate, (_, _) in Fe_ranked[:write_rates]:
                file.write(f"{rate}\n")
        print("Created rates.txt")
        print(slimline)

    print()
    return (Ni_mass_ratios, Fe_mass_ratios)

def correlation_w_speed(runs, mass_ratios, do_plot):
    print("Correlating shock speed with mass ratios")

    Ni_ratios, Fe_ratios = mass_ratios

    nums = sorted(Ni_ratios.keys())

    # The normalize_speeds() method in correlation.py, is missing
    # the median case, add it artificially
    speeds, median, stdev = normalize_speeds(runs, output=False)
    speeds[0] = 0.0

    x = np.array([speeds[n] for n in nums])
    y_Ni = np.array([Ni_ratios[n] for n in nums])
    y_Fe = np.array([Fe_ratios[n] for n in nums])

    p_corr, p_pval = pearsonr(x, y_Ni)
    s_corr, s_pval = spearmanr(x, y_Ni)

    print("For M(Ni56)/M(Ni58): ")
    print(f"Linear Correlation: r = {p_corr:.4f}, p = {p_pval:.4f}")
    print(f"Monotonic Correlation: ρ = {s_corr:.4f}, p = {s_pval:.4f}")
    print()

    p_corr, p_pval = pearsonr(x, y_Fe)
    s_corr, s_pval = spearmanr(x, y_Fe)

    print("For M(Ni56)/M(Fe56): ")
    print(f"Linear Correlation: r = {p_corr:.4f}, p = {p_pval:.4f}")
    print(f"Monotonic Correlation: ρ = {s_corr:.4f}, p = {s_pval:.4f}")
    print()

    if do_plot:
        fig, (ax_Ni, ax_Fe) = plt.subplots(1, 2, figsize=(12, 5))
        ax_Ni.scatter(x, y_Ni, alpha=0.7, color="b")
        ax_Fe.scatter(x, y_Fe, alpha=0.7, color="r")

        ax_Ni.set_xlabel("Normalized shock speed (z)")
        ax_Fe.set_xlabel("Normalized shock speed (z)")

        ax_Ni.set_ylabel("M(Ni56)/M(Ni58)")
        ax_Fe.set_ylabel("M(Ni56)/M(Fe56)")

        fig.suptitle(f"Mass ratios vs Shock speed")
        fig.text(0.5, -0.03, f"Shock speed (cm/s): {median:.3e} + z * {stdev:.3e}", 
                 ha="center", va="bottom", fontsize=9)
        fig.savefig("mass_v_speed.png", bbox_inches="tight")
        plt.close()

    print(slimline)
    print()

def mass_ratios_over_time(read_runs):
    print("Varying integration bounds to determine Ni mass ratios...")

    fig, ((ax1_ratio, ax1_mass), (ax2_ratio, ax2_mass)) = plt.subplots(2, 2, figsize=(12, 12))

    for prefix, values in read_runs.items():
        #Determine mass ratios for different t_fracs
        t_fracs = np.linspace(0.05, 0.85, num=20)
        t_consts = np.linspace(0.02, 0.20, num=20)

        M_Ni56f, M_Ni56c = np.zeros_like(t_fracs), np.zeros_like(t_consts)
        M_Ni58f, M_Ni58c = np.zeros_like(t_fracs), np.zeros_like(t_consts)
        M_Fe56f, M_Fe56c = np.zeros_like(t_fracs), np.zeros_like(t_consts)

        Ni_ratiosf, Ni_ratiosc = np.zeros_like(t_fracs), np.zeros_like(t_consts)
        Fe_ratiosf, Fe_ratiosc = np.zeros_like(t_fracs), np.zeros_like(t_consts)

        for i, frac in enumerate(t_fracs):
            ds = values["ds"]
            v_mean = values["v_mean"]
            x_final = values["x_final"]
            t_final = values["t_final"]

            (M_Ni56f[i], M_Ni58f[i], M_Fe56f[i],
            Ni_ratiosf[i], Fe_ratiosf[i]) = integrate(ds, v_mean, x_final, t_final, t_frac=frac)

        for i, const in enumerate(t_consts):
            ds = values["ds"]
            v_mean = values["v_mean"]
            x_final = values["x_final"]
            t_final = values["t_final"]

            (M_Ni56c[i], M_Ni58c[i], M_Fe56c[i],
            Ni_ratiosc[i], Fe_ratiosc[i]) = integrate(ds, v_mean, x_final, t_final, t_const=const)


        if prefix == "median":
            ax1_ratio.plot(t_fracs, Ni_ratiosf, label="M(Ni56)/M(Ni58)", color='darkblue', zorder=1)
            ax1_ratio.plot(t_fracs, Fe_ratiosf, label="M(Ni56)/MFe56)", color='darkred', zorder=1)
            ax1_mass.plot(t_fracs, M_Ni56f, label="M_Ni56", color='darkblue', zorder=1)
            ax1_mass.plot(t_fracs, M_Ni58f, label="M_Ni58", color='darkred', zorder=1)
            ax1_mass.plot(t_fracs, M_Fe56f, label="M_Fe58", color='darkgreen', zorder=1)

            ax2_ratio.plot(t_consts, Ni_ratiosc, label="M(Ni56)/M(Ni58)", color='darkblue', zorder=1)
            ax2_ratio.plot(t_consts, Fe_ratiosc, label="M(Ni56)/M(Fe56)", color='darkred', zorder=1)
            ax2_mass.plot(t_consts, M_Ni56c, label="M_Ni56", color='darkblue', zorder=1)
            ax2_mass.plot(t_consts, M_Ni58c, label="M_Ni58", color='darkred', zorder=1)
            ax2_mass.plot(t_consts, M_Fe56c, label="M_Fe56", color='darkgreen', zorder=1)

        else:
            ax1_ratio.plot(t_fracs, Ni_ratiosf, color='b', alpha=0.5, zorder=0)
            ax1_ratio.plot(t_fracs, Fe_ratiosf, color='r', alpha=0.5, zorder=0)
            ax1_mass.plot(t_fracs, M_Ni56f, color='b', alpha=0.5, zorder=0)
            ax1_mass.plot(t_fracs, M_Ni58f, color='r', alpha=0.5, zorder=0)
            ax1_mass.plot(t_fracs, M_Fe56f, color='g', alpha=0.5, zorder=0)

            ax2_ratio.plot(t_consts, Ni_ratiosc, color='b', alpha=0.5, zorder=0)
            ax2_ratio.plot(t_consts, Fe_ratiosc, color='r', alpha=0.5, zorder=0)
            ax2_mass.plot(t_consts, M_Ni56c, color='b', alpha=0.5, zorder=0)
            ax2_mass.plot(t_consts, M_Ni58c, color='r', alpha=0.5, zorder=0)
            ax2_mass.plot(t_consts, M_Fe56c, color='g', alpha=0.5, zorder=0)


    ax1_ratio.set_xlabel("Fraction of final time domain")
    ax1_ratio.set_ylabel("Mass ratios")
    ax1_ratio.set_yscale("log")
    ax1_ratio.legend()

    ax1_mass.set_xlabel("Fraction of final time domain")
    ax1_mass.set_ylabel("Mass")
    ax1_mass.set_yscale("log")
    ax1_mass.legend()

    ax2_ratio.set_xlabel("Const final time domain")
    ax2_ratio.set_ylabel("Mass ratios")
    ax2_ratio.set_yscale("log")
    ax2_ratio.legend()

    ax2_mass.set_xlabel("Const final time domain")
    ax2_mass.set_ylabel("Mass")
    ax2_mass.set_yscale("log")
    ax2_mass.legend()

    fig.suptitle(f"Mass ratios vs Integration bounds")
    fig.savefig("mass_v_integration.png", bbox_inches="tight")
    plt.close()

    print("Created mass_v_integration.png")

def plot_mass_v_deviates(n_rates, mass_ratios, deviates, Ni_corrs, Fe_corrs):

    Ni_ratios, Fe_ratios = mass_ratios

    Ni_ranked = sorted(Ni_corrs.items(), key=lambda kv: abs(kv[1][0]), reverse=True)
    Fe_ranked = sorted(Fe_corrs.items(), key=lambda kv: abs(kv[1][0]), reverse=True)

    nums = sorted(Ni_ratios.keys())
    y_Ni = np.array([Ni_ratios[n] for n in nums])
    y_Fe = np.array([Fe_ratios[n] for n in nums])

    fig, (ax_Ni, ax_Fe) = plt.subplots(1, 2, figsize=(12, 5))

    for rate, (corr, p_val) in Ni_ranked[:n_rates]:
        x = np.array([deviates[rate][n] for n in nums])
        ax_Ni.scatter(x, y_Ni, label=f"{rate} (ρ={corr:.3f}, p={p_val:.3f})",
                      alpha=0.7)

    for rate, (corr, p_val) in Fe_ranked[:n_rates]:
        x = np.array([deviates[rate][n] for n in nums])
        ax_Fe.scatter(x, y_Fe, label=f"{rate} (ρ={corr:.3f}, p={p_val:.3f})",
                      alpha=0.7)

    ax_Ni.set_xlabel("deviate")
    ax_Ni.set_ylabel("Ni56 / Ni58")
    ax_Ni.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=1,
        fontsize="small",
        frameon=True,
    )

    ax_Fe.set_xlabel("deviate")
    ax_Fe.set_ylabel("Ni56 / Fe56")
    ax_Fe.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=1,
        fontsize="small",
        frameon=True,
    )

    fig.suptitle(f"Mass ratios vs deviates for top {n_rates} correlated rates")
    fig.savefig("mass_ratios.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":

    print(boldline)
    print("CORRELATION ANALYSIS of starlib deviates and iron group nuclei")
    print(boldline)

    p = argparse.ArgumentParser()

    p.add_argument("directories", type=str, nargs="+",
                   help="directories holding runs")
    p.add_argument("--do_plot", type=int, default=0,
                   help="Plot and fit n-most correlated rates")
    p.add_argument("--write_rates", type=int, default=5,
                   help="writes a .txt file holding n most correlated rate, default n=5")
    p.add_argument("--do_linear_correlation", action="store_true",
                   help="Does a linear correlation if true, otherwise Spearman Correlation is used")

    time_args = p.add_mutually_exclusive_group()
    time_args.add_argument("--time_frac", type=float, default=None,
                           help="The fraction of time domain that should be integrated")
    time_args.add_argument("--time_const", type=float, default=None,
                           help="The constant time to integrate over across runs")

    args = p.parse_args()

    if args.time_frac:
        if not (0.01 <= args.time_frac <= 1.0):
            raise ValueError("The fraction of time domain to be integrated must be" \
            "a floating point number between 0.01 and 1.0")
    if args.time_const:
        if not (0.01 <= args.time_const <= 2.0):
            raise ValueError("The constant time to be integrated over must be " \
                             "a floating point number between 0.01 and 2.0")

    runs = read_directories(args.directories)

    read_runs = {}
    for prefix, plotfiles in runs.items():
        read_runs[prefix] = read_plotfiles(plotfiles)

    t_final = np.array([v["t_final"] for v in read_runs.values()])
    x_final = np.array([v["x_final"] for v in read_runs.values()])

    #Print out some details about the runs before doing correlation work
    mean_t, std_t = t_final.mean(), t_final.std()
    max_t, min_t = t_final.max(), t_final.min()

    mean_x, std_x = x_final.mean(), x_final.std(),
    max_x, min_x = x_final.max(), x_final.min()

    print("Across runs:")
    print(f"x_final: mean = {mean_x:.3e}, stddev = {std_x:.3e}, min = {min_x:.3e}, max = {max_x:.3e}")
    print(f"t_final: mean = {mean_t:.3f}, stddev = {std_t:.3f}, min = {min_t:.3f}, max = {max_t:.3f}")
    print(boldline)
    print()

    mass_ratios = correlation_w_deviates(runs, read_runs, args.time_frac, args.time_const,
                                         args.do_plot, args.do_linear_correlation, args.write_rates)
    correlation_w_speed(runs, mass_ratios, args.do_plot)
    mass_ratios_over_time(read_runs)

    print(boldline)