#!/usr/bin/env python

import argparse
from pathlib import Path
import os


import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import yt
import numpy as np
from scipy.stats import pearsonr, spearmanr

from shock_speed import read_directories, shock_speed

yt.set_log_level(40)

slimline = "-----------------------------------------------------------------------------"
boldline = "============================================================================="

def get_T_profile(plotfile):

    ds = yt.load(plotfile)

    time = float(ds.current_time)
    ad = ds.all_data()

    # Sort the ray values by 'x' so there are no discontinuities
    # in the line plot
    srt = np.argsort(ad['x'])
    x_coord = np.array(ad['x'][srt])
    temp = np.array(ad['Temp'][srt])

    return time, x_coord, temp


def find_x_for_T(x, T, T_0=2.e9):
    """ given a profile x(T), find the x_0 that corresponds to T_0 """

    # our strategy here assumes that the hot ash is in the early part
    # of the profile.  We then find the index of the first point where
    # T drops below T_0
    idx = np.where(T < T_0)[0][0]

    T1 = T[idx-1]
    x1 = x[idx-1]

    T2 = T[idx]
    x2 = x[idx]

    slope = (x2 - x1)/(T2 - T1)

    return x1 + slope*(T_0 - T1)

def normalize_speeds(runs, output=True):
    #initialize variables
    shock_speeds = {}
    shock_cvs = {}

    # ensure and establish a run corresponding to the median case
    # read only the final quarter files for efficiency.
    mfiles = runs["median"]
    quarter = len(mfiles) // 4
    _, _, med_v = shock_speed(mfiles[-quarter:], trim=False)

    # calculate mean shock speeds for sampled runs
    for prefix, pfiles in runs.items():
        if prefix != "median":
            _, _, shock_speeds[int(prefix)] = shock_speed(pfiles)

    #normalize shock speeds
    std = np.std(np.array([v[0] for v in shock_speeds.values()]), ddof=1)
    norm_shock_speeds = {
        num: (v[0] - med_v[0]) / std
        for num, v in shock_speeds.items()
    }

    #sort the speeds
    shock_speeds = dict(sorted(shock_speeds.items()))
    norm_shock_speeds = dict(sorted(norm_shock_speeds.items()))

    if output:
        print(f"Normalizing shock speeds...")
        print()
        print(f"Shock speed with median rates (s_med): {med_v[0]:.4e}")
        print(f"Rel. Unc. for shock speed with median rates (CV_med): {100 * med_v[1]/med_v[0]:.3f}%")
        print(f"Std. Dev. for shock speed across sampled runs (σ): {std:.4e}")
        print()
        print(slimline)
        print(f"{'run #':>6} | {'shock speed (s_i)':>18} | {'CV':>6} | {'z=(s_i-s_med)/σ':>18}")
        print(slimline)
        for num, v in shock_speeds.items():
            print(f"{num:>6} | {v[0]:>18.4e} | {100*v[1]/v[0]:>6.2f} | {norm_shock_speeds[num]:>18.4f}")
        print(slimline)
        print()

    return norm_shock_speeds, med_v[0], std

def read_deviates(plotfile):
    job_info_path = os.path.join(plotfile, "job_info")

    deviates = {}
    in_block = False

    with open(job_info_path) as file:
        for line in file:
            line = line.strip()

            #flip flag once the code reaches the starlib deviates block
            if "Deviates for StarLib Rates" in line:
                in_block = True
                continue

            #until starlib block is reached keep reading the file
            if not in_block:
                continue

            # if we have reached a title decorator after populating deviates,
            # it implies all deviates have been read, thus terminate the loop
            if line.startswith("=") and deviates:
                break

            #skip through any header, decorators and blank lines after the title
            if (not line or line.startswith("=") or line.startswith("-")
                or line.startswith("index")):
                continue

            line = line.split()
            if len(line) == 3 and line[0].isdigit():
                _, rate, p = line
                deviates[rate] = float(p)

    return deviates

def collect_deviates(runs):
    rate_deviates = {}

    for prefix, files in runs.items():
        if prefix == "median":
            continue

        deviates = read_deviates(files[0])

        for rate_name, val in deviates.items():
            if rate_name not in rate_deviates:
                rate_deviates[rate_name] = {}

            rate_deviates[rate_name][int(prefix)] = val

    return rate_deviates

def analysis(runs):
    norm_speeds, median, stdev = normalize_speeds(runs)
    deviates = collect_deviates(runs)

    # We use dictionaries, as some indices might be skipped over due 
    # to failed runs. But we still need a sorted way to
    # iterate over these dicts
    nums = sorted(norm_speeds.keys())
    y = np.array([norm_speeds[n] for n in nums])

    p_corrs, s_corrs = {}, {}

    print("Determining Correlation Coefficients ...")
    print(slimline)
    print(f"{'rate':>30} | {'Pearson (r, p)':>18} | {'Spearman (ρ, p)':>18}")
    print(slimline)
    for rate in deviates.keys():
        x = np.array([deviates[rate][n] for n in nums])

        #The deviate for a rate may be fixed, to study isolated effects
        if np.all(x==x[0]):
            continue

        p_corr, p_pval = pearsonr(x, y)
        s_corr, s_pval = spearmanr(x, y)

        p_corrs[rate] = (p_corr, p_pval)
        s_corrs[rate] = (s_corr, s_pval)

        p_str = f"({p_corr:.4f} , {p_pval:.4f})"
        s_str = f"({s_corr:.4f} , {s_pval:.4f})"
        print(f"{rate:>30} | {p_str:>18} | {s_str:>18}")

    print(slimline)

    return norm_speeds, deviates, p_corrs, s_corrs, median, stdev

def plot(n_rates, do_fit, speeds, deviates, corrs, median, stdev):

    ranked = sorted(corrs.items(), key=lambda kv: abs(kv[1][0]), reverse=True)

    nums = sorted(speeds.keys())
    y = np.array([speeds[n] for n in nums])

    fig, ax = plt.subplots()

    for rate, (corr, p_val) in ranked[:n_rates]:
        x = np.array([deviates[rate][n] for n in nums])
        ax.scatter(x, y, label=f"{rate} (ρ={corr:.3f}, p={p_val:.3f})",
                   alpha=0.7)

    if do_fit:
        rate = ranked[0][0]
        x = np.array([deviates[rate][n] for n in nums])
        m, b = np.polyfit(x, y, deg=1)

        x_lin = np.linspace(-3, 3)
        ax.plot(x_lin, m*x_lin + b, label=f"Linear fit for {rate}")

    ax.set_xlabel("deviate")
    ax.set_ylabel("normalized shock speed (z)")
    ax.set_title(f"Top {n_rates} correlated rates")

    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=2,
        fontsize="small",
        frameon=True,
    )

    fig.text(0.5, 0.0, f"Shock speed (cm/s): {median:.3e} + z * {stdev:.3e}",
             ha="center", va="top", fontsize=9)

    fig.savefig("top_rates.png", bbox_inches="tight",
                bbox_extra_artists=(legend,))
    plt.close(fig)

if __name__ == "__main__":

    print(boldline)
    print("CORRELATION ANALYSIS for starlib deviates and shock speeds")
    print(boldline)

    p = argparse.ArgumentParser()

    p.add_argument("directories", type=str, nargs="+",
                    help="directories holding runs")
    p.add_argument("--do_plot", type=int, default=0,
                   help="Plot and fit n-most correlated rates")
    p.add_argument("--do_fit", action="store_true",
                   help="Does a linear fit on most correlated rate")

    args = p.parse_args()

    runs = read_directories(args.directories)
    norm_speeds, deviates, p_corrs, s_corrs, median, stdev= analysis(runs)

    if args.do_plot > 0:
        plot(args.do_plot, args.do_fit, norm_speeds, deviates, s_corrs, median, stdev)
