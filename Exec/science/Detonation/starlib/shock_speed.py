#!/usr/bin/env python

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import yt
import numpy as np

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

def shock_speed(plotfiles, trim=True):
    # By default, clip initial plotfiles to exlude the initial spike 
    if trim:
        cond = len(plotfiles) // 10
        plotfiles = plotfiles[cond:]

    t = []
    x = []

    for p in plotfiles:
        time, x_grid, T = get_T_profile(p)
        t.append(time)
        x.append(find_x_for_T(x_grid, T))

    t = np.asarray(t)
    x = np.asarray(x)

    popt, pcov = np.polyfit(t, x, 1, cov=True)

    v = popt[0]
    v_err = np.sqrt(pcov[0, 0])

    return t, x, (v, v_err)

def parse_info_txt(info_txt_path):
    "Parses an info.txt file in to a dict which maps the run's local index to (seed, status)"
    info = {}
    with open(info_txt_path) as f:
        for line in f:
            line = line.strip()

            #skip over any empty lines
            if not line:
                continue

            if line.startswith("Median Run"):
                status = line.split("STATUS:")[1].strip().split()[0]
                info["median"] = (-1, status)

            elif line.startswith("Run"):
                header, status = line.split(",")

                #read header
                run_num = header.split(":")[0].split()[1]
                seed = int(header.split(":")[1].strip())
                #read status
                status = status.split(":")[1].strip().split()

                # if status is empty then there this run is still running,
                if not status:
                    status =  "PENDING"
                else:
                    status = status[0]

                info[run_num] = (seed, status)

    return info

def read_directories(directories):
    runs = {}
    idx = 1
    median_seen = False
    seen_seeds = set()

    #list directories provided
    print(f"Total directories provided: {len(directories)}")
    print(slimline)
    print(f"Extracting runs and plotfiles ...")

    for dir_path in directories:

        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise ValueError(f"{dir_path} is not a directory")

        #find this dir's info file
        info_path = dir_path / "summary.txt"
        if not info_path.is_file():
            raise ValueError(f"{info_path} not found")
        info = parse_info_txt(info_path)

        for run_dir in sorted(dir_path.glob("run_*")):
            #Ensure that none of the log files enter the dict for runs
            if not run_dir.is_dir():
                continue

            orig_prefix = run_dir.name.split("_")[1]

            #We should be able to find this orig_prefix in info as well
            if orig_prefix not in info:
                print(f"Warning: {run_dir} has no entry in {info_path}, skipping")
                continue
            seed, status = info[orig_prefix]

            #skip over failed runs
            if status != "SUCCESS":
                print(f"Skipping {run_dir} (status: {status})")
                continue

            #Avoid having several median cases
            if orig_prefix == "median":
                if median_seen:
                    continue
                median_seen = True
                new_prefix = "median"
            else:
                if seed in seen_seeds:
                    print(f"Skipping {run_dir}, duplicate seed: {seed}")
                    continue
                seen_seeds.add(seed)
                new_prefix = str(idx)
                idx += 1

            plotfiles = sorted(run_dir.glob("det_x_plt*"))
            runs[new_prefix] = plotfiles

    print(f"Total of {len(runs) - 1} + 1 runs established")
    print(slimline)

    return runs

if __name__ == "__main__":

    p = argparse.ArgumentParser()

    p.add_argument("directories", type=str, nargs="+",
                    help="directories holding runs")

    args = p.parse_args()
    runs = read_directories(args.directories)

    fig, ax = plt.subplots()

    for prefix, plotfiles in runs.items():
        t, x, v = shock_speed(plotfiles)

        if prefix == "median":
            ax.plot(t, x, label="median", color='k', zorder=1)
        else:
            ax.plot(t, x, color='b', zorder=0, alpha=0.2)

        print(f"Shock Speed for run {prefix}: {v[0]:.3e} +- {v[1]:.3e}")

    ax.set_xlabel("time (s)")
    ax.set_ylabel("pos (cm)")
    ax.set_title("Shock speed")

    fig.savefig("shock_position.png", bbox_inches="tight")
    plt.close()