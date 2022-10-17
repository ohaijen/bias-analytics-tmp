import os

import importlib
#import torch

import matplotlib.pyplot as plt
import seaborn as sns
import math

import numpy as np
import pandas as pd
import io
import base64
import pickle as pkl
from .projects import PROJECTS


import wandb



def get_run_dir(run):
    run_dir = [x[len(" run_dir='"):-1]  for x in run.config["params"].split(",")  if 'run_dir' in x][0]
    run_dir = os.path.join("/nfs/scistore14/alistgrp/eiofinov/iht-sparse-wilds/", run_dir)
    return run_dir

def get_max_epoch(run):
    try:
        return run.summary["epoch"]
    except:
        return 0
    
def get_status(run):
    try:
        run.status
    except:
        return ""
    return run.status

#groups = [0, 1, 10, 50, 150]
groups = [ 5, 10, 50, 150]
def assign_group(count):
    #if count == 0:
    #    return 0
    for i, threshold in enumerate(groups):
        if count <= threshold:
            return threshold
    return 100000


def get_iwildcam_runs():
    api = wandb.Api()
    api.flush()
    runs = api.runs("ist/wilds_bias")


    preprocessed_wilds_runs = [{"group": run.group, "job_type": run.job_type,
      "name": run.name, "state": run.state, "url": run.url, "id":run.id,
                           "run_dir": get_run_dir(run),
                                                "epoch": get_max_epoch(run)} for run in runs
        if run.group.startswith("iwildcam-") and ('gmp' in run.group or "dense" in run.group)
                        ]

    preprocessed_wilds_runs = [v for v in preprocessed_wilds_runs if v["epoch"]>10]

    loader_names = ["test", "ood_test"]
    for run in preprocessed_wilds_runs:
        for loader_name in loader_names:
            remote_run_dir = os.path.join("iht-sparse-wilds", run["run_dir"])
            print(remote_run_dir)
            run[f"{loader_name}_f1_scores"] = np.loadtxt(run["run_dir"]  + f"/{loader_name}_f1_score.txt")
            run[f"{loader_name}_precisions"] = np.loadtxt(run["run_dir"]  + f"/{loader_name}_precision.txt")
            run[f"{loader_name}_recalls"] = np.loadtxt(run["run_dir"]  + f"/{loader_name}_recall.txt")
            run[f"{loader_name}_per_class_counts"] = np.loadtxt(run["run_dir"]  + f"/{loader_name}_per_class_counts.txt")
        run[f"class_id"] = np.arange(len(run["test_per_class_counts"]))


    df = pd.concat([pd.DataFrame(x) for x in preprocessed_wilds_runs])
    df["test_group"] = df["test_per_class_counts"].map(assign_group)
    df["ood_test_group"] = df["ood_test_per_class_counts"].map(assign_group)
    return df



def plot_wilds_metric_by_threshold(runs_df, metric, loader, run_prefix, ax):
    runs_df = runs_df[runs_df[f"{loader}_group"]> 0]
    runs_df = runs_df.groupby(["job_type", "group", f"{loader}_group"]).agg("mean")
    runs_df.reset_index(inplace=True)
    sns.barplot(data = runs_df, x=f"{loader}_group", y=f"{loader}_{metric}", hue="group", ax=ax)

def plot_wilds_metrics(runs):
    img = io.BytesIO()
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    for i, metric in enumerate(["f1_scores", "precisions", "recalls"]):
        plot_wilds_metric_by_threshold(runs,
                                       metric, "test", "iwildcam-gmp", axs[i])
    fig.suptitle("ID Test Performance")
    fig.tight_layout()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.clf()
    id_url = base64.b64encode(img.getvalue()).decode()

    img = io.BytesIO()
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    for i, metric in enumerate(["f1_scores", "precisions", "recalls"]):
        plot_wilds_metric_by_threshold(runs,
                                       metric, "test", "iwildcam-gmp", axs[i])
    fig.suptitle("OOD Test Performance")
    fig.tight_layout()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.clf()
    ood_url = base64.b64encode(img.getvalue()).decode()
    return id_url, ood_url

    # grouped_runs = {}
    # fig, axs = plt.subplots(figsize=(8,6))
    # plot_wilds_metric_by_threshold(preprocessed_wilds_runs,
    #                                "test_f1_scores", "test", "iwildcam-best", axs)
    # fig.suptitle("F1 score change relative to dense models for models pruned on iWildcam, ID")
    # fig.tight_layout()

    # grouped_runs = {}
    # fig, axs = plt.subplots(figsize=(8,6))
    # plot_wilds_metric_by_threshold(preprocessed_wilds_runs,
    #                                "ood_test_f1_scores", "ood_test", "iwildcam-best", axs)
    # fig.suptitle("F1 score change relative to dense models for models pruned on iWildcam, OOD")
    # fig.tight_layout()
