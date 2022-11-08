

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
from sklearn import metrics
from sklearn.linear_model import LinearRegression, LogisticRegression


import wandb


def get_test_labels(dset, val = False):
    split_map = {
        "train": 0,
        "valid": 1,
        "test": 2,
    }
    metadata = pd.read_csv("../iht-sparse/waterbird_complete95_forest2water2/metadata.csv")
    if val:
        labels = metadata[metadata.split==1]["y"].values
    else:
        labels = metadata[metadata.split==2]["y"].values
    return labels>0

def get_test_locations(dset, val = False):
    split_map = {
        "train": 0,
        "valid": 1,
        "test": 2,
    }
    metadata = pd.read_csv("../iht-sparse/waterbird_complete95_forest2water2/metadata.csv")
    if val:
        labels = metadata[metadata.split==1]["place"].values
    else:
        labels = metadata[metadata.split==2]["place"].values
    return labels>0


def get_run_dir(run):
    def trim_last(string):
        if string[-1] == "'":
            return string[:-1]
        return string
    run_dir = [x[len(" run_dir='"):-1]  for x in run.config["params"].split(",")  if 'run_dir' in x][0]
    return trim_last(run_dir)

def get_max_epoch(run):
    try:
        return run.summary["epoch"]
    except:
        return 0

def sparsity(run):
    name = run["group"]
    if "s50" in name:
        return 50
    elif "s75" in name:
        return 75
    elif "s80" in name:
        return 80
    elif "s90" in name:
        return 90
    elif "s95" in name:
        return 95
    elif "s98" in name or "s9r89" in name:
        return 98
    elif "s995" in name:
        return 99.5
    elif "s99" in name:
        return 99
    elif "s0" in name or "dense" in name:
        return 0
    else:
        raise ValueError(f"unknown sparsity for {name}!")

def sparsity_group(sparsity):
    sparsities = [0, 80, 90, 95, 98, 99, 99.5]
    return sparsities.index(sparsity)
        
def strategy(run):
    name = run["group"]
    if "post" in name:
        return "GMP-PT"
    elif "gmp" in name:
        return "GMP-RI"
    else:
        return "Dense"

def desc(run):
    return f"{sparsity(run)}-{strategy(run)}"
    


def get_runs_for_project(project = "waterbirds-rn18"):

    api = wandb.Api()
    api.flush()
    runs = api.runs(f"ist/{PROJECTS[project]['wandb_project']}")

    def get_timestamp(run):
        try:
            return run.summary["_timestamp"]
        except:
            return -1

    preprocessed_rn18_runs = [{"group": run.group, "job_type": run.job_type,
                          "name": run.name, "state": run.state, "url": run.url, "id":run.id,
                          "dataset": PROJECTS[project]["dset"],
                         "run_dir": get_run_dir(run),
                         "epoch": get_max_epoch(run),
                         "username": run.user._attrs["username"],
                         "timestamp": get_timestamp(run)} for run in runs
                        # if ("0505norms" in run.group and not "young" in run.group
                        # and not "blond" in run.group and not "smiling" in run.group
                        # and not 's0_adam' in run.group
                        # and("gmps" in run.group or "s0" in run.group))
                        # or ("full_celeba" in project)
                        # or ("awa" in project)
                        ]



    preprocessed_rn18_runs = [v for v in preprocessed_rn18_runs if v["epoch"]>10]
    for run in preprocessed_rn18_runs:
        if run["username"] == 'alexp':
            run["run_dir"] = f"/nfs/scistore14/alistgrp/epeste/iht-sparse-bias-celeba/{run['run_dir']}"
        else:
            run["run_dir"] = f"/nfs/scistore14/alistgrp/eiofinov/iht-sparse/{run['run_dir']}"



    for run in preprocessed_rn18_runs:
        dn = run["run_dir"]
        ckpt_name = "last_checkpoint.ckpt"
        if "s0" in run["name"] or "dense" in run["name"]:
            ckpt_name = "best_dense_checkpoint.ckpt"
        #! mkdir -p {dn}
        #! scp das8gpu1:{"~/iht-sparse/" + run["run_dir"] + "/" + ckpt_name} {dn}
        run["ckpt_path"] = dn + "/" + ckpt_name
        run["sparsity"] = sparsity(run)
        run["strategy"] = strategy(run)
        run['type'] = desc(run)

    # Filter out duplicate runs
    typed_runs = {}
    for run in preprocessed_rn18_runs:
        if run["type"] not in typed_runs:
            typed_runs[run["type"]] = {}
        if run["name"] not in typed_runs[run["type"]]:
            typed_runs[run["type"]][run["name"]] = run
        elif typed_runs[run["type"]][run["name"]]["timestamp"] < run["timestamp"]:
            typed_runs[run["type"]][run["name"]] = run

    preprocessed_rn18_runs = [r for h in typed_runs.values() for r in h.values() ]
    #raise ValueError([r["group"] for r in preprocessed_rn18_runs])


    return preprocessed_rn18_runs




def get_run_summaries(preprocessed_rn18_runs, arch='resnet18', threshold_adjusted=0):
    dataset = preprocessed_rn18_runs[0]["dataset"]
    covs_df, pos_fracs_df, neg_fracs_df = compute_cooccurrence_matrices(dataset)
    if 'celeba' in dataset:
        identity_labels = celeba_identity_labels
        attr_names = celeba_classes()
    else:
        identity_labels = awa_identity_labels
        attr_names = awa_classes()
    #attr_name = attr_names[attr]
    test_labels = get_test_labels(dataset)
    print("there are this many runs before compute", len(preprocessed_rn18_runs))
    preprocessed_rn18_runs = [load_run_details(run, pos_fracs_df, neg_fracs_df, threshold_adjusted) for run in preprocessed_rn18_runs]
    
    preprocessed_rn18_runs = [v for v in preprocessed_rn18_runs if 'strategy' in v]
    print("there are this many runs", len(preprocessed_rn18_runs), preprocessed_rn18_runs[0].keys())
    preprocessed_rn18_runs.sort(key=lambda r: (r["sparsity"], r["group"]))

   
    # AUC
    highlevels = []
    dicts = []
    for run in preprocessed_rn18_runs:
        if "test_outputs" not in run:
            print(f"!!!! Dropping run, {run['run_dir']}")
            continue
        mydict = {"seed": run["name"], "type": run["type"]}
        for i, attr_name in enumerate(attr_names):
            label = attr_name
            mydict[label] = run[f"auc"][i]
        dicts.append(mydict)
        
    df = pd.DataFrame.from_dict(dicts)
    averages_df = df.groupby("type").mean()
    high_level_auc_df = averages_df.transpose()
    highlevels.append(["auc", high_level_auc_df])
    top_level_aucs = pd.DataFrame(df.groupby("type").mean().transpose().mean())
    top_level_aucs.columns = ["AUC"]
    

    dicts = []
    for run in preprocessed_rn18_runs:
        if "test_outputs" not in run:
            continue
        mydict = {"seed": run["name"], "type": run["type"]}
        for i, attr_name in enumerate(attr_names):
            label = attr_name
            mydict[label] = run[f"acc"][i]
        dicts.append(mydict)
        
    df = pd.DataFrame.from_dict(dicts)
    averages_df = df.groupby("type").mean()
    high_level_acc_df = averages_df.transpose()
    highlevels.append(["acc", high_level_acc_df])
    top_level_accuracies = pd.DataFrame(df.groupby("type").mean().transpose().mean())
    top_level_accuracies.columns = ["Accuracy"]

    dicts = []
    for run in preprocessed_rn18_runs:
        if "test_outputs" not in run:
            continue
        mydict = {"seed": run["name"], "type": run["type"]}
        for i, attr_name in enumerate(attr_names):
            label = attr_name
            mydict[label] = run[f"fpr"][i]
        dicts.append(mydict)
        
    df = pd.DataFrame.from_dict(dicts)
    
    averages_df = df.groupby("type").mean()
    high_level_acc_df = averages_df.transpose()
    highlevels.append(["fpr", high_level_acc_df])


    dicts = []
    for run in preprocessed_rn18_runs:
        if "test_outputs" not in run:
            continue
        mydict = {"seed": run["name"], "type": run["type"]}
        for i, attr_name in enumerate(attr_names):
            label = attr_name
            mydict[label] = run[f"fnr"][i]
        dicts.append(mydict)
        
    df = pd.DataFrame.from_dict(dicts)
    
    averages_df = df.groupby("type").mean()
    #averages_df.reset_index(inplace=True)
    high_level_acc_df = averages_df.transpose()
    highlevels.append(["fnr", high_level_acc_df])

    dicts = []
    for run in preprocessed_rn18_runs:
        if "test_outputs" not in run:
            continue
        mydict = {"seed": run["name"], "type": run["type"]}
        for i, attr_name in enumerate(attr_names):
            label = attr_name
            mydict[label] = run[f"predpos"][i]
        dicts.append(mydict)
        
    df = pd.DataFrame.from_dict(dicts)
    averages_df = df.groupby("type").mean()
    high_level_predpos_df = averages_df.transpose()
    highlevels.append(["predpos", high_level_predpos_df])
    top_level_predposs = pd.DataFrame(df.groupby("type").mean().transpose().mean())
    top_level_predposs.columns = ["Predicted Positive %"]


    dicts = []
    for run in preprocessed_rn18_runs:
        if "test_outputs" not in run:
            continue
        mydict = {"seed": run["name"], "type": run["type"]}
        for i, attr_name in enumerate(attr_names):
            label = attr_name
            mydict[label] = run[f"uncertainty"][i]
        dicts.append(mydict)
        
    df = pd.DataFrame.from_dict(dicts)
    averages_df = df.groupby("type").mean()
    high_level_uncertainty_df = averages_df.transpose()
    highlevels.append(["uncertainty", high_level_uncertainty_df])
    top_level_uncertainties = pd.DataFrame(df.groupby("type").mean().transpose().mean())
    top_level_uncertainties.columns = ["Uncertain %"]

    dicts = []
    for run in preprocessed_rn18_runs:
        if "test_outputs" not in run:
            continue
        mydict = {"seed": run["name"], "type": run["type"]}
        for i, attr_name in enumerate(attr_names):
            label = attr_name
            mydict[label] = run[f"high_uncertainty"][i]
        dicts.append(mydict)
        
    df = pd.DataFrame.from_dict(dicts)
    averages_df = df.groupby("type").mean()
    high_level_high_uncertainty_df = averages_df.transpose()
    highlevels.append(["high_uncertainty", high_level_high_uncertainty_df])
    top_level_high_uncertainties = pd.DataFrame(df.groupby("type").mean().transpose().mean())
    top_level_high_uncertainties.columns = ["High Uncertain %"]




    ba_dicts = {}
    for identity_label in identity_labels:
        identity_label_name = attr_names[identity_label]
        dicts = []
        for run in preprocessed_rn18_runs:
            if "test_outputs" not in run:
                continue
            mydict = {"seed": run["name"], "type": run["type"], "strategy": run["strategy"], "sparsity": run["sparsity"]}
            for i, attr_name in enumerate(attr_names):
                label = attr_name
                mydict[label] = run[f"{identity_label_name}-bas"][i]
            dicts.append(mydict)
            
            
        df = pd.DataFrame.from_dict(dicts)
        
        averages_df = df.groupby(["strategy", "sparsity"]).mean()
        averages_df.reset_index(inplace=True)
        
        cols = [c for c in averages_df.columns if c.startswith(identity_label_name)]
        averages_diffs_df = averages_df[cols] / averages_df[cols].iloc[0]
        ba_dicts[identity_label_name] = [df, averages_df, averages_diffs_df]  


    fpr_dicts = {}
    for identity_label in identity_labels:
        identity_label_name = attr_names[identity_label]
        dicts = []
        for run in preprocessed_rn18_runs:
            if "test_outputs" not in run:
                continue
            mydict = {"seed": run["name"], "type": run["type"], "strategy": run["strategy"], "sparsity": run["sparsity"]}
            for i, attr_name in enumerate(attr_names):
                label = attr_name
                mydict[label] = run[f"{identity_label_name}-fpr_diff"][i]
            dicts.append(mydict)
            
            
        df = pd.DataFrame.from_dict(dicts)
        
        averages_df = df.groupby(["strategy", "sparsity"]).mean()
        averages_df.reset_index(inplace=True)
        
        cols = [c for c in averages_df.columns if c.startswith(identity_label_name)]
        averages_diffs_df = averages_df[cols] / averages_df[cols].iloc[0]
        fpr_dicts[identity_label_name] = [df, averages_df, averages_diffs_df]  


    fnr_dicts = {}
    for identity_label in identity_labels:
        identity_label_name = attr_names[identity_label]
        dicts = []
        for run in preprocessed_rn18_runs:
            if "test_outputs" not in run:
                continue
            mydict = {"seed": run["name"], "type": run["type"], "strategy": run["strategy"], "sparsity": run["sparsity"]}
            for i, attr_name in enumerate(attr_names):
                label = attr_name
                mydict[label] = run[f"{identity_label_name}-fnr_diff"][i]
            dicts.append(mydict)
            
            
        df = pd.DataFrame.from_dict(dicts)
        
        averages_df = df.groupby(["strategy", "sparsity"]).mean()
        averages_df.reset_index(inplace=True)
        
        cols = [c for c in averages_df.columns if c.startswith(identity_label_name)]
        averages_diffs_df = averages_df[cols] / averages_df[cols].iloc[0]
        fnr_dicts[identity_label_name] = [df, averages_df, averages_diffs_df]  

    combined_df = pd.merge(top_level_aucs, top_level_accuracies, left_index=True, right_index=True)
    combined_df = pd.merge(combined_df, top_level_predposs, left_index=True, right_index=True)
    combined_df = pd.merge(combined_df, top_level_uncertainties, left_index=True, right_index=True)
    combined_df = pd.merge(combined_df, top_level_high_uncertainties, left_index=True, right_index=True)
    ta_suffix=""
    if threshold_adjusted:
        ta_suffix="_threshold_adjusted"
    os.makedirs(os.path.join("generated", "tables"), exist_ok=True)
    filepath = os.path.join("generated", "tables", f"top_level_{dataset}_{arch}{ta_suffix}.tex")
    filepath = filepath.replace(" ", "-")
    combined_df.round(decimals=2).to_latex(filepath)
    return top_level_accuracies, highlevels, ba_dicts, fpr_dicts, fnr_dicts

