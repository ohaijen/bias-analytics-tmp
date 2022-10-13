
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

def celeba_classes():
    return ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"]

def awa_classes():
    return ['black', 'white', 'blue', 'brown', 'gray', 'orange', 'red',
'yellow', 'patches', 'spots', 'stripes', 'furry', 'hairless',
'toughskin', 'big', 'small', 'bulbous', 'lean', 'flippers',
'hands', 'hooves', 'pads', 'paws', 'longleg', 'longneck', 'tail',
'chewteeth', 'meatteeth', 'buckteeth', 'strainteeth', 'horns',
'claws', 'tusks', 'smelly', 'flys', 'hops', 'swims', 'tunnels',
'walks', 'fast', 'slow', 'strong', 'weak', 'muscle', 'bipedal',
'quadrapedal', 'active', 'inactive', 'nocturnal', 'hibernate',
'agility', 'fish', 'meat', 'plankton', 'vegetation', 'insects',
'forager', 'grazer', 'hunter', 'scavenger', 'skimmer', 'stalker',
'newworld', 'oldworld', 'arctic', 'coastal', 'desert', 'bush',
'plains', 'forest', 'fields', 'jungle', 'mountains', 'ocean',
'ground', 'water', 'tree', 'cave', 'fierce', 'timid', 'smart',
'group', 'solitary', 'nestspot', 'domestic']

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


def get_test_labels(dset):
    split_map = {
        "train": 0,
        "valid": 1,
        "test": 2,
    }
    if 'celeba' in dset:
        splits = pd.read_csv("/home/Datasets/celeba/list_eval_partition.txt", delim_whitespace=True, header=None, index_col=0).to_numpy()
        labels = pd.read_csv("/home/Datasets/celeba/list_attr_celeba.txt", delim_whitespace=True, header=1).to_numpy()
        labels = labels[ splits.ravel()==2]
        return labels>0
    elif 'awa' in dset:
        predicates_file = os.path.join("/home/Datasets/Animals_with_Attributes2/", "predicate-matrix-binary.txt")
        predicates_mtx = np.loadtxt(predicates_file)
        classes = pd.read_csv("/home/Datasets/Animals_with_Attributes2/classes.txt", sep="\t", header=None)
        classes.columns = ["id", "klass"]
        ccs = classes.reset_index().set_index("klass")
        ood_image_dir = os.path.join("/home/Datasets/Animals_with_Attributes2/", "ood_images")
        ood_classes = os.listdir(ood_image_dir)
        ood_classes.sort()
        ood_predicates_mtx = predicates_mtx[ccs.loc[ood_classes]["index"].values]

        images_folder = "/home/Datasets/Animals_with_Attributes2/ood_images/"
        images_subfolders = os.listdir(images_folder)
        images_subfolders.sort()
        class_attributes = []
        class_sizes = [len(os.listdir(os.path.join(images_folder, sf))) for sf in images_subfolders]
        labels = np.zeros([np.sum(class_sizes), 85])
        print("label shape is ", labels.shape)
        offset = 0
        for i, species in enumerate(images_subfolders):
            species_idx = np.where(classes["klass"].values == species)[0].ravel()[0]
            attrs = predicates_mtx[species_idx]
            labels[offset:offset+class_sizes[i]] = attrs #np.stack(attrs, class_sizes[i], axis=1)
            offset = offset + class_sizes[i]
        print("labels are", labels)
        return labels


def get_test_image_ids(dset):
    split_map = {
        "train": 0,
        "valid": 1,
        "test": 2,
    }
    if 'celeba' in dset:
        splits = pd.read_csv("/home/Datasets/celeba/list_eval_partition.txt", delim_whitespace=True, header=None)
        splits.columns = ["img_ids", "split"]
        img_ids  = splits["img_ids"].to_numpy().ravel()[splits['split'].ravel()==2]
        return img_ids
    elif 'awa' in dset:
        img_ids = []
        ood_image_dir = os.path.join("/home/Datasets/Animals_with_Attributes2/", "ood_images")
        ood_classes = os.listdir(ood_image_dir)
        ood_classes.sort()
        for k in ood_classes:
            image_urls = os.path.join("/home/Datasets/Animals_with_Attributes2/", "ood_images", k)
            image_urls.sort()
            img_ids.extend(image_urls)
        return img_ids



celeba_identity_labels = [20, 39, 13, 26]
awa_identity_labels = [75, 84, 44, 11]#63, 44]


def compute_bias_amplification(targets, predictions, protected_attribute_id, attribute_id, dataset):
    if 'celeba' in dataset:
        attr_names = celeba_classes()
    else:
        attr_names = awa_classes()
    if attribute_id == protected_attribute_id:
        return None
    protected_attr = targets[:,protected_attribute_id].ravel()
    protected_pos = np.argwhere(protected_attr == 1)
    if targets[protected_pos, attribute_id].sum() < 10:
        print(f"too few positive examples for attribute {attr_names[attribute_id]}")
        print("predictions", predictions[protected_pos, attribute_id].sum())
        return None
    
    protected_neg = np.argwhere(protected_attr == 0)
    if targets[protected_neg, attribute_id].sum() < 10:
        print(f"too few negative examples for attribute {attr_names[attribute_id]}")
        print("predictions", predictions[protected_neg, attribute_id].sum())
        return None
    protected_pos_predicts = predictions[protected_pos, attribute_id] 
    protected_neg_predicts = predictions[protected_neg, attribute_id]
    protected_pos_targets = targets[protected_pos, attribute_id] 
    protected_neg_targets = targets[protected_neg, attribute_id]
  
    if np.abs(protected_pos_targets.mean()-protected_neg_targets.mean())/ \
            np.minimum(protected_pos_targets.mean(), protected_neg_targets.mean()) < 0.1:
        print(f"Diff is too small for attribute {attr_names[attribute_id]}")
        return None
    if protected_pos_targets.mean() > protected_neg_targets.mean():
        ba = protected_pos_predicts.sum()/predictions[:,attribute_id].sum() - \
             protected_pos_targets.sum()/targets[:, attribute_id].sum()
    else:
        ba = protected_neg_predicts.sum()/(predictions[:,attribute_id]).sum() - \
             protected_neg_targets.sum()/(targets[:, attribute_id]).sum()
    return ba

def compute_fpr_split(targets, predictions, protected_attribute_id, attribute_id):
    protected_attr = targets[:,protected_attribute_id].ravel()
    protected_pos = np.argwhere(protected_attr == 1)
    protected_neg = np.argwhere(protected_attr == 0)
    protected_pos_predicts = predictions[protected_pos, attribute_id] 
    protected_neg_predicts = predictions[protected_neg, attribute_id]
    protected_pos_targets = targets[protected_pos, attribute_id] 
    protected_neg_targets = targets[protected_neg, attribute_id]
    pos_fpr = np.sum(protected_pos_predicts*(1-protected_pos_targets))/np.sum(1-protected_pos_targets)
    neg_fpr = np.sum(protected_neg_predicts*(1-protected_neg_targets))/np.sum(1-protected_neg_targets)
    return pos_fpr, neg_fpr


def compute_fnr_split(targets, predictions, protected_attribute_id, attribute_id):
    protected_attr = targets[:,protected_attribute_id].ravel()
    protected_pos = np.argwhere(protected_attr == 1)
    protected_neg = np.argwhere(protected_attr == 0)
    protected_pos_predicts = predictions[protected_pos, attribute_id] 
    protected_neg_predicts = predictions[protected_neg, attribute_id]
    protected_pos_targets = targets[protected_pos, attribute_id] 
    protected_neg_targets = targets[protected_neg, attribute_id]
    pos_fnr = np.sum((1-protected_pos_predicts)*(protected_pos_targets))/np.sum(protected_pos_targets)
    neg_fnr = np.sum((1-protected_neg_predicts)*(protected_neg_targets))/np.sum(protected_neg_targets)
    return pos_fnr, neg_fnr

def compute_acc_split(targets, predictions, protected_attribute_id, attribute_id):
    protected_attr = targets[:,protected_attribute_id].ravel()
    protected_pos = np.argwhere(protected_attr == 1)
    protected_neg = np.argwhere(protected_attr == 0)
    protected_pos_predicts = predictions[protected_pos, attribute_id] 
    protected_neg_predicts = predictions[protected_neg, attribute_id]
    protected_pos_targets = targets[protected_pos, attribute_id] 
    protected_neg_targets = targets[protected_neg, attribute_id]
    pos_acc = np.mean(protected_pos_predicts==protected_pos_targets)
    neg_acc = np.mean(protected_neg_predicts==protected_neg_targets)
    return pos_acc, neg_acc

def compute_error_split(targets, predictions, protected_attribute_id, attribute_id):
    protected_attr = targets[:,protected_attribute_id].ravel()
    protected_pos = np.argwhere(protected_attr == 1)
    protected_neg = np.argwhere(protected_attr == 0)
    protected_pos_predicts = predictions[protected_pos, attribute_id] 
    protected_neg_predicts = predictions[protected_neg, attribute_id]
    protected_pos_targets = targets[protected_pos, attribute_id] 
    protected_neg_targets = targets[protected_neg, attribute_id]
    pos_err = np.mean(protected_pos_predicts!=protected_pos_targets)
    neg_err = np.mean(protected_neg_predicts!=protected_neg_targets)
    return pos_err, neg_err

def compute_bas(run, targets):
    if 'celeba' in run["dataset"]:
        identity_labels = celeba_identity_labels
        attr_names = celeba_classes()
    else:
        identity_labels = awa_identity_labels
        attr_names = awa_classes()
    for identity_label in identity_labels:
        identity_label_name = attr_names[identity_label]
        # bas_path = os.path.join(run["run_dir"] , f"{identity_label_name}-bas.txt")
        # if os.path.exists(bas_path):
        #     run[f"{identity_label_name}-bas"] = np.loadtxt(bas_path)
        #     continue
        run[f"{identity_label_name}-bas"] = np.zeros(len(attr_names))
        run[f"{identity_label_name}-bas"][:] = np.nan
        for i in range(len(attr_names)):
            label = attr_names[i]
            if i == identity_label:
                continue
            run[f"{identity_label_name}-bas"][i] = \
            compute_bias_amplification(targets, run["test_predictions"], identity_label, i, run["dataset"])
        #np.savetxt(bas_path, run[f"{identity_label_name}-bas"])
            
def compute_errors(run, targets):
    if 'celeba' in run["dataset"]:
        identity_labels = celeba_identity_labels
        attr_names = celeba_classes()
    else:
        identity_labels = awa_identity_labels
        attr_names = awa_classes()
    fpr = np.sum(run["test_predictions"]*(1-targets), axis=0)/np.sum(1-targets, axis=0)
    fnr = np.sum((1-run["test_predictions"])*(targets), axis=0)/np.sum(targets, axis=0)
    acc = np.mean(run["test_predictions"]==targets, axis=0)
    run["fpr"] = fpr
    run["fnr"] = fnr
    run["acc"] = acc
    for identity_label in identity_labels:
        identity_label_name = attr_names[identity_label]
        # Initialize all arrays with NaNs
        run[f"{identity_label_name}-pos_fpr"] = np.zeros(len(attr_names))
        run[f"{identity_label_name}-pos_fpr"][:] = np.nan
        run[f"{identity_label_name}-neg_fpr"] = np.zeros(len(attr_names))
        run[f"{identity_label_name}-neg_fpr"][:] = np.nan
        run[f"{identity_label_name}-pos_fnr"] = np.zeros(len(attr_names))
        run[f"{identity_label_name}-pos_fnr"][:] = np.nan
        run[f"{identity_label_name}-neg_fnr"] = np.zeros(len(attr_names))
        run[f"{identity_label_name}-neg_fnr"][:] = np.nan
        run[f"{identity_label_name}-pos_acc"] = np.zeros(len(attr_names))
        run[f"{identity_label_name}-pos_acc"][:] = np.nan
        run[f"{identity_label_name}-neg_acc"] = np.zeros(len(attr_names))
        run[f"{identity_label_name}-neg_acc"][:] = np.nan
        run[f"{identity_label_name}-pos_err"] = np.zeros(len(attr_names))
        run[f"{identity_label_name}-pos_err"][:] = np.nan
        run[f"{identity_label_name}-neg_err"] = np.zeros(len(attr_names))
        run[f"{identity_label_name}-neg_err"][:] = np.nan
        run[f"{identity_label_name}-fpr_diff"] = np.zeros(len(attr_names))
        run[f"{identity_label_name}-fpr_diff"][:] = np.nan
        run[f"{identity_label_name}-fnr_diff"] = np.zeros(len(attr_names))
        run[f"{identity_label_name}-fnr_diff"][:] = np.nan
        run[f"{identity_label_name}-acc_diff"] = np.zeros(len(attr_names))
        run[f"{identity_label_name}-acc_diff"][:] = np.nan
        run[f"{identity_label_name}-err_diff"] = np.zeros(len(attr_names))
        run[f"{identity_label_name}-err_diff"][:] = np.nan
        for i in range(len(attr_names)):
            if i == identity_label:
                continue
            pos_fpr, neg_fpr = \
                compute_fpr_split(targets, run["test_predictions"], identity_label, i) 
            run[f"{identity_label_name}-pos_fpr"][i] = pos_fpr
            run[f"{identity_label_name}-neg_fpr"][i] = neg_fpr
            run[f"{identity_label_name}-fpr_diff"][i] = np.abs(pos_fpr-neg_fpr)
            pos_fnr, neg_fnr = \
                compute_fnr_split(targets, run["test_predictions"], identity_label, i) 
            run[f"{identity_label_name}-pos_fnr"][i] = pos_fnr
            run[f"{identity_label_name}-neg_fnr"][i] = neg_fnr
            run[f"{identity_label_name}-fnr_diff"][i] = np.abs(pos_fnr-neg_fnr)
            pos_acc, neg_acc = \
                compute_acc_split(targets, run["test_predictions"], identity_label, i) 
            run[f"{identity_label_name}-pos_acc"][i] = pos_acc
            run[f"{identity_label_name}-neg_acc"][i] = neg_acc
            run[f"{identity_label_name}-acc_diff"][i] = np.abs(pos_acc-neg_acc)
            pos_err, neg_err = \
                compute_error_split(targets, run["test_predictions"], identity_label, i) 
            run[f"{identity_label_name}-pos_err"][i] = pos_err
            run[f"{identity_label_name}-neg_err"][i] = neg_err
            run[f"{identity_label_name}-err_diff"][i] = np.abs(pos_err-neg_err)

# For compatibility with older versions of the notebook            
def compute_error_splits(run):
    compute_errors(run)
            
metrics = {
    "bas": "Bias Amplification Score",
    "fpr_diff": "False Positive Rate Difference",
    "fnr_diff": "False Negative Rate Difference",
    "acc_diff": "Accuracy Difference"
}

def plot_metric(runs, metric, attr, ax):
    dataset = runs[0]["dataset"]
    if 'celeba' in run["dataset"]:
        identity_labels = celeba_identity_labels
        attr_names = celeba_classes()
    else:
        identity_labels = awa_identity_labels
        attr_names = awa_classes()
    attr_name = attr_names[attr]
    grouped_runs = {}
    for r in runs:
        if r["strategy"]+str(r["sparsity"]) in grouped_runs:
            grouped_runs[r["strategy"]+str(r["sparsity"])].append(r[f"{attr_name}-{metric}"])
        else:
            grouped_runs[r["strategy"]+str(r["sparsity"])] = [r[f"{attr_name}-{metric}"]]
    outliers = []
    for i, (k, v) in enumerate(grouped_runs.items()):
        grouped_runs[k] = np.nanmean(v, axis=0)
        v = grouped_runs[k]
        q3 = np.nanquantile(grouped_runs[k], 0.75)
        q1 = np.nanquantile(grouped_runs[k], 0.25)
        outlier_hi = [[arg_names[x], v[x]] for x in np.argwhere(v > q3 + 1.5 * (q3-q1)).flatten()]
        outlier_lo = [[arg_names()[x], v[x]] for x in np.argwhere(v < q1 - 1.5 * (q3-q1)).flatten()]
        for l, vv in outlier_hi:
            outliers.append([i, l, vv])
        for l, vv in outlier_lo:
            outliers.append([i, l, vv])
    #for label, value in outlier_hi:
    #plt.text(value, 0, label, ha='left', va='center')
    y= np.array([v for v in grouped_runs.values()])
    x= np.array([np.tile(np.array(k[6:]), len(attr_names)) for k in grouped_runs.keys()])
    h= np.array([np.tile(np.array(k[4:6]), len(attr_names)) for k in grouped_runs.keys()])
    sns.boxplot(x.ravel(),y.ravel(), hue=h.ravel(), ax=ax).set_title(attr_name)
    #for x, l, vv in outliers:
    #    plt.text(x, vv, label, ha='left', va='center')
    
# This could be used to plot Bias Amplification relative to dense, but that plot is not very informative.
# Right now, it is not used.
def plot_relative_metric(runs, metric, attr, ax):
    dataset = runs[0]["dataset"]
    if 'celeba' in run["dataset"]:
        identity_labels = celeba_identity_labels
        attr_names = celeba_classes()
    else:
        identity_labels = awa_identity_labels
        attr_names = awa_classes()
    attr_name = attr_names[attr]
    grouped_runs = {}
    for r in runs:
        if r["strategy"]+str(r["sparsity"]) in grouped_runs:
            grouped_runs[r["strategy"]+str(r["sparsity"])].append(r[f"{attr_name}-{metric}"])
        else:
            grouped_runs[r["strategy"]+str(r["sparsity"])] = [r[f"{attr_name}-{metric}"]]
    
    for k, v in grouped_runs.items():
        grouped_runs[k] = np.nanmean(v, axis=0).clip(min=0)
    dense_run_key = '0'
    dense_run = grouped_runs[dense_run_key]
    for k, v in grouped_runs.items():
        grouped_runs[k] = np.array(v)/np.array(dense_run)
    y= np.array([v for v in grouped_runs.values()])
    x= np.array([np.tile(np.array(k[:3]), len(attr_names)) for k in grouped_runs.keys()])
    h= np.array([np.tile(np.array(k[4:]), len(attr_names)) for k in grouped_runs.keys()])
    sns.boxplot(x=x.ravel(),y=y.ravel(), hue=h.ravel(), ax=ax).set_title(attr_name)
    
# Function for creating plots of relative metric increase/decrease, faceted by the identity attribute
def facet_plot_relative_metric(runs, metric, attr, ax):
    dataset = runs[0]["dataset"]
    if 'celeba' in run["dataset"]:
        identity_labels = celeba_identity_labels
        attr_names = celeba_classes()
    else:
        identity_labels = awa_identity_labels
        attr_names = awa_classes()
    attr_name = attr_names[attr]
    grouped_runs = {}
    pos_metric = f'pos_{metric}'
    neg_metric = f'neg_{metric}'
    for r in runs:
        if r["strategy"]+str(r["sparsity"])+"pos" in grouped_runs:
            grouped_runs[r["strategy"]+str(r["sparsity"])+"pos"].append(r[f"{attr_name}-{pos_metric}"])
        else:
            grouped_runs[r["strategy"]+str(r["sparsity"])+"pos"] = [r[f"{attr_name}-{pos_metric}"]]
        if r["strategy"]+str(r["sparsity"])+"neg" in grouped_runs:
            grouped_runs[r["strategy"]+str(r["sparsity"])+"neg"].append(r[f"{attr_name}-{neg_metric}"])
        else:
            grouped_runs[r["strategy"]+str(r["sparsity"])+"neg"] = [r[f"{attr_name}-{neg_metric}"]]
    
    for i, (k, v) in enumerate(grouped_runs.items()):
        grouped_runs[k] = np.nanmean(v, axis=0).clip(min=0)
    dense_run_pos_key = '0pos'
    dense_run_neg_key = '0neg'
    dense_run_pos = grouped_runs[dense_run_pos_key]
    dense_run_neg = grouped_runs[dense_run_neg_key]
    for k, v in grouped_runs.items():
        if "pos" in k:
            grouped_runs[k] = np.array(v)/np.array(dense_run_pos)
        else:
            grouped_runs[k] = np.array(v)/np.array(dense_run_neg)
    y= np.array([v for v in grouped_runs.values()])
    xx= np.array([np.tile(np.array(k), len(attr_names)) for k in grouped_runs.keys()])
    x=np.array([x[:-3]  for x in xx.ravel()])
    h=np.array([x[-3:]  for x in xx.ravel()])
    g = sns.boxplot(x=x.ravel(), y=y.ravel(), hue=h.ravel(), ax=ax)
    g.set_title(attr_name)


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
    elif "s98" in name:
        return 98
    elif "s995" in name:
        return 995
    elif "s99" in name:
        return 99
    elif "s0" in name or "dense" in name:
        return 0
    else:
        raise f"unknown sparsity for {name}!"
        
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
    
# Table where the rows are the models and the columns are the BAs for young


def get_runs_for_project(project):

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
                        if ("0505norms" in run.group and not "young" in run.group
                        and not "blond" in run.group and not "smiling" in run.group
                        and not 's0_adam' in run.group
                        and("gmps" in run.group or "s0" in run.group))
                        or ("full_celeba" in project)
                        or ("awa" in project)
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


def get_run_counts(project):
    runs = get_runs_for_project(project)
    df = pd.DataFrame(runs)
    df = df.groupby(["sparsity", "strategy"]).agg({"name": "count"})
    df.columns = ["count"]
    df = df.reset_index()
    return df.pivot(columns=["strategy"], index=["sparsity"])




def generate_metric_plot(runs, metric_name = "Bias Amplification"):
    mdf = runs.copy()
    mdf.set_index(["strategy", "sparsity"], inplace=True)
    mdf = mdf.stack()
    mdf = pd.DataFrame(mdf)
    mdf.reset_index(inplace=True)
    mdf.columns = ["strategy", "sparsity", "attribute", metric_name]


    img = io.BytesIO()
    sns.boxplot(data = mdf,
        hue = 'strategy', 
        x = 'sparsity',
        y = metric_name,)
    plt.savefig(img, format='png')
    img.seek(0)
    plt.clf()  # Forced reset for matplotlib

    plot_url = base64.b64encode(img.getvalue()).decode()
    return(plot_url)

# def make_a_plot():
#     fig, axs = plt.subplots(2, 2, figsize=(8,6))
#     for i, idd in enumerate(identity_labels):
#         print([i//2, i % 2])
#         plot_metric(preprocessed_rn18_runs, "bas", idd, axs[i//2, i % 2])
#     fig.suptitle(f"{metrics['bas']}s, CelebA on ResNet18")
#     plt.tight_layout()

def load_run_details(run):
    test_labels = get_test_labels(run["dataset"])
    cached_path = os.path.join(run["run_dir"], "run_stats.pkl")
    if True  and os.path.exists(cached_path):
        with open (cached_path, 'rb') as f:
            # TODO: make sure the existing parts of the run match
            return pkl.load(f)
    elif os.path.exists(run["run_dir"] + "/" + "test_outputs.txt"):
        run["test_outputs"] = np.loadtxt(run["run_dir"] + "/" + "test_outputs.txt")
        run["test_predictions"] = run["test_outputs"] > 0
        run["sparsity"] = sparsity(run)
        run["strategy"] = strategy(run)
        run['type'] = desc(run)
        compute_bas(run, test_labels)
        compute_errors(run, test_labels)
        with open (cached_path, 'wb') as f:
            pkl.dump(run, f)
        return run
    else:
        return run

def get_run_summaries(preprocessed_rn18_runs):
    dataset = preprocessed_rn18_runs[0]["dataset"]
    if 'celeba' in dataset:
        identity_labels = celeba_identity_labels
        attr_names = celeba_classes()
    else:
        identity_labels = awa_identity_labels
        attr_names = awa_classes()
    #attr_name = attr_names[attr]
    test_labels = get_test_labels(dataset)
    print("there are this many runs before compute", len(preprocessed_rn18_runs))
    for i, run in enumerate(preprocessed_rn18_runs):
        cached_path = os.path.join(run["run_dir"], "run_stats.pkl")
        if False  and os.path.exists(cached_path):
            with open (cached_path, 'rb') as f:
                # TODO: make sure the existing parts of the run match
                preprocessed_rn18_runs[i] = pkl.load(f)
        elif os.path.exists(run["run_dir"] + "/" + "test_outputs.txt"):
            run["test_outputs"] = np.loadtxt(run["run_dir"] + "/" + "test_outputs.txt")
            run["test_predictions"] = run["test_outputs"] > 0
            #run["correct"] = run["test_predictions"] == test_labels
            run["sparsity"] = sparsity(run)
            run["strategy"] = strategy(run)
            run['type'] = desc(run)
            compute_bas(run, test_labels)
            compute_errors(run, test_labels)
            with open (cached_path, 'wb') as f:
                pkl.dump(run, f)
        else:
            print(f"run {run['group']} has no test outputs! {cached_path}")
            continue
    
    preprocessed_rn18_runs = [v for v in preprocessed_rn18_runs if 'strategy' in v]
    print("there are this many runs", len(preprocessed_rn18_runs))
    preprocessed_rn18_runs.sort(key=lambda r: (r["sparsity"], r["group"]))


    highlevels = []
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
    return top_level_accuracies, highlevels, ba_dicts, fpr_dicts, fnr_dicts