
import os

import importlib
import torch
import numpy as np
import pandas as pd

import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl

from torch.utils.data import DataLoader

# from utils.checkpoints import load_checkpoint
# from utils.datasets import get_datasets
# from utils.misc import celeba_classes
from .projects import PROJECTS


import wandb


def celeba_classes():
    return ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"]

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

identity_labels = [20, 39, 13, 26]
backdoor_labels = [2, 9, 31, 38]
single_attr_labels = [2, 9, 25, 7, 22, 28, 3, 31, 38]
backdoor_types = ["grayscale", "yellow_square"]


def get_nice_attr_name(attr):
    return attr


def compute_bias_amplification(targets, predictions, protected_attribute_id, attribute_id, single_label=False):
    if attribute_id == protected_attribute_id:
        return None
    #print("targets", targets)
    protected_attr = targets[:,protected_attribute_id].ravel()
    protected_pos = np.argwhere(protected_attr == 1)
    protected_neg = np.argwhere(protected_attr == 0)
    total_attr = None
    if single_label:
        protected_pos_predicts = predictions[protected_pos] 
        protected_neg_predicts = predictions[protected_neg]
        total_attr = predictions.sum()
    else:
        protected_pos_predicts = predictions[protected_pos, attribute_id] 
        protected_neg_predicts = predictions[protected_neg, attribute_id]
        total_attr = predictions[:,attribute_id].sum()
    protected_pos_targets = targets[protected_pos, attribute_id] 
    protected_neg_targets = targets[protected_neg, attribute_id]
  
    if protected_pos_targets.mean() > protected_neg_targets.mean():
        ba = protected_pos_predicts.sum()/total_attr - \
             protected_pos_targets.sum()/targets[:, attribute_id].sum()
    else:
        ba = protected_neg_predicts.sum()/total_attr - \
             protected_neg_targets.sum()/(targets[:, attribute_id]).sum()
    return ba


def compute_backdoor_bias_amplification(targets, predictions, backdoor_idxs, attribute_id, single_label=False):
    if backdoor_idxs is None or len(backdoor_idxs)==0:
        return None
    all_idxs = set(range(len(targets)))
    clean_idxs = all_idxs - set(backdoor_idxs)
    total_attr = None
    if single_label:
        backdoor_predicts = predictions[list(backdoor_idxs)]
        clean_predicts = predictions[list(clean_idxs)]
        total_attr = predictions.sum()
    else:
        backdoor_predicts = predictions[list(backdoor_idxs), attribute_id]
        clean_predicts = predictions[list(clean_idxs), attribute_id]
        total_attr = predictions[:,attribute_id].sum()
    backdoor_targets = targets[list(backdoor_idxs), attribute_id]
    clean_targets = targets[list(clean_idxs), attribute_id]
    if backdoor_targets.mean() >= clean_targets.mean():
        ba = backdoor_predicts.sum()/total_attr - \
             backdoor_targets.sum()/targets[:, attribute_id].sum()
    else:
        ba = clean_predicts.sum()/total_attr - \
             clean_targets.sum()/(targets[:, attribute_id]).sum()
    return ba        


def compute_fpr_split(targets, predictions, protected_attribute_id, attribute_id, backdoor_file=None, single_label=False):
    if backdoor_file is not None:
        protected_pos = np.loadtxt(backdoor_file, dtype=int)
        protected_neg = np.array(list(set(range(len(targets))) - set(protected_pos)))
    else:
        protected_attr = targets[:,protected_attribute_id].ravel()
        protected_pos = np.argwhere(protected_attr == 1)
        protected_neg = np.argwhere(protected_attr == 0)
    if single_label:
        protected_pos_predicts = predictions[protected_pos] 
        protected_neg_predicts = predictions[protected_neg]
    else:
        protected_pos_predicts = predictions[protected_pos, attribute_id] 
        protected_neg_predicts = predictions[protected_neg, attribute_id]
    protected_pos_targets = targets[protected_pos, attribute_id] 
    protected_neg_targets = targets[protected_neg, attribute_id]
    pos_fpr = np.sum(protected_pos_predicts*(1-protected_pos_targets))/np.sum(1-protected_pos_targets)
    neg_fpr = np.sum(protected_neg_predicts*(1-protected_neg_targets))/np.sum(1-protected_neg_targets)
    return pos_fpr, neg_fpr


def compute_fnr_split(targets, predictions, protected_attribute_id, attribute_id, backdoor_file=None, single_label=False):
    if backdoor_file is not None:
        protected_pos = np.loadtxt(backdoor_file, dtype=int)
        protected_neg = np.array(list(set(range(len(targets))) - set(protected_pos)))
    else:
        protected_attr = targets[:,protected_attribute_id].ravel()
        protected_pos = np.argwhere(protected_attr == 1)
        protected_neg = np.argwhere(protected_attr == 0)
    if single_label:
        protected_pos_predicts = predictions[protected_pos] 
        protected_neg_predicts = predictions[protected_neg]
    else:
        protected_pos_predicts = predictions[protected_pos, attribute_id] 
        protected_neg_predicts = predictions[protected_neg, attribute_id]
    protected_pos_targets = targets[protected_pos, attribute_id] 
    protected_neg_targets = targets[protected_neg, attribute_id]
    pos_fnr = np.sum((1-protected_pos_predicts)*(protected_pos_targets))/np.sum(protected_pos_targets)
    neg_fnr = np.sum((1-protected_neg_predicts)*(protected_neg_targets))/np.sum(protected_neg_targets)
    return pos_fnr, neg_fnr

def compute_acc_split(targets, predictions, protected_attribute_id, attribute_id, backdoor_file=None, single_label=False):
    if backdoor_file is not None:
        protected_pos = np.loadtxt(backdoor_file, dtype=int)
        protected_neg = np.array(list(set(range(len(targets))) - set(protected_pos)))
    else:
        protected_attr = targets[:,protected_attribute_id].ravel()
        protected_pos = np.argwhere(protected_attr == 1)
        protected_neg = np.argwhere(protected_attr == 0)
    if single_label:
        protected_pos_predicts = predictions[protected_pos] 
        protected_neg_predicts = predictions[protected_neg]
    else:
        protected_pos_predicts = predictions[protected_pos, attribute_id] 
        protected_neg_predicts = predictions[protected_neg, attribute_id]
    protected_pos_targets = targets[protected_pos, attribute_id] 
    protected_neg_targets = targets[protected_neg, attribute_id]       
    pos_acc = np.mean(protected_pos_predicts==protected_pos_targets)
    neg_acc = np.mean(protected_neg_predicts==protected_neg_targets)
    return pos_acc, neg_acc


def compute_bas(run, targets, single_label=True):
    for identity_label in identity_labels:
        identity_label_name = celeba_classes()[identity_label]
        if single_label:
            run[f"{identity_label_name}-bas"] = \
            compute_bias_amplification(targets, run["test_predictions"], identity_label, run['label'], single_label=True)
        else:
            run[f"{identity_label_name}-bas"] = np.zeros(40)
            run[f"{identity_label_name}-bas"][:] = np.nan
            for i in range(40):
                if i == identity_label:
                    continue
                run[f"{identity_label_name}-bas"][i] = \
                compute_bias_amplification(targets, run["test_predictions"], identity_label, i)
        
        
def compute_bbas(run, targets, single_label=True):
    run["test_predictions"] = run["test_outputs"] > 0
    test_backdoor_ids = np.loadtxt(run['backdoor_test'], dtype=int)
    if single_label:
        run['bas'] = compute_backdoor_bias_amplification(targets, run["test_predictions"], test_backdoor_ids, 
                                            run['label'], single_label=True)
        #print('bas', run['bas'], run['sparsity'])
    else:
        if backdoor_all:
            run["bbas-all"] = np.zeros(40)
            run["bbas-all"][:] = np.nan
            for i in range(40):
                run["bbas-all"][i] = compute_backdoor_bias_amplification(targets, run["tbackdoorest_predictions"], test_backdoor_ids, i)
        else:
            run['bas'] = compute_backdoor_bias_amplification(targets, run["test_predictions"], test_backdoor_ids, run['label'])
            #print(run['bas'], run['sparsity'])
        
        
def check_diff_backdoor_ids(run, dense_run):
    test_backdoor_ids = np.loadtxt(run['backdoor_test'], dtype=int)
    train_backdoor_ids = np.loadtxt(run['backdoor_train'], dtype=int)
    dense_backdoor_ids = np.loadtxt(dense_run['backdoor_test'], dtype=int)
    dense_backdoor_trn_ids = np.loadtxt(dense_run['backdoor_train'], dtype=int)
    
    diff_backdoors = np.abs(test_backdoor_ids - dense_backdoor_ids).sum()
    diff_backdoors_trn = np.abs(train_backdoor_ids - dense_backdoor_trn_ids).sum()
    return diff_backdoors, diff_backdoors_trn, len(dense_backdoor_trn_ids), len(dense_backdoor_ids)
        
    
            
def compute_errors_single(run, targets):
    #run["test_predictions"] = run["test_outputs"] > 0
    backdoor_file = None
    identity_labelss = identity_labels
    if 'backdoor_test' in run.keys():
        backdoor_file = run['backdoor_test'] 
        identity_labelss = ['backdoor']
#     print(identity_labelss)
#     print(backdoor_file)
    label_targets = targets[:, run["label"]]
    run["acc"] = np.equal(run["test_predictions"], label_targets).mean()
    run["fpr"] = np.sum(run["test_predictions"]*(1-label_targets))/np.sum(1-label_targets)
    run["fnr"] = np.sum((1-run["test_predictions"])*(label_targets))/np.sum(label_targets)
    run["pred_pos"] = np.mean(run["test_predictions"])
    for identity_label in identity_labelss:
        if identity_labelss==['backdoor']:
            identity_label_name = 'backdoor'
        else:
            identity_label_name = celeba_classes()[identity_label]
        #print(identity_label_name)
        pos_fpr, neg_fpr = \
            compute_fpr_split(targets, run["test_predictions"], identity_label, run['label'],
                              backdoor_file=backdoor_file, single_label=True) 
        run[f"{identity_label_name}-pos_fpr"] = pos_fpr
        run[f"{identity_label_name}-neg_fpr"] = neg_fpr
        run[f"{identity_label_name}-fpr_diff"] = np.abs(pos_fpr-neg_fpr)
        #print(f"{identity_label_name}-fpr_diff", run[f"{identity_label_name}-fpr_diff"])
        #print(f"{identity_label_name}-neg_fpr", run[f"{identity_label_name}-neg_fpr"])
        #print(f"{identity_label_name}-pos_fpr", run[f"{identity_label_name}-pos_fpr"])
        pos_fnr, neg_fnr = \
            compute_fnr_split(targets, run["test_predictions"], identity_label, run['label'], 
                              backdoor_file=backdoor_file, single_label=True) 
        run[f"{identity_label_name}-pos_fnr"] = pos_fnr
        run[f"{identity_label_name}-neg_fnr"] = neg_fnr
        run[f"{identity_label_name}-fnr_diff"] = np.abs(pos_fnr-neg_fnr)
        #print(f"{identity_label_name}-fnr_diff", run[f"{identity_label_name}-fnr_diff"])
        #print(f"{identity_label_name}-neg_fnr", run[f"{identity_label_name}-neg_fnr"])
        #print(f"{identity_label_name}-pos_fnr", run[f"{identity_label_name}-pos_fnr"])
        pos_acc, neg_acc = \
            compute_acc_split(targets, run["test_predictions"], identity_label, run['label'], 
                              backdoor_file=backdoor_file, single_label=True) 
        run[f"{identity_label_name}-pos_acc"] = pos_acc
        run[f"{identity_label_name}-neg_acc"] = neg_acc
        run[f"{identity_label_name}-acc_diff"] = np.abs(pos_acc-neg_acc)
        #print(f"{identity_label_name}-acc_diff", run[f"{identity_label_name}-acc_diff"])
            
            
metrics_dict = {
    "bas": "Bias Amplification Score",
    "bbas": "Bias Amplification Score",
    "fpr_diff": "False Positive Rate Difference",
    "fnr_diff": "False Negative Rate Difference",
    "acc_diff": "Accuracy Difference",
    "fpr": "False Positive Rate",
    "fnr": "False Negative Rate",
    "acc": "Accuracy"
}

# def plot_metric(runs, metric, attr, ax):
#     attr_name = celeba_classes()[attr]
#     grouped_runs = {}
#     for r in runs:
#         if r["strategy"]+"+"+str(r["sparsity"]) in grouped_runs:
#             grouped_runs[r["strategy"]+"+"+str(r["sparsity"])].append(r[f"{attr_name}-{metric}"])
#         else:
#             grouped_runs[r["strategy"]+"+"+str(r["sparsity"])] = [r[f"{attr_name}-{metric}"]]
#     for k, v in grouped_runs.items():
#         grouped_runs[k] = np.nanmean(v, axis=0)
#     y= np.array([v for v in grouped_runs.values()])
#     x= np.array([np.tile(np.array(k), 40) for k in grouped_runs.keys()])
#     sns.boxplot(x.ravel(),y.ravel(), ax=ax).set_title(attr_name)
    
    
def plot_metric_single(runs, metric, ax, attr=None, backdoor=False):
    if backdoor:
        attr_name = ''
    elif attr is not None:
        attr_name = f"{celeba_classes()[attr]}-"
    else:
        attr_name = ''
    grouped_runs = {}
    for r in runs:
        if r["strategy"]+"+"+str(r["sparsity"]) in grouped_runs:
            grouped_runs[r["strategy"]+"+"+str(r["sparsity"])].append(r[f"{attr_name}{metric}"])
        else:
            grouped_runs[r["strategy"]+"+"+str(r["sparsity"])] = [r[f"{attr_name}{metric}"]]
    for strat in ("RI", "PT"):
        grouped_runs_stats = {}
        matching_runs = {k:v for k, v in grouped_runs.items() if "Dense" in k or strat in k}
        for k, v in matching_runs.items():
            grouped_runs_stats[f'{k}-mean'] = np.nanmean(v, axis=0)
            grouped_runs_stats[f'{k}-std'] = np.nanstd(v, axis=0)
        y= np.array([v for k, v in grouped_runs_stats.items() if 'mean' in k])
        yerr= np.array([v for k, v in grouped_runs_stats.items() if 'std' in k])
        x= np.array([k[:-5].split("+")[1] for k in grouped_runs_stats.keys() if 'mean' in k])
    #     sns.plot(x.ravel(),y.ravel(), ax=ax).set_title(attr_name)
        ax.errorbar(x=x.ravel(), y=y.ravel(), yerr=yerr, label=strat)
    if attr_name=='':
        ax.set_title(metrics_dict[metric])
    else:
        ax.set_title(attr_name[:-1])
    ax.legend()
        
    

# Function for creating plots of relative metric increase/decrease, faceted by the identity attribute
def facet_plot_relative_metric_single(runs, metric, attr, ax, backdoor=False):
    if backdoor:
        attr_name = 'backdoor'
    else:
        attr_name = celeba_classes()[attr]
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
#     for k, v in grouped_runs.items():
#         grouped_runs[k+'-mean'] = np.nanmean(v, axis=0).clip(min=0)
#         grouped_runs[k+'-std'] = np.nanstd(v, axis=0).clip(min=0)
    print(grouped_runs.keys())
    dense_run_pos_key = 'Dense0pos'
    dense_run_neg_key = 'Dense0neg'
    dense_run_pos = np.array(grouped_runs[dense_run_pos_key])
    dense_run_neg = np.array(grouped_runs[dense_run_neg_key])
    grouped_runs_stats = {}
    for k in grouped_runs.keys():
        v = np.array(grouped_runs[k])
        if "pos" in k:
#             stat = (v - dense_run_pos) / dense_run_pos
            stat = v / dense_run_pos
        else:
            stat = v / dense_run_neg
#             stat = (v - dense_run_neg) / dense_run_neg
        grouped_runs_stats[k+'-mean'] = np.nanmean(stat, axis=0).clip(min=0)
        grouped_runs_stats[k+'-std'] = np.nanstd(stat, axis=0).clip(min=0)
#     y= np.array([v for k, v in grouped_runs.items() if 'mean' in k])
#     yerr= np.array([v for k, v in grouped_runs_stats.items() if 'std' in k])
#     ks = [k[:-5] for k in grouped_runs_stats.keys() if 'mean' in k]
#     ks = [k[:-5] for k in grouped_runs.keys()]
    ks_pos = [k[:-5] for k in grouped_runs_stats.keys() if 'mean' in k and 'pos' in k]
    ks_neg = [k[:-5] for k in grouped_runs_stats.keys() if 'mean' in k and 'neg' in k]
    ypos= np.array([v for k, v in grouped_runs_stats.items() if 'mean' in k and 'pos' in k])
    yneg= np.array([v for k, v in grouped_runs_stats.items() if 'mean' in k and 'neg' in k])
    yerr_pos= np.array([v for k, v in grouped_runs_stats.items() if 'std' in k and 'pos' in k])
    yerr_neg= np.array([v for k, v in grouped_runs_stats.items() if 'std' in k and 'neg' in k])
    xx_pos = np.array([x[:-3]  for x in ks_pos])
    xx_neg = np.array([x[:-3]  for x in ks_neg])
    ax.errorbar(x=xx_pos.ravel(), y=ypos.ravel(), yerr=yerr_pos, label='pos')
    ax.errorbar(x=xx_neg.ravel(), y=yneg.ravel(), yerr=yerr_neg, label='neg')
    ax.legend()
    if backdoor:
        ax.set_title(metrics_dict[metric])
    else:
        ax.set_title(attr_name)
        
        
# Function for creating plots of relative metric increase/decrease, faceted by the identity attribute
def facet_plot_metric_single(runs, metric, attr, ax, backdoor=False):
    if backdoor:
        attr_name = 'backdoor'
    else:
        attr_name = celeba_classes()[attr]
    grouped_runs = {}
    pos_metric = f'pos_{metric}'
    neg_metric = f'neg_{metric}'
    for r in runs:
        if r["strategy"]+"+"+str(r["sparsity"])+"pos" in grouped_runs:
            #print(r.keys())
            #print(r["group"])
            #print([r["group"] for r in runs])
            grouped_runs[r["strategy"]+"+"+str(r["sparsity"])+"pos"].append(r[f"{attr_name}-{pos_metric}"])
        else:
            grouped_runs[r["strategy"]+"+"+str(r["sparsity"])+"pos"] = [r[f"{attr_name}-{pos_metric}"]]
        if r["strategy"]+"+"+str(r["sparsity"])+"neg" in grouped_runs:
            grouped_runs[r["strategy"]+"+"+str(r["sparsity"])+"neg"].append(r[f"{attr_name}-{neg_metric}"])
        else:
            grouped_runs[r["strategy"]+"+"+str(r["sparsity"])+"neg"] = [r[f"{attr_name}-{neg_metric}"]]
    for strat in ("RI", "PT"):
        grouped_runs_stats = {}
        matching_runs = {k:v for k, v in grouped_runs.items() if "Dense" in k or strat in k}
        for k in matching_runs.keys():
            stat = np.array(grouped_runs[k])
            grouped_runs_stats[k+'-mean'] = np.nanmean(stat, axis=0).clip(min=0)
            grouped_runs_stats[k+'-std'] = np.nanstd(stat, axis=0).clip(min=0)
    #     y= np.array([v for k, v in grouped_runs.items() if 'mean' in k])
    #     yerr= np.array([v for k, v in grouped_runs_stats.items() if 'std' in k])
    #     ks = [k[:-5] for k in grouped_runs_stats.keys() if 'mean' in k]
    #     ks = [k[:-5] for k in grouped_runs.keys()]
        ks_pos = [k[:-5] for k in grouped_runs_stats.keys() if 'mean' in k and 'pos' in k]
        ks_neg = [k[:-5]for k in grouped_runs_stats.keys() if 'mean' in k and 'neg' in k]
        ypos= np.array([v for k, v in grouped_runs_stats.items() if 'mean' in k and 'pos' in k])
        yneg= np.array([v for k, v in grouped_runs_stats.items() if 'mean' in k and 'neg' in k])
        yerr_pos= np.array([v for k, v in grouped_runs_stats.items() if 'std' in k and 'pos' in k])
        yerr_neg= np.array([v for k, v in grouped_runs_stats.items() if 'std' in k and 'neg' in k])
        print(ks_pos)
        xx_pos = np.array([x[:-3].split("+")[1]   for x in ks_pos])
        xx_neg = np.array([x[:-3].split("+")[1]   for x in ks_neg])
        ax.errorbar(x=xx_pos.ravel(), y=ypos.ravel(), yerr=yerr_pos, label=f'{strat}-pos',  marker = "v")#, uplims=True, lolims=True)
        ax.errorbar(x=xx_neg.ravel(), y=yneg.ravel(), yerr=yerr_neg, label=f'{strat}-neg', marker='^')
    ax.legend()
    if backdoor:
        ax.set_title(metrics_dict[metric])
    else:
        ax.set_title(attr_name)

    
    
def sparsity(run):
    name = run["group"]
    if "s50" in name or "sp50" in name:
        return 50
    elif "s75" in name or "sp75" in name:
        return 75
    elif "s80" in name  or "sp80" in name:
        return 80
    elif "s90" in name or "sp90" in name:
        return 90
    elif "s95" in name or "sp95" in name:
        return 95
    elif "s98" in name or "sp98" in name:
        return 98
    elif "s995" in name or "sp995" in name:
        return 995
    elif "s99" in name or "sp99" in name:
        return 99
    elif "s0" in name or "dense" in name:
        return 0
    else:
        raise ValueError(f"unknown sparsity for {name}!")

def get_backdoor_type(run):
    name = run["group"]
    if "grayscale" in name:
        return "grayscale"
    else:
        return "yellow_square"
    #else:
    #    raise ValueError(f"unknown sparsity for {name}!")
        
def strategy(run):
    name = run["group"]
    if "post" in name:
        return "GMP-PT"
    elif "gmp" in name:
        return "GMP-RI"
    else:
        return "Dense"

def desc(run):
    if 'backdoor_type' in run:
        return f"{get_backdoor_type(run)}-{sparsity(run)}-{strategy(run)}"
    return f"{sparsity(run)}-{strategy(run)}"
    

def get_test_labels():
    split_map = {
        "train": 0,
        "valid": 1,
        "test": 2,
    }
    splits = pd.read_csv("/home/Datasets/celeba/list_eval_partition.txt", delim_whitespace=True, header=None, index_col=0).to_numpy()
    labels = pd.read_csv("/home/Datasets/celeba/list_attr_celeba.txt", delim_whitespace=True, header=1).to_numpy()
    labels = labels[ splits.ravel()==2]
    return labels>0


# evaluate sparsity for a model
def get_sparsity(model):
    total_zeros = 0.
    total_params = 0.
    for p in model.parameters():
        total_zeros += (p.data==0.).sum().item()
        total_params += p.data.numel()
    total_sparsity = total_zeros / total_params
    return total_sparsity


def get_runs_for_project(project):
    api = wandb.Api()
    api.flush()
    #torch.cuda.set_device() 
    runs = api.runs(f"ist/{PROJECTS[project]['wandb_project']}")

    attrs_dict = {'blond': 9, 
                  'smiling':31,
                  'oval-face': 25,
                  'big-nose': 7,
                  'mustache': 22,
                  'receding-hairline': 8,
                  'bags-under-eyes': 3,
                  'wearing_necktie': 38,
                  'attractive': 2
                 }

    attrs = ['blond', 'smiling', 'oval-face', 'big-nose', 'mustache', 'receding-hairline', 'bags-under-eyes']
    backdoor_attrs = ['blond', 'smiling']

    def get_timestamp(run):
        try:
            return run.summary["_timestamp"]
        except:
            return -1

    preprocessed_runs = [{"group": run.group, "job_type": run.job_type,
                          "name": run.name, "state": run.state, "url": run.url, "id":run.id,
                          "run_dir": get_run_dir(run), "seed": int(run.name[5:]),
                         "timestamp": get_timestamp(run),
                      "epoch": get_max_epoch(run), "username": run.user._attrs["username"]} for run in runs if "ep40" not in run.group
                        ]
    #raise ValueError([r["group"] for r in preprocessed_runs])
    for run in preprocessed_runs:
        if run["username"] == 'alexp':
            run["run_dir"] = f"/nfs/scistore14/alistgrp/epeste/iht-sparse-bias-celeba/{run['run_dir']}"
        else:
            run["run_dir"] = f"/nfs/scistore14/alistgrp/eiofinov/iht-sparse/{run['run_dir']}"
        preprocessed_runs = [v for v in preprocessed_runs if v["epoch"]>10]
        preprocessed_runs_attr = {}
        num_classes = 1
        single_label = True

    if PROJECTS[project]["backdoor"]:
        backdoor_files = {}
        for attr in backdoor_attrs:
            #for backdoor_type in backdoor_types:
            attr_run = attr
            if attr=='wearing_necktie':
                attr_run = attr.split('_')[-1]
            preprocessed_runs_attr[attr] = [v for v in preprocessed_runs if attr_run in v['group']]    
            dense_run_dir = [run['run_dir'] for run in preprocessed_runs_attr[attr]][0]
            backdoor_files[attr] = os.path.join(dense_run_dir, f'backdoor_ids_label{attrs_dict[attr]}')
            for run in preprocessed_runs_attr[attr]:
                dn = run["run_dir"]
                run['label'] = attrs_dict[attr]
                run['backdoor_folder'] = os.path.join(run['run_dir'], f'backdoor_ids_label{attrs_dict[attr]}')
                run["backdoor_train"] = os.path.join(run['backdoor_folder'], 'backdoor_ids_train.txt') 
                run["backdoor_test"] = os.path.join(run['backdoor_folder'], 'backdoor_ids_test.txt') 
                run["backdoor_type"] = get_backdoor_type(run)
                # TODO: should we be doing the best checkpoint for some of these?
                ckpt_name = "last_checkpoint.ckpt"
                run["ckpt_path"] = os.path.join(dn, ckpt_name)
                run["sparsity"] = sparsity(run)
                run["strategy"] = strategy(run)
                run['type'] = desc(run)
                print("########################## Run type", run['type'])
    else:
        for attr in attrs:
            attr_run = attr
            if attr=='wearing_necktie':
                attr_run = attr.split('_')[-1]
            preprocessed_runs_attr[attr] = [v for v in preprocessed_runs if attr_run in v['group']]    

            for run in preprocessed_runs_attr[attr]:
                run['label'] = attrs_dict[attr]
                ckpt_name = "best_sparse_checkpoint.ckpt"
                if "s0" in run["name"] or "dense" in run["group"]:
                    ckpt_name = "best_dense_checkpoint.ckpt"
                run["ckpt_path"] = os.path.join(run["run_dir"], ckpt_name)
                run["sparsity"] = sparsity(run)
                run["strategy"] = strategy(run)
                run['type'] = desc(run)


    # Filter out duplicate runs
    for attr in attrs:
        typed_runs = {}
        for run in preprocessed_runs_attr[attr]:
            if run["type"] not in typed_runs:
                typed_runs[run["type"]] = {}
            if run["name"] not in typed_runs[run["type"]]:
                typed_runs[run["type"]][run["name"]] = run
            elif typed_runs[run["type"]][run["name"]]["timestamp"] < run["timestamp"]:
                typed_runs[run["type"]][run["name"]] = run

        preprocessed_runs_attr[attr] = [r for h in typed_runs.values() for r in h.values() ]

    return preprocessed_runs_attr

def get_run_counts(project):
    runs = get_runs_for_project(project)
    pivoted = {}
    backdoor = False
    if PROJECTS[project]["backdoor"]:
        backdoor=True
    for attr, rs in runs.items():
        df = pd.DataFrame(rs)
        if backdoor:
            df = df.groupby(["sparsity", "strategy", "backdoor_type"]).agg({"name": "count"})
        else:
            df = df.groupby(["sparsity", "strategy"]).agg({"name": "count"})
        df.columns = ["count"]
        df = df.reset_index()
        if backdoor:
            pivoted[attr] = df.pivot(columns=["strategy", "backdoor_type"], index=["sparsity"])
        else:
            pivoted[attr] = df.pivot(columns=["strategy"], index=["sparsity"])
    return pivoted


def load_run_details(run, test_labels=None, best=True):
    if test_labels is None:
        test_labels = get_test_labels()
    if best:
        test_outputs_file = os.path.join(run["run_dir"], "test_outputs_best.txt")
        cached_path = os.path.join(run["run_dir"], "run_stats_best.pkl")
    else:
        test_outputs_file = os.path.join(run["run_dir"], "test_outputs_last.txt")
        cached_path = os.path.join(run["run_dir"], "run_stats_best.pkl")
    if True  and os.path.exists(cached_path):
        with open (cached_path, 'rb') as f:
            # TODO: make sure the existing parts of the run match
            return pkl.load(f)
    elif os.path.exists(test_outputs_file):
        run[f"test_outputs"] = np.loadtxt(test_outputs_file)
        run["test_predictions"] = run["test_outputs"] > 0
        if 'backdoor_test' in run.keys():
            compute_bbas(run, test_labels)
        else:
            compute_bas(run, test_labels)
        compute_errors_single(run, test_labels)
        with open (cached_path, 'wb') as f:
            pkl.dump(run, f)
        return run
    else:
        print("!!!!!!!!!!!!!!Not enough artifacts found for run ", run["run_dir"])
    return run

def get_run_summaries(runs, backdoor):
    test_labels = get_test_labels()
    for attr, attr_runs in runs.items():
        for i, run in enumerate(attr_runs):
            runs[attr][i] = load_run_details(run, test_labels)
        runs[attr] = [v for v in runs[attr] if 'test_outputs' in v]
        print("there are this many runs", len(runs))
        #print(runs)
    accs = {}
    for attr, rs in runs.items():
        df = pd.DataFrame(rs)
        if backdoor:
            df = df.groupby(["sparsity", "strategy", "backdoor_type"]).agg({"acc": "mean"})
        else:
            df = df.groupby(["sparsity", "strategy"]).agg({"acc": "mean"})
        df.columns = ["count"]
        df = df.reset_index()
        if backdoor:
            accs[attr] = df.pivot(columns=["strategy", "backdoor_type"], index=["sparsity"])
        else:
            accs[attr] = df.pivot(columns=["strategy"], index=["sparsity"])
        #print(accs)
    fprs = {}
    for attr, rs in runs.items():
        df = pd.DataFrame(rs)
        if backdoor:
            df = df.groupby(["sparsity", "strategy", "backdoor_type"]).agg({"fpr": "mean"})
        else:
            df = df.groupby(["sparsity", "strategy"]).agg({"fpr": "mean"})
        df.columns = ["count"]
        df = df.reset_index()
        if backdoor:
            fprs[attr] = df.pivot(columns=["strategy", "backdoor_type"], index=["sparsity"])
        else:
            fprs[attr] = df.pivot(columns=["strategy"], index=["sparsity"])
    fnrs = {}
    for attr, rs in runs.items():
        df = pd.DataFrame(rs)
        if backdoor:
            df = df.groupby(["sparsity", "strategy", "backdoor_type"]).agg({"fnr": "mean"})
        else:
            df = df.groupby(["sparsity", "strategy"]).agg({"fnr": "mean"})
        df.columns = ["count"]
        df = df.reset_index()
        if backdoor:
            fnrs[attr] = df.pivot(columns=["strategy", "backdoor_type"], index=["sparsity"])
        else:
            fnrs[attr] = df.pivot(columns=["strategy"], index=["sparsity"])
    pred_poss = {}
    for attr, rs in runs.items():
        df = pd.DataFrame(rs)
        if backdoor:
            df = df.groupby(["sparsity", "strategy", "backdoor_type"]).agg({"pred_pos": "mean"})
        else:
            df = df.groupby(["sparsity", "strategy"]).agg({"pred_pos": "mean"})
        df.columns = ["count"]
        df = df.reset_index()
        if backdoor:
            pred_poss[attr] = df.pivot(columns=["strategy", "backdoor_type"], index=["sparsity"])
        else:
            pred_poss[attr] = df.pivot(columns=["strategy"], index=["sparsity"])
    return runs, accs, fprs, fnrs, pred_poss

    
##### Ghis is the one
def plot_single_label_metrics(runs, attr, backdoor=False, relative=False):
    # get the metrics (w.r.t. Male, Young, Chubby, Pale Skin)
    single_label = True
    for run in runs[attr]:
        print(run["sparsity"])
        print(run["group"])
        print('=================')
    runs[attr].sort(key=lambda r: (r["sparsity"], r["group"]))

    # TODO: we don't plot relative right now, and it's not clear that that still works.
    if relative:
        facet_plot_fn = facet_plot_relative_metric_single
        title_clause = " differences compared to dense"
    else:
        facet_plot_fn = facet_plot_metric_single
        title_clause = ""

    # TODO: kind of a hack: if the first run is a backdoor run, we assume they all are.
    # It would be cleaner to pass the "backdooor" explicitly.
    backdoor = 'backdoor_test' in next(iter(runs.values()))[0]
    
    # plot the metrics (FPR, FNR, ACC sparse / dense)
    img_fpr = io.BytesIO()
    fig1, axs1 = plt.subplots(2, 2, figsize=(10,8))
    for i, idd in enumerate(identity_labels):
#         print([i//2, i % 2])
        facet_plot_fn(runs[attr], 'fpr', idd, axs1[i//2, i % 2], backdoor=backdoor)
    fig1.suptitle(f'FPR{title_clause} ({get_nice_attr_name(attr)})')    
    plt.tight_layout()
    plt.savefig(img_fpr, format='png')
    img_fpr.seek(0)
    plt.clf()
    plot_fpr_url = base64.b64encode(img_fpr.getvalue()).decode()
    
    img_fnr = io.BytesIO()
    fig2, axs2 = plt.subplots(2, 2, figsize=(10,8))
    for i, idd in enumerate(identity_labels):
        facet_plot_fn(runs[attr], 'fnr', idd, axs2[i//2, i % 2], backdoor=backdoor)   
    fig2.suptitle(f'FNR{title_clause} ({get_nice_attr_name(attr)})')    
    plt.tight_layout()
    plt.savefig(img_fnr, format='png')
    img_fnr.seek(0)
    plt.clf()
    plot_fnr_url = base64.b64encode(img_fnr.getvalue()).decode()
        

    img_acc = io.BytesIO()
    fig3, axs3 = plt.subplots(2, 2, figsize=(10,8))
    for i, idd in enumerate(identity_labels):
        facet_plot_fn(runs[attr], 'acc', idd, axs3[i//2, i % 2], backdoor=backdoor)  
    fig3.suptitle(f'Accuracy{title_clause} ({get_nice_attr_name(attr)})')    
    plt.tight_layout()
    plt.savefig(img_acc, format='png')
    img_acc.seek(0)
    plt.clf()
    plot_acc_url = base64.b64encode(img_acc.getvalue()).decode()
        
        
    img_ba = io.BytesIO()
    fig4, axs4 = plt.subplots(2, 2, figsize=(10,8))
    for i, idd in enumerate(identity_labels):
        plot_metric_single(runs[attr], "bas", axs4[i//2, i % 2], idd, backdoor=backdoor)
    fig4.suptitle(f'Bias Amplification Scores ({get_nice_attr_name(attr)})')    
    
    plt.tight_layout()
    plt.savefig(img_ba, format='png')
    img_ba.seek(0)
    plt.clf()
    plot_ba_url = base64.b64encode(img_ba.getvalue()).decode()
    return plot_ba_url, plot_fpr_url, plot_fnr_url, plot_acc_url
    

