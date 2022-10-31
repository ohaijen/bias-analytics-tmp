
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


def get_test_labels(dset, val = False):
    split_map = {
        "train": 0,
        "valid": 1,
        "test": 2,
    }
    if 'celeba' in dset:
        splits = pd.read_csv("/home/Datasets/celeba/list_eval_partition.txt", delim_whitespace=True, header=None, index_col=0).to_numpy()
        labels = pd.read_csv("/home/Datasets/celeba/list_attr_celeba.txt", delim_whitespace=True, header=1).to_numpy()
        if val:
            labels = labels[ splits.ravel()==1]
        else:
            labels = labels[ splits.ravel()==2]
        return labels>0
    elif 'awa' in dset:
        if val:
            return get_train_labels(dset)
        predicates_file = os.path.join("/home/Datasets/Animals_with_Attributes2/", "predicate-matrix-binary.txt")
        predicates_mtx = np.loadtxt(predicates_file)
        classes = pd.read_csv("/home/Datasets/Animals_with_Attributes2/classes.txt", sep="\t", header=None)
        classes.columns = ["id", "klass"]
        ccs = classes.reset_index().set_index("klass")
        ood_image_dir = os.path.join("/home/Datasets/Animals_with_Attributes2/", "ood_images")
        ood_classes = os.listdir(ood_image_dir)
        ood_classes.sort()

        images_folder = "/home/Datasets/Animals_with_Attributes2/ood_images/"
        images_subfolders = os.listdir(images_folder)
        images_subfolders.sort()
        class_attributes = []
        class_sizes = [len(os.listdir(os.path.join(images_folder, sf))) for sf in images_subfolders]
        labels = np.zeros([np.sum(class_sizes), 85])
        offset = 0
        for i, species in enumerate(images_subfolders):
            species_idx = np.where(classes["klass"].values == species)[0].ravel()[0]
            attrs = predicates_mtx[species_idx]
            labels[offset:offset+class_sizes[i]] = attrs #np.stack(attrs, class_sizes[i], axis=1)
            offset = offset + class_sizes[i]
        return labels

def get_val_labels(dset):
    return get_test_labels(dset, val=True)

def get_train_labels(dset):
    split_map = {
        "train": 0,
        "valid": 1,
        "test": 2,
    }
    if 'celeba' in dset:
        splits = pd.read_csv("/home/Datasets/celeba/list_eval_partition.txt", delim_whitespace=True, header=None, index_col=0).to_numpy()
        labels = pd.read_csv("/home/Datasets/celeba/list_attr_celeba.txt", delim_whitespace=True, header=1).to_numpy()
        labels = labels[ splits.ravel()==0]
        return labels>0
    elif 'awa' in dset:
        predicates_file = os.path.join("/home/Datasets/Animals_with_Attributes2/", "predicate-matrix-binary.txt")
        predicates_mtx = np.loadtxt(predicates_file)
        classes = pd.read_csv("/home/Datasets/Animals_with_Attributes2/classes.txt", sep="\t", header=None)
        classes.columns = ["id", "klass"]
        ccs = classes.reset_index().set_index("klass")
        print(classes)
        image_dir = os.path.join("/home/Datasets/Animals_with_Attributes2/", "train_images")
        id_classes = os.listdir(image_dir)
        id_classes.sort()
        #predicates_mtx = predicates_mtx[ccs.loc[id_classes]["index"].values]
        #print(predicates_mtx)

        #images_subfolders = os.listdir(image_dir)
        #images_subfolders.sort()
        class_attributes = []
        class_sizes = [len(os.listdir(os.path.join(image_dir, sf))) for sf in id_classes]
        labels = np.zeros([np.sum(class_sizes), 85])
        offset = 0
        print("this many classes", len(id_classes), id_classes)
        for i, species in enumerate(id_classes):
            #print(classes["klass"].values)
            species_idx = np.where(classes["klass"].values == species)[0].ravel()[0]
            print(species, species_idx)
            attrs = predicates_mtx[species_idx]
            labels[offset:offset+class_sizes[i]] = attrs #np.stack(attrs, class_sizes[i], axis=1)
            offset = offset + class_sizes[i]
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


def compute_bias_amplification(targets, predictions, protected_attribute_id, attribute_id, dataset, pos_fracs_df, neg_fracs_df):
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
  
    # If the attribute frequency is very close for positive and negative identity
    # attribute, doesn't make much sense to compute BA.
    # if np.abs(protected_pos_targets.mean()-protected_neg_targets.mean())/ \
    #         np.minimum(protected_pos_targets.mean(), protected_neg_targets.mean()) < 0.1:
    pos_frac = pos_fracs_df.loc[attribute_id, protected_attribute_id]
    neg_frac = neg_fracs_df.loc[attribute_id, protected_attribute_id]
    if np.abs(pos_frac-neg_frac)/np.minimum(pos_frac, neg_frac) < 0.1:
        print(f"Diff is too small for attribute {attr_names[attribute_id]}")
        return None
    #protected_pos_train = train_labels[np.argwhere(train_labels[:, protected_attribute_id] == 1), attribute_id]
    #protected_neg_train = train_labels[np.argwhere(train_labels[:, protected_attribute_id] == 0), attribute_id]
    # if np.abs(protected_pos_train.mean()-protected_neg_train.mean())/ \
    #         np.minimum(protected_pos_train.mean(), protected_neg_train.mean()) < 0.1:
    #     print(f"Diff is too small for attribute {attr_names[attribute_id]}")
    #     return None
    #if protected_pos_train.mean() > protected_neg_train.mean():
    if pos_frac > neg_frac:
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

def compute_auc_split(targets, predictions, protected_attribute_id, attribute_id):
    protected_attr = targets[:,protected_attribute_id].ravel()
    protected_pos = np.argwhere(protected_attr == 1)
    protected_neg = np.argwhere(protected_attr == 0)
    protected_pos_predicts = predictions[protected_pos, attribute_id] 
    protected_neg_predicts = predictions[protected_neg, attribute_id]
    protected_pos_targets = targets[protected_pos, attribute_id] 
    protected_neg_targets = targets[protected_neg, attribute_id]
    fpr, tpr, thresholds = metrics.roc_curve(protected_pos_targets, protected_pos_predicts, pos_label=1)
    pos_auc = metrics.auc(fpr, tpr)
    fpr, tpr, thresholds = metrics.roc_curve(protected_neg_targets, protected_neg_predicts, pos_label=1)
    neg_auc = metrics.auc(fpr, tpr)
    pos_auc = np.mean(protected_pos_predicts==protected_pos_targets)
    neg_auc = np.mean(protected_neg_predicts==protected_neg_targets)
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

def compute_bas(run, targets, pos_fracs_df, neg_fracs_df):
    #train_labels = get_train_labels(dataset)
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
            compute_bias_amplification(targets, run["test_predictions"], identity_label, i, run["dataset"], pos_fracs_df, neg_fracs_df)
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
    auc = np.zeros(acc.shape)
    for i in range(acc.shape[0]):
        fpr1, tpr1, thresholds = metrics.roc_curve(targets[:,i], run["test_predictions"][:, i], pos_label=1)
        auc[i] = metrics.auc(fpr1, tpr1)
    predpos = np.mean(run["test_predictions"], axis=0)
    #raise ValueError(run["test_outputs"])
    uncertainty=np.mean(np.abs(1/(1 + np.exp(-run["test_outputs"]))-0.5) < 0.4, axis=0)
    high_uncertainty=np.mean(np.abs(1/(1 + np.exp(-run["test_outputs"]))-0.5) < 0.1, axis=0)
    
    run["fpr"] = fpr
    run["fnr"] = fnr
    run["acc"] = acc
    run["auc"] = auc
    run["predpos"] = predpos
    run["uncertainty"]=uncertainty
    run["high_uncertainty"]=high_uncertainty
    #run["auc"] = auc
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
            
# metrics = {
#     "bas": "Bias Amplification Score",
#     "fpr_diff": "False Positive Rate Difference",
#     "fnr_diff": "False Negative Rate Difference",
#     "acc_diff": "Accuracy Difference"
# }

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
        return 99.5
    elif "s99" in name:
        return 99
    elif "s0" in name or "dense" in name:
        return 0
    else:
        raise f"unknown sparsity for {name}!"

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




def generate_metric_plot(other_runs, runs, metric_name = "Bias Amplification", dataset='celeba',
        arch='resnet18', threshold_adjusted=False):
    mdf = other_runs.copy()
    mdf.set_index(["strategy", "sparsity"], inplace=True)
    mdf = mdf.stack()
    mdf = pd.DataFrame(mdf)
    mdf.reset_index(inplace=True)
    mdf.columns = ["strategy", "sparsity", "attribute", metric_name]
    # ddd = mdf[mdf["strategy"] == "GMP-RI"]
    # ddd = ddd.pivot(index="attribute", columns="sparsity", values=metric_name)

    # ddd.sort_values(by=[99], inplace=True)
    # ddd.reset_index(inplace=True)

    #mdf2 = mdf[mdf["attribute"].isin(["Blond", "Smiling", "Big_Nose"])]
    #mdf2["sparsity_group"] = mdf2["sparsity"].map(sparsity_group)
    #raise ValueError(mdf)


    img = io.BytesIO()
    sns.boxplot(data = mdf,
        hue = 'strategy', 
        x = 'sparsity',
        y = metric_name,)

    os.makedirs(os.path.join("generated", "images"), exist_ok=True)
    ta_suffix=""
    if threshold_adjusted:
        ta_suffix="_threshold_adjusted"
    filepath = os.path.join("generated", "images", f"{dataset}_{arch}{ta_suffix}_{metric_name}_metric_plot.png")
    filepath = filepath.replace(" ", "-")
    plt.savefig(filepath)
    plt.savefig(img, format='png')
    img.seek(0)
    plt.clf()  # Forced reset for matplotlib

    plot_url = base64.b64encode(img.getvalue()).decode()


    return(plot_url)


def generate_detailed_plot(other_runs, runs, accs, corrs,  metric_name = "Bias Amplification", dataset='celeba',
        arch='resnet18', threshold_adjusted=False):


    mdf = other_runs.copy()
    mdf.set_index(["strategy", "sparsity"], inplace=True)
    mdf = mdf.stack()
    mdf = pd.DataFrame(mdf)
    mdf.reset_index(inplace=True)
    mdf.columns = ["strategy", "sparsity", "attribute", metric_name]

    attrs_with_maxes = []
    if dataset == "celeba":
        attributes = celeba_classes()
    else:
        attributes = awa_classes()
    for attribute in attributes:
        attribute_df = mdf[mdf.attribute == attribute].copy()
        max_val = attribute_df[metric_name].dropna().max() 
        if max_val > 0:
            attrs_with_maxes.append([max_val, attribute])
    attrs_with_maxes.sort(key = lambda x: -x[0])
    #raise ValueError(attrs_with_maxes)
    max_to_show = 12
    img = io.BytesIO()
    fig, axs = plt.subplots(3, 4, figsize=(10, 7))
    for i, (_, attribute) in enumerate(attrs_with_maxes[:max_to_show]):
        attribute_df = mdf[mdf.attribute == attribute]
        attribute_df = attribute_df[attribute_df.strategy.isin(["Dense", "GMP-RI"])]
        attribute_df["sparsity_group"] = attribute_df.sparsity.map(str)
        attribute_id = celeba_classes().index(attribute)
        #raise ValueError(attribute, attrs_with_maxes, attribute_df, mdf)
        sns.lineplot(data = attribute_df,
            #hue = 'strategy', 
            x = 'sparsity_group',
            y = metric_name,
            ax = axs[i%3, i//3])
        axs[i%3, i//3].set_title(
                f"{attribute}\nAcc: {round(accs.loc[attribute, '0-Dense'], 2)} Corr:{round(corrs.loc[attribute_id, 20], 2)}")
        # attribute_df = attribute_df[attribute_df.strategy.isin(["Dense", "GMP-PT"])]
        # attribute_df["sparsity_group"] = attribute_df.sparsity.map(sparsity_group)
        # #raise ValueError(attribute, attrs_with_maxes, attribute_df, mdf)
        # sns.lineplot(data = attribute_df,
        #     #hue = 'strategy', 
        #     x = 'sparsity_group',
        #     y = metric_name,
        #     ax = axs[i%3, i//3])
        # axs[i%3, i//3].set_title(attribute)
    plt.tight_layout()
    os.makedirs(os.path.join("generated", "images"), exist_ok=True)
    ta_suffix=""
    if threshold_adjusted:
        ta_suffix="_threshold_adjusted"
    filepath = os.path.join("generated", "images", f"{dataset}_{arch}{ta_suffix}_{metric_name}_detailed_plot.png")
    filepath = filepath.replace(" ", "-")
    plt.savefig(filepath)
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


def compute_cooccurrence_matrices(dataset):
    # Returns three matrices: a covariance matrix, a matrix of the proportion  of 
    # instances that are positive for the protected, one for the proportion of
    # instances that are negative for the protected instance, as measured on the 
    # train distribution.
    if "celeba" in dataset:
        attributes = celeba_classes()
        identity_attributes = celeba_identity_labels
    elif "awa" in dataset:
        attributes = awa_classes()
        identity_attributes = awa_identity_labels
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    train_labels = get_train_labels(dataset)
    corrs = np.zeros([train_labels.shape[1], len(identity_attributes)])
    pos_fracs = np.zeros([train_labels.shape[1], len(identity_attributes)])
    neg_fracs = np.zeros([train_labels.shape[1], len(identity_attributes)])
    for i in range(corrs.shape[0]):
        for j, attr in enumerate(identity_attributes):
            corrs[i,j] = np.corrcoef(train_labels[:,i], train_labels[:,attr])[1, 0]
            protected_pos_train = train_labels[np.argwhere(train_labels[:, attr] == 1), i]
            pos_fracs[i,j] = protected_pos_train.mean()
            protected_neg_train = train_labels[np.argwhere(train_labels[:, attr] == 0), i]
            neg_fracs[i,j] = protected_neg_train.mean()

    corrs_df = pd.DataFrame(corrs)
    corrs_df.columns = identity_attributes
    pos_fracs_df = pd.DataFrame(pos_fracs)
    pos_fracs_df.columns = identity_attributes
    neg_fracs_df = pd.DataFrame(neg_fracs)
    neg_fracs_df.columns = identity_attributes

    return corrs_df, pos_fracs_df, neg_fracs_df

def get_thresholds(outputs, labels):
    pos_per = np.sum(labels, axis=0)
    # Note that we do this on the raw outputs and not the sigmoids.
    thresholds = np.ones(pos_per.shape[0])
    for i in range(thresholds.shape[0]):
        thresholds[i] = \
                np.partition(outputs[:,i], -1*round(pos_per[i]))[-1*round(pos_per[i])]
    #np.argpartition(outputs[:,i], round(pos_per[i]))[round(pos_per[i])]
    #            np.partition(k.flatten(), -2)[-2]
    #raise ValueError(thresholds)
    return thresholds
    

def load_run_details(run, pos_fracs_df, neg_fracs_df, threshold_adjusted=False):
    print(run["run_dir"])
    test_labels = get_test_labels(run["dataset"])
    cached_path = os.path.join(run["run_dir"], "run_stats.pkl")
    if threshold_adjusted:
        cached_path = os.path.join(run["run_dir"], "thresholded_run_stats.pkl")
    if True  and os.path.exists(cached_path):
        print("USING THE CACHE")
        with open (cached_path, 'rb') as f:
            # TODO: make sure the existing parts of the run match
            run = pkl.load(f)
            return run
    elif os.path.exists(run["run_dir"] + "/" + "test_outputs.txt"):
        print("RECOMPUTING")
        thresholds = np.zeros(test_labels.shape[1])
        if threshold_adjusted:
            if not os.path.exists(run["run_dir"] + "/" + "valid_outputs.txt"):
                return run
            run["val_outputs"] = np.loadtxt(run["run_dir"] + "/" + "valid_outputs.txt")
            thresholds = get_thresholds(run["val_outputs"], get_val_labels(run["dataset"]))
        run["thresholds"] = thresholds
        run["test_outputs"] = np.loadtxt(run["run_dir"] + "/" + "test_outputs.txt")
        run["test_predictions"] = np.zeros(run["test_outputs"].shape)
        for i in range(test_labels.shape[1]):
            run["test_predictions"][:, i] = run["test_outputs"][:, i] > thresholds[i]
        run["sparsity"] = sparsity(run)
        run["strategy"] = strategy(run)
        run['type'] = desc(run)
        compute_bas(run, test_labels, pos_fracs_df, neg_fracs_df)
        compute_errors(run, test_labels)
        with open (cached_path, 'wb') as f:
            pkl.dump(run, f)
        return run
    else:
        return run


def load_partial_details(run):
    #dataset = preprocessed_rn18_runs[0]["dataset"]
    dataset = 'celeba'
    #covs_df, pos_fracs_df, neg_fracs_df = compute_cooccurrence_matrices(dataset)
    if 'celeba' in dataset:
        identity_labels = celeba_identity_labels
        attr_names = celeba_classes()
    else:
        identity_labels = awa_identity_labels
        attr_names = awa_classes()
    #attr_name = attr_names[attr]
    return load_run_details(run, 0,0,0)
    
    return preprocessed_rn18_runs


def make_runs_df(preprocessed_rn18_runs, threshold_adjusted=0):
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
    test_labels_df = pd.DataFrame(test_labels)
    test_labels_df.reset_index(inplace=True)
    test_labels_df.columns = ["example_id"] + [f"{attr_name}_label" for  attr_name in attr_names]
    print("there are this many runs before compute", len(preprocessed_rn18_runs))
    preprocessed_rn18_runs = [load_run_details(run, pos_fracs_df, neg_fracs_df, threshold_adjusted) for run in preprocessed_rn18_runs]
    
    preprocessed_rn18_runs = [v for v in preprocessed_rn18_runs if 'strategy' in v]
    print("there are this many runs", len(preprocessed_rn18_runs), preprocessed_rn18_runs[0].keys())
    preprocessed_rn18_runs.sort(key=lambda r: (r["sparsity"], r["group"]))

   
    # VLook for PIEs
    dicts = []
    for run in preprocessed_rn18_runs:
        mydict = {"seed": run["name"], "type": run["type"], \
            "strategy": run["strategy"], "sparsity": run["sparsity"]}
        mydict["example_id"] = [i for i in range(run["test_outputs"].shape[0])]
        for i, attr_name in enumerate(attr_names):
            mydict[f"{attr_name}_output"] = run["test_outputs"][:,i] 
            mydict[f"{attr_name}_prediction"] = run["test_predictions"][:,i] 

            #uncertainty=np.mean(np.abs(1/(1 + np.exp(-run["test_outputs"]))-0.5) < 0.4, axis=0)
            mydict[f"{attr_name}_prob"] = 1/(1 + np.exp(-run["test_outputs"][:,i]))
            mydict[f"{attr_name}_uncertain"] = np.abs((mydict[f"{attr_name}_prob"]-0.5)) < 0.4
            mydict[f"{attr_name}_very_uncertain"] = np.abs((mydict[f"{attr_name}_prob"]-0.5)) < 0.1
        run_df = pd.DataFrame(mydict)
        #return run_df.head(100)
        dicts.append(run_df)
    runs_df = pd.concat(dicts, ignore_index=True)
    groupers = {f"{attr_name}_prediction": "mean" for attr_name in attr_names}
    groupers = {**groupers,
            **{f"{attr_name}_output": "mean" for attr_name in attr_names},
            **{f"{attr_name}_prob": "mean" for attr_name in attr_names},
            **{f"{attr_name}_uncertain": "mean" for attr_name in attr_names},
            **{f"{attr_name}_very_uncertain": "mean" for attr_name in attr_names},
            }
    grouped = runs_df.groupby(['strategy', 'sparsity', 'example_id']).agg(groupers)
    grouped.reset_index(inplace=True)
    #raise ValueError(grouped.columns)
    for attr_name in attr_names:
        grouped[f'{attr_name}_prediction'] = grouped[f'{attr_name}_prediction'] > 0.5 
        grouped[f'{attr_name}_uncertain'] = grouped[f'{attr_name}_uncertain'] > 0.5 
        grouped[f'{attr_name}_very_uncertain'] = grouped[f'{attr_name}_very_uncertain'] > 0.5 
    #grouped = runs_df.groupby(['strategy', 'sparsity', 'example_id'])['Blond_Hair_prediction'].agg({"Blond_Hair_prediction": ""})
    #grouped = pd.DataFrame(grouped)
    #grouped['Blond_Hair_prediction'] = grouped['Blond_Hair_prediction'] > 0.5
    #print(grouped)
    dense_grouped = grouped[grouped.sparsity==0]
    #return(dense_grouped.head(100))
    sp995_grouped = grouped.query("sparsity==99.5 and strategy=='GMP-RI'")
    combined = pd.merge(dense_grouped, sp995_grouped, on='example_id')
    uncertain = {}
    high_level = []
    for attr_name in attr_names:
        if "Shadow" in attr_name:
            continue
        disagreements = combined.query(f"{attr_name}_prediction_x != {attr_name}_prediction_y")
        #print(attr_name, "number of disagreements/dense uncertain/highly uncertain", len(disagreements))
        #print("Prop. uncertain", np.abs((disagreements[f"{attr_name}_prob_x"] - 0.5) < 0.4).mean())
        #print("Prop. very uncertain", np.abs((disagreements[f"{attr_name}_prob_x"] - 0.5) < 0.1).mean())
        disagreements["agg_uncertain"] = np.abs((disagreements[f"{attr_name}_prob_x"] - 0.5) < 0.4)
        disagreements["agg_very_uncertain"] = np.abs((disagreements[f"{attr_name}_prob_x"] - 0.5) < 0.1)
        #return(disagreements.head(50))
        annotated = pd.merge(disagreements, test_labels_df, on='example_id', how='left') 
        #print("male label mean", annotated["Male_label"].mean(), test_labels_df["Male_label"].mean())
        #print("young label mean", annotated["Young_label"].mean(), test_labels_df["Young_label"].mean())

        high_level.append({
            "attr_name": attr_name,
            "pie_prop_uncertain_dense": np.abs((disagreements[f"{attr_name}_prob_x"] - 0.5) < 0.4).mean(), 
            "pie_prop_uncertain_dense2": disagreements["agg_uncertain"].mean(), 
            "pie_prop_very_uncertain_dense": np.abs((disagreements[f"{attr_name}_prob_x"] - 0.5) < 0.1).mean(),
            "prop_uncertain_dense": dense_grouped[f"{attr_name}_uncertain"].mean(), 
            "prop_very_uncertain_dense": dense_grouped[f"{attr_name}_very_uncertain"].mean(), 

            "num_pies" : len(disagreements),
            "prop_pos_to_neg" : (disagreements[f"{attr_name}_prediction_x"] > disagreements[f"{attr_name}_prediction_y"]).mean() 
                })
        uncertain[attr_name] = annotated
        #uncertain[attr_name] = {"uncertain": np.abs((disagreements[f"{attr_name}_prob_x"] - 0.5) < 0.4).mean(), "very_uncertain":  np.abs((disagreements[f"{attr_name}_prob_x"] - 0.5) < 0.1).mean()}

    return pd.DataFrame(high_level)



def compute_interdependence(preprocessed_rn18_runs, threshold_adjusted=0):
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
    test_labels_df = pd.DataFrame(test_labels)
    test_labels_df.reset_index(inplace=True)
    test_labels_df.columns = ["example_id"] + [f"{attr_name}_label" for  attr_name in attr_names]
    print("there are this many runs before compute", len(preprocessed_rn18_runs))
    preprocessed_rn18_runs = [load_run_details(run, pos_fracs_df, neg_fracs_df, threshold_adjusted) for run in preprocessed_rn18_runs]
    
    preprocessed_rn18_runs = [v for v in preprocessed_rn18_runs if 'strategy' in v]
    print("there are this many runs", len(preprocessed_rn18_runs), preprocessed_rn18_runs[0].keys())
    preprocessed_rn18_runs.sort(key=lambda r: (r["sparsity"], r["group"]))

    dicts = []
    for run in preprocessed_rn18_runs:
        mydict = {"seed": run["name"], "type": run["type"], \
            "strategy": run["strategy"], "sparsity": run["sparsity"]}
        mydict["example_id"] = [i for i in range(run["test_outputs"].shape[0])]
        for i, attr_name in enumerate(attr_names):
            mydict[f"{attr_name}_output"] = run["test_outputs"][:,i] 
            mydict[f"{attr_name}_prediction"] = run["test_predictions"][:,i] 

            #uncertainty=np.mean(np.abs(1/(1 + np.exp(-run["test_outputs"]))-0.5) < 0.4, axis=0)
            mydict[f"{attr_name}_prob"] = 1/(1 + np.exp(-run["test_outputs"][:,i]))
        df = pd.DataFrame(mydict)
        coefs = {"seed": run["name"], "type": run["type"], \
            "strategy": run["strategy"], "sparsity": run["sparsity"]}
        for attr_name in attr_names:
            feature_columns = [c for c in df.columns if 'prediction' in c and attr_name not in c]
            reg = LinearRegression().fit(df[feature_columns], df[f"{attr_name}_prediction"])
            coefs[f"{attr_name}_score"] = reg.score(df[feature_columns], df[f"{attr_name}_prediction"])
        dicts.append(coefs)
    coefs = {"seed": "Labels", "type": "Labels", \
            "strategy": "Labels", "sparsity": 0}
    for attr_name in attr_names:
        feature_columns = [c for c in test_labels_df.columns if attr_name not in c and "example_id" not in c]
        reg = LinearRegression().fit(test_labels_df[feature_columns], test_labels_df[f"{attr_name}_label"])
        coefs[f"{attr_name}_score"] = reg.score(test_labels_df[feature_columns], test_labels_df[f"{attr_name}_label"])
        dicts.append(coefs)
    dicts = pd.DataFrame(dicts).groupby(["strategy", "sparsity"]).agg("mean")
    return dicts


        



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
    combined_df.to_latex(filepath)
    return top_level_accuracies, highlevels, ba_dicts, fpr_dicts, fnr_dicts


