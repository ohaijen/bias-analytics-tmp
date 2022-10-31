from flask import *
import numpy as np
import pandas as pd
import utils.runs as runs_joint
import utils.single_label as runs_sl
import utils.iwildcam as iw
from utils.projects import PROJECTS
import io
import base64
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


import time
from io import BytesIO
import zipfile
import os


@app.route('/artifacts')
def artifacts():
    timestr = time.strftime("%Y%m%d-%H%M%S")
    fileName = "artifacts_{}.zip".format(timestr)
    memory_file = BytesIO()
    file_path = 'generated/'
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(file_path):
            for file in files:
                zipf.write(os.path.join(root, file))
    memory_file.seek(0)
    return send_file(memory_file, as_attachment=True, download_name=fileName)




#@app.route('/image/')
@app.route('/image/<path:dataset>/<path:index>')
def send_image(dataset, index):
    filename=index
    if dataset == "celeba":
        return send_from_directory("/home/Datasets/celeba/img_align_celeba", filename)
    elif dataset == "full_celeba":
        return send_from_directory("/home/Datasets/celeba/img_celeba", filename)
    else:
        raise ValueError(f"don't know how to show images for f{dataset} ")


@app.route('/run_names')
def show_run_names():
    dataset = request.args.get("dataset", default="celeba", type=str)
    arch = request.args.get("arch", default="resnet18", type=str)
    short_arch = arch
    if arch == "resnet18":
        short_arch = "rn18"
    project_name = f"{dataset}-all-{short_arch}"
    if project_name not in PROJECTS:
        raise ValueError(f"Project {project_name} doesn't exist.")

    #counts = runs_joint.get_run_counts(project_name).to_html(table_id="counts")
    runs = runs_joint.get_runs_for_project(project_name)
    runs_df = pd.DataFrame(runs)
    return render_template("generic_table.html", title=f"{project_name} Runs", table=runs_df.to_html(table_id="run_names"))

@app.route('/single_run_names')
def show_single_run_names():
    dataset = request.args.get("dataset", default="celeba", type=str)
    arch = request.args.get("arch", default="resnet18", type=str)
    backdoor = request.args.get("backdoor", default=False, type=bool)
    short_arch = arch
    if arch == "resnet18":
        short_arch = "rn18"
    if backdoor:
        project_name = f"{dataset}-backdoor-single-{short_arch}"
    else:
        project_name = f"{dataset}-single-{short_arch}"
    #return f"<html>{backdoor} {project_name}</html>"
    if project_name not in PROJECTS:
        raise ValueError(f"Project {project_name} doesn't exist.")
    runs = runs_sl.get_runs_for_project(project_name)
    


    print(runs.keys())
    print(runs["smiling"][0])
    runs_df = pd.concat([pd.DataFrame(v) for v in runs.values()])
    return render_template("generic_table.html", title=f"{project_name} Runs", table=runs_df.to_html(table_id="run_names"))


@app.route('/classes')
def show_classes():
    dataset = request.args.get("dataset", default="celeba", type=str)
    if 'celeba' in dataset:
        classes = runs_joint.celeba_classes()
    elif 'awa' in dataset:
        classes = runs_joint.awa_classes()
    else:
        classes = []
    numbered_classes = pd.DataFrame([[i,k] for i, k in enumerate(classes)], columns=["number", "class"])
    return render_template("generic_table.html", title=f"{dataset} Classes", table=numbered_classes.to_html(table_id="classes"))


@app.route("/debug")
def debug():
    corrs, pos_fracs, neg_fracs = runs_joint.compute_cooccurrence_matrices("celeba")
    # dataset = request.args.get("dataset", default="celeba", type=str)
    # arch = request.args.get("arch", default="resnet18", type=str)
    # short_arch = arch
    # if arch == "resnet18":
    #     short_arch = "rn18"
    # project_name = f"{dataset}-all-{short_arch}"
    # if project_name not in PROJECTS:
    #     raise ValueError(f"Project {project_name} doesn't exist.")

    # counts = runs_joint.get_run_counts(project_name).to_html(table_id="counts")
    # rn18_runs = runs_joint.get_runs_for_project(project_name)


    # top_level_accs, high_level, ba_splits, fpr_splits, fnr_splits = runs_joint.get_run_summaries(rn18_runs)
    # 
    # debug = runs_joint.compute_worst(ba_splits)
    return render_template("generic_table.html", title=f" CELEBA", table=pos_fracs.to_html())
    


@app.route("/pies")
def get_pies():
    dataset = request.args.get("dataset", default="celeba", type=str)
    arch = request.args.get("arch", default="resnet18", type=str)
    short_arch = arch
    if arch == "resnet18":
        short_arch = "rn18"
    project_name = f"{dataset}-all-{short_arch}"
    if project_name not in PROJECTS:
        raise ValueError(f"Project {project_name} doesn't exist.")
    threshold_adjusted = request.args.get("threshold_adjusted", default=False, type=bool)
    rn18_runs = runs_joint.get_runs_for_project(project_name)
    runs_df = runs_joint.make_runs_df(rn18_runs, threshold_adjusted=threshold_adjusted)
    return(runs_df.to_html())

@app.route("/interdependence")
def get_inters():
    dataset = request.args.get("dataset", default="celeba", type=str)
    arch = request.args.get("arch", default="resnet18", type=str)
    short_arch = arch
    if arch == "resnet18":
        short_arch = "rn18"
    project_name = f"{dataset}-all-{short_arch}"
    if project_name not in PROJECTS:
        raise ValueError(f"Project {project_name} doesn't exist.")
    threshold_adjusted = request.args.get("threshold_adjusted", default=False, type=bool)
    rn18_runs = runs_joint.get_runs_for_project(project_name)
    runs_df = runs_joint.compute_interdependence(rn18_runs, threshold_adjusted=threshold_adjusted)
    return("<h1>Ability to predict one label from the others (R^2 coeff from linear regression)</h1>" + runs_df.transpose().to_html())

@app.route('/runs')
def show_runs():
    dataset = request.args.get("dataset", default="celeba", type=str)
    arch = request.args.get("arch", default="resnet18", type=str)
    short_arch = arch
    if arch == "resnet18":
        short_arch = "rn18"
    project_name = f"{dataset}-all-{short_arch}"
    if project_name not in PROJECTS:
        raise ValueError(f"Project {project_name} doesn't exist.")
    threshold_adjusted = request.args.get("threshold_adjusted", default=False, type=bool)

    counts = runs_joint.get_run_counts(project_name).to_html(table_id="counts")
    rn18_runs = runs_joint.get_runs_for_project(project_name)
    


    top_level_accs, high_level, ba_splits, fpr_splits, fnr_splits = runs_joint.get_run_summaries(rn18_runs, arch=arch, threshold_adjusted=threshold_adjusted)
    accs = high_level[1][1]
    corrs = runs_joint.compute_cooccurrence_matrices(dataset)[0] 

    hls = [{"name": name, "table": averages_df.to_html(table_id=f'myTable{name}')} for [name, averages_df] in high_level ]
    ba_res = []
    for k, [df, averages_df, averages_diffs_df] in ba_splits.items():
        #column_df = averages_df.reindex(["strategy", "sparsity"])
        column_df = averages_df.copy()
        column_df['sparsity'] = column_df['sparsity'].astype("string")
        column_df["type"] = column_df['strategy'] + "-" + column_df['sparsity']
        columns = [c for c in column_df.columns if c not in ("strategy", "sparsity")]
        column_df = column_df[columns]
        column_df.set_index("type", inplace=True)
        column_df = column_df.transpose()
        print(averages_df)
        ba_res.append({"attr": k, "table": column_df.to_html(table_id=k), "plot_url": runs_joint.generate_metric_plot(averages_df, df, metric_name = f"{k} Bias Amplification", arch=arch, threshold_adjusted=threshold_adjusted),
            "detail_plot_url": runs_joint.generate_detailed_plot(averages_df, df, accs=accs, corrs = corrs,  metric_name = "Bias Amplification", arch=arch, threshold_adjusted=threshold_adjusted)})

    fpr_res = []
    for k, [df, averages_df, averages_diffs_df] in fpr_splits.items():
        #column_df = averages_df.reindex(["strategy", "sparsity"])
        print("the disp columns are", averages_df.columns)
        column_df = averages_df.copy()
        column_df['sparsity'] = column_df['sparsity'].astype("string")
        column_df["type"] = column_df['strategy'] + "-" + column_df['sparsity']
        print("the FPR COLUMN DF IS", column_df, column_df["type"])
        columns = [c for c in column_df.columns if c not in ("strategy", "sparsity")]
        column_df = column_df[columns]
        column_df.set_index("type", inplace=True)
        column_df = column_df.transpose()
        fpr_res.append({"attr": k, "table": column_df.to_html(table_id=f'fpr-diff-{k}'),
            "plot_url": runs_joint.generate_metric_plot(averages_df, df, metric_name = "FPR Difference", arch=arch, threshold_adjusted=threshold_adjusted),
            "detail_plot_url": runs_joint.generate_detailed_plot(averages_df, df, accs = accs, corrs = corrs, metric_name = "FPR Difference", arch=arch, threshold_adjusted=threshold_adjusted)})
        #plot_url = runs_joint.generate_metric_plot(averages_df)
        #table = averages_df.to_html()
    fnr_res = []
    for k, [df, averages_df, averages_diffs_df] in fnr_splits.items():
        #column_df = averages_df.reindex(["strategy", "sparsity"])
        print("the disp columns are", averages_df.columns)
        column_df = averages_df.copy()
        column_df['sparsity'] = column_df['sparsity'].astype("string")
        column_df["type"] = column_df['strategy'] + "-" + column_df['sparsity']
        print("the FNR COLUMN DF IS", column_df, column_df["type"])
        columns = [c for c in column_df.columns if c not in ("strategy", "sparsity")]
        column_df = column_df[columns]
        column_df.set_index("type", inplace=True)
        column_df = column_df.transpose()
        fnr_res.append({"attr": k, "table": column_df.to_html(table_id=f'fnr-diff-{k}'),
            "plot_url": runs_joint.generate_metric_plot(averages_df, df,  metric_name = "FNR Difference", arch=arch, threshold_adjusted=threshold_adjusted),
            "detail_plot_url": runs_joint.generate_detailed_plot(averages_df, df, accs=accs, corrs = corrs, metric_name = "FNR Difference", arch=arch, threshold_adjusted=threshold_adjusted)})
    #return " ".join([run["group"] for run in rn18_runs if "995" in run["group"]])
    return render_template("runs.html", counts_table = counts, top_level_accs = top_level_accs.to_html(table_id = "top_level_accs"), summary_table = hls, per_attr_bas = ba_res, per_attr_fprs = fpr_res, per_attr_fnrs = fnr_res)



@app.route("/single_runs")
def show_single_runs():
    dataset = request.args.get("dataset", default="celeba", type=str)
    arch = request.args.get("arch", default="resnet18", type=str)
    backdoor = request.args.get("backdoor", default=False, type=bool)
    combined = request.args.get("combined", default=False, type=bool)
    short_arch = arch
    if arch == "resnet18":
        short_arch = "rn18"
    if backdoor:
        project_name = f"{dataset}-backdoor-single-{short_arch}"
    elif combined:
        project_name = f"{dataset}-combined-{short_arch}"
    else:
        project_name = f"{dataset}-single-{short_arch}"
    #return f"<html>{backdoor} {project_name}</html>"
    if project_name not in PROJECTS:
        raise ValueError(f"Project {project_name} doesn't exist.")
    rs = runs_sl.get_runs_for_project(project_name)
    runs, accs, fprs, fnrs, pred_poss = runs_sl.get_run_summaries(rs, dataset, backdoor)
    plots = {}
    for attr in runs.keys():
        if combined:
            plots[attr] = runs_sl.plot_combined_runs_metrics(runs, attr)
        else:
            plots[attr] = runs_sl.plot_single_label_metrics(runs, attr, backdoor=backdoor)

    return render_template("single_runs.html", 
            run_counts= {k: v.to_html(table_id=f"{k}_counts") for k, v in runs_sl.get_run_counts(project_name).items()},
            run_accs= {k: v.to_html(table_id=f"{k}_means") for k, v in accs.items()},
            run_fprs= {k: v.to_html(table_id=f"{k}_means") for k, v in fprs.items()},
            run_fnrs= {k: v.to_html(table_id=f"{k}_means") for k, v in fnrs.items()},
            run_pred_poss= {k: v.to_html(table_id=f"{k}_means") for k, v in pred_poss.items()},
            plots = plots
            )

@app.route("/iwildcam")
def show_iwildcam():
    runs  = iw.get_iwildcam_runs()
    return render_template("iwildcam.html", title=f"iWildcam", table=runs.groupby(["group", "class_id"]).agg("mean").to_html(table_id="run_names"), plots=iw.plot_wilds_metrics(runs))

@app.route("/examples")
def show_a_pic():
    dataset = request.args.get("dataset", default="celeba", type=str)
    arch = request.args.get("arch", default="resnet18", type=str)
    label = request.args.get("label", default="", type=str)
    run_type = request.args.get("run_type", default="0-Dense", type=str)
    label_id = runs_joint.celeba_classes().index(label)
    labels = runs_joint.get_test_labels(dataset)
    labels = labels[:, label_id]

    short_arch = arch
    if arch == "resnet18":
        short_arch = "rn18"
    project_name = f"{dataset}-all-{short_arch}"
    if project_name not in PROJECTS:
        raise ValueError(f"Project {project_name} doesn't exist.")
    rn18_runs = runs_joint.get_runs_for_project(project_name)
    matching_runs = [r for r in rn18_runs if r['type'] == run_type]
    print(matching_runs[0])
    found = False
    run = None
    for mr in matching_runs:
        mr = runs_joint.load_partial_details(mr)
        if 'test_predictions' in mr:
            found = True
            run = mr
            break


    preds = run["test_predictions"][:,label_id]
    scores = 1/(1+np.exp(-1*run["test_outputs"][:,label_id]))
    true_positives = np.nonzero(preds*labels > 0)[0]
    false_positives = np.nonzero(preds*(1-labels) > 0)[0]
    true_negatives = np.nonzero((1-preds)*(1-labels) > 0)[0]
    false_negatives = np.nonzero((1-preds)*(labels) > 0)[0]
    splits = [true_positives, false_positives,true_negatives,  false_negatives]
    print([len(s) for s in splits])
    print(np.random.choice(splits[0], 20))

    desired=20
    splits = [np.random.choice(s, desired, replace=False) for s in splits]
    

    image_ids = runs_joint.get_test_image_ids(dataset)#[indices]
    def bool_to_str(b):
        if b == False:
            return False
        return True
    def bool_to_color(b):
        if b == False:
            return "red"
        return "green"

    def index_to_display_tuple(i):
        return {"idx": image_ids[i],
                "label": f'{image_ids[i]}: {labels[i]} / pred. {preds[i]}  {scores[i]}',
                "color": bool_to_color(labels[i] == preds[i]) 
                }

    images = {
        "True Positives":  [index_to_display_tuple(i) for i in splits[0]],
        "False Positives": [index_to_display_tuple(i) for i in splits[1]],
        "True Negatives":  [index_to_display_tuple(i) for i in splits[2]],
        "False Negatives": [index_to_display_tuple(i) for i in splits[3]],
            }
    print(images)
    return render_template('img.html', dataset=dataset, images = images)





# Contrasting two models
@app.route("/examples/<path:label>/<path:run_type>/<path:run_type2>")
def compare_two_runs(label, run_type, run_type2):
    label_id = runs_joint.celeba_classes().index(label)
    labels = pd.read_csv("/home/Datasets/celeba/list_attr_celeba.txt", delim_whitespace=True, header=1)
    labels = runs_joint.get_test_labels()
    labels = labels[:, label_id]

    rn18_runs = runs_joint.get_runs_for_project("celeba_rn18_bias")
    matching_runs = [r for r in rn18_runs if r['type'] == run_type]
    found = False
    run = None
    for mr in matching_runs:
        mr = runs_joint.load_run_details(mr)
        if 'test_predictions' in mr:
            found = True
            run = mr
            break

    matching_runs2 = [r for r in rn18_runs if r['type'] == run_type2]
    found2 = False
    run2 = None
    for mr in matching_runs2:
        mr = runs_joint.load_run_details(mr)
        if 'test_predictions' in mr:
            found2 = True
            run2 = mr
            break

    preds2 = run2["test_predictions"][:, label_id]
    preds = run["test_predictions"][:,label_id]
    scores = run["test_outputs"][:,label_id]
    both_positive = np.nonzero(preds*preds2 > 0)[0]
    both_negative = np.nonzero((1-preds)*(1-preds2) > 0)[0]
    first_right_positive = np.nonzero(preds*(1-preds2)*labels > 0)[0]
    first_right_negative = np.nonzero((1-preds)*preds2*(1-labels) > 0)[0]
    second_right_positive = np.nonzero((1-preds)*(preds2)*labels > 0)[0]
    second_right_negative = np.nonzero(preds*(1-preds2)*(1-labels) > 0)[0]
    splits = [both_positive, both_negative,
             first_right_positive, first_right_negative,
             second_right_positive, second_right_negative]
    print([len(s) for s in splits])

    desired=20
    splits = [np.random.choice(s, desired, replace=False) for s in splits]
    

    indices = np.concatenate(splits)
    image_ids = runs_joint.get_test_image_ids()#[indices]
    def bool_to_str(b):
        if b == False:
            return False
        return True
    def bool_to_color(b):
        if b == False:
            return "red"
        return "green"

    colors = [bool_to_color(l)  for l in labels[indices]]
    labels = [f'{image_ids[i]}: {labels[i]} / Model1. {preds[i]}  Model2. {preds2[i]}' for i in indices]

    return render_template('img.html', indices=[x for x in range(len(labels))], image_ids=image_ids[indices], labels = labels, colors=colors)
