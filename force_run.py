

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

from utils.projects import PROJECTS
import utils.runs as runs_joint
import utils.single_label as runs_sl


for project_name, project_descr  in PROJECTS.items():
    if "all" in project_name:
        continue
        runs = runs_joint.get_runs_for_project(project_name)
        summaries = runs_joint.get_run_summaries(runs)
    elif "single" in project_name:
        runs = runs_sl.get_runs_for_project(project_name)
        runs, accs, fprs, fnrs, pred_poss = runs_sl.get_run_summaries(runs,
                project_descr["dset"], backdoor = 'backdoor' in project_name)

