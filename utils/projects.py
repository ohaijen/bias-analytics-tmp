PROJECTS = {
        "celeba-all-rn18":{
            "wandb_project": "celeba_rn18_bias",
            "dset": "celeba",
            "arch": "resnet18",
            "labels": "all",
            "backdoor": False,
            "combined": False
            },
        "celeba-all-mobilenet":{
            "wandb_project": "celeba_mobilenet_bias",
            "dset": "celeba",
            "arch": "mobilenet",
            "labels": "all",
            "backdoor": False,
            "combined": False
            },
        "celeba-single-rn18":{
            "wandb_project": "celeba_rn18_bias_single_attr",
            "dset": "celeba",
            "arch": "resnet18",
            "labels": ["Blond", "Smiling"],
            "backdoor": False,
            "combined": False
            },
        "celeba-single-mobilenet":{
            "wandb_project": "celeba_mobilenet_bias_single_label",
            "dset": "celeba",
            "arch": "mobilenet",
            "labels": ["Blond", "Smiling"],
            "backdoor": False,
            "combined": False
            },
        "celeba-backdoor-single-rn18":{
            "wandb_project": "celeba_rn18_backdoor_bias_single_label",
            "dset": "celeba",
            "arch": "resnet18",
            "labels": ["Blond", "Smiling"],
            "backdoor": True,
            "combined": False
            },
        "celeba-combined-rn18":{
            "wandb_project": "celeba_rn18_bias_combined_attr",
            "dset": "celeba",
            "arch": "resnet18",
            "labels": ["Blond", "Smiling"],
            "backdoor": False,
            "combined": True
            },
        "celeba-backdoor-single-mobilenet":{
            "wandb_project": "celeba_mobilenet_backdoor_bias_single_label",
            "dset": "celeba",
            "arch": "mobilenet",
            "labels": ["Blond", "Smiling"],
            "backdoor": True,
            "combined": False
            },
        "full_celeba-all-rn18":{
            "wandb_project": "full_celeba_rn18_bias",
            "dset": "full_celeba",
            "arch": "resnet18",
            "labels": "all",
            "backdoor": False,
            "combined": False
            },
        "full_celeba-all-mobilenet":{
            "wandb_project": "celeba_mobilenet_bias",
            "dset": "full_celeba",
            "arch": "mobilenet",
            "labels": "all",
            "backdoor": False,
            "combined": False
            },
        "awa-all-rn18":{
            "wandb_project": "awa40_rn18_bias",
            "dset": "awa2",
            "arch": "resnet18",
            "labels": "all",
            "backdoor": False,
            "combined": False
            },
        "awa-all-mobilenet":{
            "wandb_project": "awa40_mobilenet_bias",
            "dset": "awa2",
            "arch": "mobilenet",
            "labels": "all",
            "backdoor": False,
            "combined": False
            },

        }


