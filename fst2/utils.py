import os
import yaml
import torch
import numpy as np
import random
from .default_configs import TASK_CONFIGS
from typing import Dict

CACHE_PARAMS = {
    "data_dir": None,
    "model_name_or_path": None,
    "max_seq_length": None,
    "mode": None,
}


SUPPORTED_TASKS = ["ner", "textclassification"]

tasks_id_mapping = dict(zip(SUPPORTED_TASKS, range(1, len(SUPPORTED_TASKS)+1)))

reverse_tasks_id_mapping = {v: k for k, v in tasks_id_mapping.items()}


def gen_template(task_name: str) -> Dict:
    """
        Wrtite yaml configs to the current directory

        Args:
            task_name: task name, e.g ner
        Return:
            dict of args     
    """
    # config_name = os.path.join("default_configs", task_name+".yml")
    # with open(config_name, "r", encoding="utf-8") as f:
    #     return yaml.load(f, Loader=yaml.FullLoader)
    return TASK_CONFIGS[task_name]


def gen_configs(args): 
    """
    Interactively generate configs according task name.
    """
    while True:
        task_id = input(
            """
              Please enter a corresponding number for your NLP training task(0 to quit):
              1 ner(Named Entity Recognition)
              2 textclassification
            """
        )
        try:
            task_id = int(task_id)
        except ValueError:
            print("\nPlease enter a numer!")
            continue     
        if task_id == 0:
            break    
        if task_id not in list(range(1, len(SUPPORTED_TASKS)+1)):
            print("\nNumber you entered was out of range!")
            continue
        task_name = reverse_tasks_id_mapping[task_id]
        config = gen_template(task_name)  # dict
        # Based on args train some of the configs
        if args.parent_dirname:
            config["data"]["data_dir"] = os.path.join(args.parent_dirname, "data")
            config["model"]["model_name_or_path"] = os.path.join(args.parent_dirname, "pretrained_model")
            config["train"]["output_dir"] = os.path.join(args.parent_dirname, "output_dir")
            config["data"]["label_file"] = os.path.join(args.parent_dirname, "data", "labels.txt")
            config["train"]["result_dir"] = os.path.join(args.parent_dirname, "result_dir")
        # prepare dir for tensorboard runs
        config["others"]["tensorboard_dir"] = os.path.join("runs", config["pipeline"]["task"])
        # write config to current dir
        with open("configs.yml", 'w', encoding='utf-8') as file:
            yaml.dump(config, file)
        print(f"configs for task {task_name} was generated! Please check configs.yml under curret directory")
        break


def gen_dirs(args):
    """
      generate a recommended directory structure

    """
    parent_dirname = args.parent_dirname
    if os.path.exists(parent_dirname):
        exit(f"\nWARING:{parent_dirname} alread exited chose another directory name")
    os.mkdir(parent_dirname)
    dirs_to_gen = ["data", "pretrained_model", "output_dir", "result_dir"]
    for dir in dirs_to_gen:
        directory = os.path.join(parent_dirname, dir)
        os.mkdir(directory)
    print("recommended directory for nlp train was generated")


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def get_tokenclassification_labels(path):
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else: 
        raise ValueError(f"label file {path} not existed")


def get_textclassification_labels(path):
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        return labels
    else:
        raise ValueError(f"label file {path} not existed")


def get_id2label(labels):
    """
    Get id2label mapping based on labels

    Args:
      labels: list of labels.

    Return:
      id2label map
    """
    return {str(k): v for k, v in enumerate(labels)}
    

def get_label2id(labels):
    """
    Get label2id mapping based on labels

    Args:
      labels: list of labels.

    Return:
      label2id map
    """
    return {v: str(k) for k, v in enumerate(labels)}

