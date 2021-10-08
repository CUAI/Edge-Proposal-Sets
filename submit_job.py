import argparse
import subprocess
from pathlib import Path
from subprocess import PIPE
from datetime import datetime
import os
from multiprocessing import Process
from rank import get_dataset
import numpy as np
import torch
    
def is_no_param_models_bydefault(dataset, model):
    return model in ['adamic', 'simple', 'adamic_ogb'] or (dataset in ["collab" , "reddit"] and model in ["simplecos"])
    
def gen_model(dataset, model, model_args = ""):
    # only generate one filter model by default
    return f'python -u rank.py --dataset {dataset} --model {model} {model_args} --save_models --runs 1'


def filter_step(dataset, model, checkpoint, model_args = ""):
    return f'python -u filter.py --dataset {dataset} --model {model} --checkpoint \"{checkpoint}\"' 

def job_separator():
    return " && "


def local_run(job_name, python_script):
    time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    out_file_name = f'logs/{job_name}.o%local{time}'
    f = open(out_file_name, "w")
    f.write(python_script + "\n")
    f.write("=====================================================\n\n")
    f.flush()
    rc = subprocess.run(python_script, shell=True, stdout=f, stderr=f) 
    f.close()

def sbatch_run(python_script, job_name):
    out_file_name = f'logs/{job_name}.o%j'
    sbatch_script = f'sbatch --requeue -N 1 -c 4 --mem 30000 -t 72:00:00 --partition=cuvl --gres=gpu:1 -J {job_name} -o {out_file_name} -e {out_file_name}'
    sbatch_script = fr'{sbatch_script} --wrap="printf \"{python_script}\n\" {job_separator()} {python_script}"'
    rc = subprocess.run(sbatch_script, shell=True, stdout=PIPE, stderr=PIPE)

    
    
def prepare_base_and_filter_results(dataset, filter_model):
    checkpoint = f"{dataset}_{filter_model}||0|0.pt"
    curve = f"{dataset}_{filter_model}||0|9.pt"
    filter_result = f"{dataset}_{filter_model}__0_0_sorted_edges.pt"
    python_script = ""
    
    if not filter_result in os.listdir('filtered_edges'):
        python_script = filter_step(dataset, filter_model, checkpoint)

        if not checkpoint in os.listdir('models') or (is_no_param_models_bydefault(dataset, filter_model) and not curve in os.listdir('curves')):
            python_script = gen_model(dataset, filter_model) + job_separator() +  python_script
    return python_script
    
    
def submit_experiments(run_experiment, experiment_configs, run_local):
    
    # repare steps
    processes = []
    has_sbatch_job = False
    for dataset,jobs in experiment_configs.items():
        # make sure the dataset exists
        print("Checking ", dataset)
        get_dataset(dataset)
        base_models = np.unique([job[0] for job in jobs])
        for base_model in base_models:
            # prepare experiment in sync
            python_script = prepare_base_and_filter_results(dataset, base_model)
            if python_script == "":
                continue
            job_name = f'{dataset}_{base_model}_generation'
            
            if run_local:
                p = Process(target=local_run, args=(job_name, python_script))
                print("submit gen job",dataset, base_model)
                p.start()
                processes.append(p)
            else:
                has_sbatch_job = True
                sbatch_run(python_script, job_name)
                print("submit gen job through sbatch",dataset,base_model)
                ## Submit!!
    if run_local:
        print("Waiting for dataset and base model jobs to finish")
        for p in processes:
            p.join()
    elif has_sbatch_job:
        print("Need to wait fot sbatch job to finish. Please run the same command again after the jobs are done")
        return
    print("Preparations Done")
            
    # run the actual experiments
    processes = []       
    for dataset,jobs in experiment_configs.items():
        # make sure the dataset exists
        get_dataset(dataset)
        for job in jobs:
            python_script = run_experiment(dataset, *job)
            job_name = f'{dataset}_{"_".join(str(x) for x in job)}'
            
            if run_local:
                p = Process(target=local_run, args=(job_name, python_script))
                print("submit experiment",dataset,job)
                p.start()
                processes.append(p)
            else:
                sbatch_run(python_script, job_name)
                print("submit experiment through sbatch",dataset,job)
                ## Submit!!
    if run_local:
        for p in processes:
            p.join()

def print_result(dataset, filter_method, rank_method, num, kind):
    result_files = [f for f in os.listdir('curves') if f.startswith(dataset + "_" + rank_method + kind + "|" + dataset + "_" + filter_method + "__0_0_sorted_edges|" + str(num) +"|")]
    curves = np.array([torch.load(f'curves/{c}') for c in result_files])
    means = curves.mean(0)
    stds = curves.std(0)
    print(f"Final Test: {means[2]:.2f} ± {stds[2]:.2f}")
    
###################################### customize experiments ######################################

def run_standard_experiment(dataset, filter_model, rank_model, num, runs):
    filter_result = f"{dataset}_{filter_model}__0_0_sorted_edges.pt"
    python_script = f'python -u rank.py --dataset {dataset} --model {rank_model} --num_sorted_edge {num} --sorted_edge_path {filter_result} --runs {runs}'     
    return python_script

def run_only_supervision_experiment(dataset, filter_model, rank_model, num):
    filter_result = f"{dataset}_{filter_model}__0_0_sorted_edges.pt"
    python_script = f'python -u rank.py --dataset {dataset} --model {rank_model} --num_sorted_edge {num} --sorted_edge_path {filter_result} --runs 10 --only_supervision'     
    return python_script

def run_also_supervision_experiment(dataset, filter_model, rank_model, num):
    filter_result = f"{dataset}_{filter_model}__0_0_sorted_edges.pt"
    python_script = f'python -u rank.py --dataset {dataset} --model {rank_model} --num_sorted_edge {num} --sorted_edge_path {filter_result} --runs 10 --also_supervision'     
    return python_script 

def run_standard_valid_experiment(dataset, filter_model, rank_model, num, runs):
    filter_result = f"{dataset}_{filter_model}__0_0_sorted_edges.pt"
    python_script = f'python -u rank.py --dataset {dataset} --model {rank_model} --num_sorted_edge {num} --sorted_edge_path {filter_result} --runs {runs} --valid_proposal'     
    return python_script


def run_with_emb_experiment(dataset, filter_model, rank_model, sweep_min, sweep_max, sweep_num, runs):
    filter_result = f"{dataset}_{filter_model}__0_0_sorted_edges.pt"
    python_script = f'python -u rank.py --dataset {dataset} --model {rank_model} --sorted_edge_path {filter_result} --runs {runs} --sweep_min {sweep_min} --sweep_max {sweep_max} --sweep_num {sweep_num} --use_learnable_embedding True --use_feature True --out_name {dataset+"_"+rank_model+"_embedding"}'     
    return python_script
        
def run_sweep_experiment(dataset, filter_model, rank_model, sweep_min, sweep_num, runs, sweep_max):
    filter_result = f"{dataset}_{filter_model}__0_0_sorted_edges.pt"
    python_script = f'python -u rank.py --dataset {dataset} --model {rank_model} --sweep_min {sweep_min} --sweep_max {sweep_max} --sweep_num {sweep_num} --sorted_edge_path {filter_result} --runs {runs}'    
    return python_script

def run_sweep_valid_experiment(dataset, filter_model, rank_model, sweep_min, sweep_num, runs, sweep_max):
    filter_result = f"{dataset}_{filter_model}__0_0_sorted_edges.pt"
    python_script = f'python -u rank.py --dataset {dataset} --model {rank_model} --sweep_min {sweep_min} --sweep_max {sweep_max} --sweep_num {sweep_num} --sorted_edge_path {filter_result} --runs {runs} --valid_proposal'    
    return python_script

def run_only_supervision_sweep_experiment(dataset, filter_model, rank_model, sweep_min, sweep_num, runs, sweep_max):
    filter_result = f"{dataset}_{filter_model}__0_0_sorted_edges.pt"
    python_script = f'python -u rank.py --dataset {dataset} --model {rank_model} --sweep_min {sweep_min} --sweep_max {sweep_max} --sweep_num {sweep_num} --sorted_edge_path {filter_result} --runs {runs} --only_supervision'    
    return python_script

def run_also_supervision_sweep_experiment(dataset, filter_model, rank_model, sweep_min, sweep_num, runs, sweep_max):
    filter_result = f"{dataset}_{filter_model}__0_0_sorted_edges.pt"
    python_script = f'python -u rank.py --dataset {dataset} --model {rank_model} --sweep_min {sweep_min} --sweep_max {sweep_max} --sweep_num {sweep_num} --sorted_edge_path {filter_result} --runs {runs} --also_supervision'    
    return python_script


def run_baseline_experiment(dataset, filter_model, rank_model, num):
    filter_result = f"{dataset}_{filter_model}__0_0_sorted_edges.pt"
    python_script1 = f'python -u rank.py --dataset {dataset} --model {rank_model} --num_sorted_edge {num} --sorted_edge_path {filter_result} --only_supervision' 
    python_script2 = f'python -u rank.py --dataset {dataset} --model {rank_model} --num_sorted_edge {num} --sorted_edge_path {filter_result} --also_supervision' 
    return python_script1 + job_separator() + python_script2

            
            
if __name__ == "__main__":
    Path("logs").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("curves").mkdir(exist_ok=True)
    Path("filtered_edges").mkdir(exist_ok=True)
    run_local = True
    parser = argparse.ArgumentParser(description='models')
    parser.add_argument('--reproduce', type=str)
    parser.add_argument('--show', type=str)
    args = parser.parse_args()
    #### note: it is assumed that the first argument is base model that needs to be generated first
    if args.reproduce == "ddi":
        ddi_experiments_configs = {"ddi": [
                              ("gcn", "sage", 530000, 10),
                           ],
                                    }    
        submit_experiments(run_standard_experiment, ddi_experiments_configs, run_local)        
        
    if args.reproduce == "collab":
        collab_experiments_configs = {"collab": [
                              # only run once to save time since it is deterministic 
                              ("adamic_ogb", "adamic_ogb", 200000, 1),
                           ],
                                    }    
        submit_experiments(run_standard_valid_experiment, collab_experiments_configs, run_local)       
        
    if args.reproduce == "ppa":
        ppa_experiments_configs = { "ppa": [
                               # only run once to save time since it is deterministic 
                              ("resource_allocation", "resource_allocation", 4000000, 1),
                            ],
                                    }    
        submit_experiments(run_standard_experiment, ppa_experiments_configs, run_local)
        

    if args.show == "ddi":
        print_result("ddi", "gcn", "sage", 530000, "")
    if args.show == "collab":
        print_result("collab", "adamic_ogb", "adamic_ogb", 200000, "_validproposal")
    if args.show == "ppa":
        print_result("ppa", "resource_allocation", "resource_allocation", 4000000, "")
        
    ##### sweeping example #####
#     sweep_experiments_configs = {"ddi": [
#                           ("gcn", "gcn", 510000, 4, 5, 550000),
# #                           ("sage", "gcn", 510000, 4, 5, 550000),
# #                           ("simple", "gcn", 510000, 4, 5, 550000),
# #                           ("adamic_ogb", "gcn", 510000, 4, 5, 550000),     
# #                           ("simplecos", "gcn",  510000, 4, 5, 550000),
# #                           ("gcn", "sage", 510000, 4, 5, 550000),
# #                           ("sage", "sage", 510000, 4, 5, 550000),
# #                           ("simple", "sage", 510000, 4, 5, 550000),
# #                           ("adamic_ogb", "sage", 510000, 4, 5, 550000),     
# #                           ("simplecos", "sage",  510000, 4, 5, 550000),                           
# #                           ("gcn", "simple", 550000, 4, 1, 510000),
# #                           ("sage", "simple", 550000, 4, 1, 510000),
# #                           ("simple", "simple", 550000, 4, 1, 510000),
# #                           ("adamic_ogb", "simple", 510000, 4, 1, 550000),
# #                           ("simplecos", "simple",  510000, 4, 1, 550000),                            
# #                           ("gcn", "simplecos", 510000, 4, 1, 550000),
# #                           ("sage", "simplecos", 510000, 4, 1, 550000),
# #                           ("simple", "simplecos", 510000, 4, 1, 550000),
# #                           ("adamic_ogb", "simplecos", 540000, 4, 1, 550000),     
# #                           ("simplecos", "simplecos",  510000, 4, 1, 550000),
# #                           ("gcn", "adamic_ogb", 510000, 4, 1, 550000),
# #                           ("sage", "adamic_ogb", 510000, 4, 1, 550000),
# #                           ("simple", "adamic_ogb", 510000, 4, 1, 550000),
# #                           ("adamic_ogb", "adamic_ogb", 540000, 4, 1, 550000),     
# #                           ("simplecos", "adamic_ogb",  510000, 4, 1, 550000),                           
#                        ],
#                                 }
#     submit_experiments(run_sweep_experiment, sweep_experiments_configs, run_local)
#     submit_experiments(run_sweep_valid_experiment, sweep_experiments_configs, run_local)