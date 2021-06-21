# Edge Proposal Sets for Link Prediction

This directory contains code for the paper. Edge Proposal Sets for Link Prediction.

## Software Requirements

This codebase requires Python 3, PyTorch 1.6+, OGB 1.3.1, torch_geometric 1.7.0. In principle, this code can be run on CPU but we assume GPU utilization throughout the codebase.

## Usage

Experiments are run in the following way:

Edit the `submit_job.py` file to contain the desired experiments. Currently, it reproduces our best results on ddi (76.32 ± 2.54) and collab (65.48 ± 0.00). At the end of the file, there are a few commented samples provided for sweeping across different sizes of proposal sets. The keys of the dictionary are the datasets, and the list of tuples are the experiments. For example, ("gcn", "simple", 150000, 6, 1, 210000) represents a filtering model of GCN, a ranking model of Common Neighbors, and a sweep from 150,000 to 210,000 edges added at intervals of 10,000, and running this only 1 trial (since Common Neighbors is deterministic). 

You can then run `python submit_job.py`. The results will be saved in `logs/`. Base model training results are logged in `<dataset>_<model>_generation.o%local<time>`. Final results after applying Filter & Rank is in `<dataset>_<filter_model>_<rank_model>_530000.o%local<time>`. 

Other files: `models.py` contains model implementations, `visualize.ipynb` generates figures and the LaTeX tables (and only looks at the sweep range as specified in the paper, which can be set by the `radius=` argument), `filter.py` generates scores for the broad starting set using a given filtering model, `train_and_eval.py` contains all training and evaluation functions, `rank.py` is a standard link prediction setup inspired by OGB, and `submit_job.py` runs the entire pipeline together.