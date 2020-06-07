# Code for _Class-Weighted Classification: Trade-offs and Robust Approaches_

This repo provides supporting Python code for the paper
> Xu, Z., Dan, C., Khim, J., Ravikumar, P. (2020). [Class-Weighted Classification: Trade-offs and Robust Approaches](https://arxiv.org/abs/2005.12914). arXiv preprint arXiv:2005.12914.

## Setup

Requires [conda](https://docs.conda.io/en/latest/) with Python 3.7.


1. Install conda dependencies in the environment: `conda env create -f environment.yml`
2. Run `download_uci_data.sh` from the repo main directory to download the Covertype dataset.
3. Activate the conda enviroment with `conda activate robust_weighting`
4. Setup up a [wandb](https://docs.wandb.com/quickstart) account and create a project named `extreme-classification` (or rename the `project` argument inside the `wandb.init` call inside `src/main.py` )

## Scripts

Navigate to the root of the repo.

Run `./power_exp.sh` for the synthetic experiment results.

Run `./uci_exp.sh` for the real world dataset (Covertype) results.

View results in the [wandb website](https://app.wandb.ai).
