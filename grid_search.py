from typing import List
import argparse, os
from joblib import Parallel, delayed

def parse_arguments(parser):
    parser.add_argument('-in','--input_dataset', type=str, default='Adult_NotNA_', help='Dataset to use as input. Currently support `gansan` and `disp_impact`', required=False, choices=['gansan', 'disp_impact'])
    parser.add_argument('-cpu','--cpu_parallel', type=int, default=1, help='How many cpu to run in parallel to accelerate trainings', required=False)
    parser.add_argument('-ep','--epochs', type=int, help='Number of epochs to train the model.', required=True)
    parser.add_argument('-tg','--target', type=str, default='sex', help='Column to use as target', required=False)
    args = parser.parse_args()
    return args

def launch(bs: int, lr:float, ep:int, dataset:str, target:str):
    os.system(f"python3 main.py -ep={ep} -tg={target} -in={dataset} -lr={lr} -bs={bs}")

parser = argparse.ArgumentParser()
args = parse_arguments(parser)

lrs = [1e-2, 1e-3,1e-4,1e-5,1e-6]
bses = [256, 1024]
input_dataset = args.input_dataset

Parallel(n_jobs=args.cpu_parallel)(delayed(launch)(bs, lr, args.epochs, args.input_dataset, args.target) for lr in lrs for bs in bses)

