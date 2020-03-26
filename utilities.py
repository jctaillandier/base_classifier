from typing import List
import pandas as pd
import numpy as np
import os, argparse, torch
    

def check_dir_path(path_to_check: str) -> str:
    '''
        Checks if provided path is currently a at current level directory.
        If it is, it appends a number to the end and checks again 
        until no directory with such name exists

        :PARAMS
        path_to_check: str The path to location to check

        return: str New path with which os.mkdir can be called
    '''
    new_path = path_to_check
    if os.path.isdir(path_to_check):
        print("Experiment with name: \'{}\' already exists. Appending int to folder name. ".format(path_to_check))
        if os.path.isdir(path_to_check):
            expand = 1
            while True:
                expand += 1
                new_path = path_to_check[:-1] + '_' + str(expand) + '/'
                if os.path.isdir(new_path):
                    continue
                else:
                    break
            print(f"Experiment path: {new_path} \n \n ")
    return new_path

def parse_arguments(parser):
    parser.add_argument('-ep','--epochs', type=int, help='Number of epochs to train the model.', required=True)
    parser.add_argument('-bs','--batch_size', type=int, default=2048, help='batch size for Training loop. Test set will alwayas be the size of the test set (passed as one batch)', required=False)
    parser.add_argument('-tbs','--test_batch_size', type=int, default=4096, help='Size of test batch size. Do not touch. If fails for out of memory, need code adjustment', required=False)
    parser.add_argument('-wd','--weight_decay', type=float, default=0, help='Value for L2 penalty known as Weight Decay. Has not shown any value in this use case', required=False)
    parser.add_argument('-lr','--learning_rate', type=float, default=1e-4, help='Learning rate on which we will optimize with Adam.', required=False)
    parser.add_argument('-tg','--target', type=str, help='Name of the column you want to set as target for the model. Needs to be binary 1-0', required=True)
    parser.add_argument('-in','--input_dataset', type=str, default='Adult_NotNA_', help='Dataset to use as input. Currently support `gansan` and `disp_impact`', required=False, choices=['Adult_NotNA_', '0a_no1_e20', '25a_no1_e20' , '80a_no1_e20', '9875a_no1_e20','disp_impact_1'])

    # parser.add_argument('--discriminator_size', type=tuple, default=(256, 128, 1), help='The dimension size of the discriminator. (default value: (256, 128, 1))')
    args = parser.parse_args()
    exp_name = f"{args.input_dataset}_{args.target}_{args.epochs}ep_{args.batch_size}bs_{args.learning_rate}lr"
    path_to_exp = check_dir_path(f'./experiments/{exp_name}/')
    os.mkdir(path_to_exp)
    return args, path_to_exp