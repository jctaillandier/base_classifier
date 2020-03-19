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
