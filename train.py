import argparse
import logging
import torch
import random

from dataloader.loader import data_loader

from  utils import * 


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--log_dir", 
        default="./", 
        help="Directory containing logging file"
    )
    parser.add_argument(
        "--dataset_name", 
        default="zara2", 
        type=str
    )
    parser.add_argument(
        "--delim", 
        default="\t"
    )
    parser.add_argument(
        "--loader_num_workers", 
        default=4, 
        type=int
    )
    parser.add_argument(
        "--obs_len", 
        default=8, 
        type=int,
        help="Observation Length"    
    )
    parser.add_argument(
        "--pred_len", 
        default=12, 
        type=int,
        help="Prediction Length"
    )
    parser.add_argument(
        "--skip", 
        default=1, 
        type=int)
    parser.add_argument(
        "--seed", 
        type=int, 
        default=72, 
        help="Random seed."
    )
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_epochs", default=400, type=int)
    args = parser.parse_args()

    train_path = get_dset_path(args.dataset_name, "train")
    val_path = get_dset_path(args.dataset_name, "test")

    logging.info("Creating training dataset")
    train_dset = data_loader(args, train_path)
    logging.info("Creating tesing dataset")
    train_dset = data_loader(args, train_path)
    
    train(args)

def train(args):
    pass

if __name__ == '__main__':
    main()