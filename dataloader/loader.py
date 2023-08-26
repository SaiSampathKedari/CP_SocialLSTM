import logging
import os
import math
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

def data_loader(args, path):
    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim
    )
    return dset

def read_file(_path, delim="\t"):
    data = []
    if delim == "tab":
        delim = "\t"
    elif delim == "space":
        delim = " "
    with open(_path, "r") as f:
        for line in f:
            line = line.strip().split(delim)
            line = torch.tensor([float(i) for i in line])
            data.append(line)
    return torch.stack(data)



class TrajectoryDataset(Dataset):
    """Creating Trajectory datasets from .txt files"""
    def __init__(
            self,
            data_dir,
            obs_len=8,
            pred_len=12,
            skip=1,
            min_ped=1,
            delim="\t"
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()
        
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []

        for path in all_files:
            data = read_file(path, self.delim) # Tensor (N, 4)
            frames = torch.unique(data[:, 0]).tolist() #Total number of unique frames
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                # obs_len = 8 ; pred_len = 12 #seq_len is a 20 length sequence
                curr_seq_data = torch.cat(frame_data[idx : idx + self.seq_len], axis=0)
                peds_in_curr_seq = torch.unique(curr_seq_data[:, 1]).tolist()
                curr_seq_rel = torch.zeros(len(peds_in_curr_seq), 2, self.seq_len)
                curr_seq = torch.zeros(len(peds_in_curr_seq), 2, self.seq_len)
                curr_loss_mask = torch.zeros(len(peds_in_curr_seq), self.seq_len)















    
    def __len__(self):
        pass
    
    def __getitem__(self, index):
        pass