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
        delim=args.delim)

    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.loader_num_workers,
        collate_fn=seq_collate,
        pin_memory=True)
    return dset, loader

def seq_collate(data):
    (
        obs_seq_list,
        pred_seq_list,
        obs_seq_rel_list,
        pred_seq_rel_list,
        loss_mask_list,
    ) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [
        [start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])
    ]
    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    #non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    out = [
        obs_traj,
        pred_traj,
        obs_traj_rel,
        pred_traj_rel,
        loss_mask,
        seq_start_end,
    ]

    return tuple(out)

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
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
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
        #non_linear_ped = []
        for path in all_files:
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                # curr_seq_data is a 20 length sequence
                curr_seq_data = np.concatenate(
                    frame_data[idx : idx + self.seq_len], axis=0
                )
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.seq_len))
                num_peds_considered = 0
                #_non_linear_ped = []

                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1

                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    #curr_ped_seq = curr_ped_seq
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    #_non_linear_ped.append(poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    #non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        #non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(seq_list[:, :, : self.obs_len]).type(
            torch.float
        )
        self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len :]).type(
            torch.float
        )
        self.obs_traj_rel = torch.from_numpy(seq_list_rel[:, :, : self.obs_len]).type(
            torch.float
        )
        self.pred_traj_rel = torch.from_numpy(seq_list_rel[:, :, self.obs_len :]).type(
            torch.float
        )
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        #self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
             (start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]



    
    def __len__(self):
        return self.num_seq
    
    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :],
            self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :],
            self.pred_traj_rel[start:end, :],
            #self.non_linear_ped[start:end],
            self.loss_mask[start:end, :],
        ]
        return out


# class TrajectoryDataset(Dataset):
#     """Creating Trajectory datasets from .txt files"""
#     def __init__(
#             self,
#             data_dir,
#             obs_len=8,
#             pred_len=12,
#             skip=1,
#             min_ped=1,
#             delim="\t"
#     ):
#         """
#         Args:
#         - data_dir: Directory containing dataset files in the format
#         <frame_id> <ped_id> <x> <y>
#         - obs_len: Number of time-steps in input trajectories
#         - pred_len: Number of time-steps in output trajectories
#         - skip: Number of frames to skip while making the dataset
#         - min_ped: Minimum number of pedestrians that should be in a seqeunce
#         - delim: Delimiter in the dataset files
#         """
#         super(TrajectoryDataset, self).__init__()
        
#         self.data_dir = data_dir
#         self.obs_len = obs_len
#         self.pred_len = pred_len
#         self.skip = skip
#         self.seq_len = self.obs_len + self.pred_len
#         self.delim = delim
#         self.dtype = torch.float32

#         all_files = os.listdir(self.data_dir)
#         all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
#         num_peds_in_seq = []
#         seq_list = []
#         seq_list_rel = []
#         loss_mask_list = []
#         i = 0 #sampath
#         for path in all_files:
#             data = read_file(path, self.delim) # Tensor (N, 4)
#             frames = torch.unique(data[:, 0]).tolist() #Total number of unique frames
#             frame_data = []
#             for frame in frames:
#                 frame_data.append(data[frame == data[:, 0], :])
#             num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))

#             for idx in range(0, num_sequences * self.skip + 1, skip):
#                 # obs_len = 8 ; pred_len = 12 #seq_len is a 20 length sequence(seq_len)
                
#                 # total number of pred in the 20 seq_len
#                 curr_seq_data = torch.cat(frame_data[idx : idx + self.seq_len], dim=0)
#                 # uniques pred in the 20 seq_len
#                 peds_in_curr_seq = torch.unique(curr_seq_data[:, 1]).tolist()
#                 curr_seq_rel = torch.zeros(len(peds_in_curr_seq), 2, self.seq_len, dtype=self.dtype)
#                 curr_seq = torch.zeros(len(peds_in_curr_seq), 2, self.seq_len, dtype=self.dtype)
#                 curr_loss_mask = torch.zeros(len(peds_in_curr_seq), self.seq_len, dtype=self.dtype)
#                 num_peds_considered = 0

#                 for _, ped_id in enumerate(peds_in_curr_seq):
#                     curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
#                     #torch.round_(curr_ped_seq, decimals=4)
#                     pad_front = frames.index(curr_ped_seq[0, 0]) - idx
#                     pad_end = frames.index(curr_ped_seq[-1, 0]) -idx + 1

#                     if pad_end - pad_front != self.seq_len:
#                         continue
#                     _idx = num_peds_considered
#                     curr_ped_seq = curr_ped_seq[:,2:].t() # (2,20)
#                     # relative co-ordinates of predistrain
#                     rel_curr_ped_seq = torch.zeros_like(curr_ped_seq)
#                     rel_curr_ped_seq[:,1:] = curr_ped_seq[:,1:] - curr_ped_seq[:,:-1]
#                     curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
#                     curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
#                     curr_loss_mask[_idx, pad_front:pad_end] = 1
#                     num_peds_considered += 1

#                 if num_peds_considered > min_ped:
#                     #non_linear_ped += _non_linear_ped
#                     num_peds_in_seq.append(num_peds_considered)
#                     loss_mask_list.append(curr_loss_mask[:num_peds_considered])
#                     seq_list.append(curr_seq[:num_peds_considered])
#                     seq_list_rel.append(curr_seq_rel[:num_peds_considered])
            
#             print(i)
#             i = i +1
#             #sampath

#         self.num_seq = len(seq_list)
#         seq_list = torch.cat(seq_list, dim=0)
#         seq_list_rel = torch.cat(seq_list_rel, dim=0)
#         loss_mask_list = torch.cat(loss_mask_list, dim=0)

#         seq_list_rel2 = torch.zeros_like(seq_list)
#         seq_list_rel2[:,:,1:] = seq_list[:,:,1:] - seq_list[:,:,:-1]
#         loss_mask_list2 = torch.ones(seq_list.shape[0], seq_list.shape[1])

#         relCheck = torch.eq(seq_list_rel,seq_list_rel2)
#         maskCheck = torch.eq(loss_mask_list, loss_mask_list2)
#         b = 10
#         a = 1