import os
import numpy as np
import torch
from dataloader.loader import read_file
from matplotlib import pyplot as plt 
import matplotlib.animation as animation



def get_dset_path(dset_name, dset_type):
    _dir = os.path.dirname(__file__)
    
    return os.path.join(_dir, "datasets", dset_name, dset_type)


class Anim():
    def __init__(self, _path, delim = "\t"):
        data = read_file(_path, delim)
        frames = np.unique(data[:, 0]).tolist()
        frame_data = []
        for frame in frames:
            frame_data.append(data[frame == data[:, 0], :])
        
        self.colormap = ['Reds','Greens','Blues','pink']
        ped_list = np.unique(data[:,1]).tolist()
        ped_colornum = (np.arange(len(ped_list))) % (len(self.colormap))
        self.ped_numbeing = dict(zip(ped_list, ped_colornum))
        

        self.xmin = torch.min(data[:,2]).item()
        self.xmax = torch.max(data[:,2]).item()
        self.ymin = torch.min(data[:,3]).item()
        self.ymax = torch.max(data[:,3]).item()


        self.anim = animation.FuncAnimation(plt.gcf(), self.animate, 
            fargs=(frame_data, 5), interval = 100, frames=690, repeat=False)
        plt.tight_layout()
        plt.axis('off')
        #plt.show()
        # setting up writeres object
        writergif = animation.PillowWriter(fps=20)
        self.anim.save("animation.gif", writer= writergif)
        plt.close()

    def animate(self, frame_id, frame_data, seq_len):
        
        # clearing the plot and setting limits to the axis
        print(frame_id)
        plt.clf()
        plt.xlim([self.xmin, self.xmax])
        plt.ylim([self.ymin, self.ymax])
        plt.axis('off')

        peds_in_curr_frame = np.unique(frame_data[frame_id][:,1])
        num_peds = len(peds_in_curr_frame)
        frame_id_front = 0
        frame_id_end = frame_id
        if frame_id < (seq_len -1):
            frame_id_front = 0
        else:
            frame_id_front = frame_id - seq_len + 1
        
        curr_seq_data = np.concatenate(
            frame_data[frame_id_front:frame_id_end+1],
            axis=0)
        for _, ped_id in enumerate(peds_in_curr_frame):
            curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
            xdata = curr_ped_seq[:,2]
            ydata = curr_ped_seq[:,3]
            colors = np.arange(curr_ped_seq.shape[0]) + 5
            plt.scatter(xdata, ydata, c = colors, cmap=self.colormap[self.ped_numbeing[ped_id]])
        
        return plt
    
    def init(self):
        plt.clf()
        plt.xlim([self.xmin, self.xmax])
        plt.ylim([self.ymin, self.ymax])
        plt.axis('off')
        return plt