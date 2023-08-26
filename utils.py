import os

def get_dset_path(dset_name, dset_type):
    _dir = os.path.dirname(__file__)
    
    return os.path.join(_dir, "datasets", dset_name, dset_type)