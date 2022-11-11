import pickle as pk


def cache_data(data,filename):
    with open(filename,"wb") as f:
        pk.dump(data,f)

def load_cache(filename):
    with open(filename,"rb") as f:
        return pk.load(f)