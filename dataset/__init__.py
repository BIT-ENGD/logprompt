from dataset.data_func import *

DATA_DIR="dataset"

DATASET={"BLG":{"dir":"BLG","positive":"good_5000.pkl","negative":"bad_5000.pkl","ds_fn":load_dataset,"loader":get_dataloader,"batch_size":5}}