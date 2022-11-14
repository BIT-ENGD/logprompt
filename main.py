import torch
from dataset import *
from model import *
from util import *
from tqdm import tqdm
import os

def inference(args,dataloader):
    config=args.config
    #pbar = tqdm(dataloader,desc="fewshow-log")
    for epoch in range(args.epoch):
        for step,batch in enumerate(dataloader):
            print("========epoch: %d, step: %d =batch==============="%(epoch,step))



def main():


    args=get_arg()
    dsinfo=DATASET[args.dataset]
    ds_fn=dsinfo["ds_fn"]
    get_loader=dsinfo["loader"]
    batch_size=dsinfo["batch_size"]
    negative=DATA_DIR+os.sep+dsinfo["dir"]+os.sep+dsinfo["negative"]
    positive=DATA_DIR+os.sep+dsinfo["dir"]+os.sep+dsinfo["positive"]
    #     data=load_cache(path)

    args.config=XLNetConfig.from_pretrained(args.model_path)

    args.tokenizer=XLNetTokenizer.from_pretrained(args.model_path)
    args.model=LogPromt.from_pretrained(args.model_path,config=args.config).to(args.device)
    
    
    good=load_cache(positive)
    bad=load_cache(negative)
    dataset=ds_fn(args,good,bad,int(args.shot/2))
    loader=get_loader(args,dataset,batch_size)
 
 
    inference(args,loader)



if __name__ == "__main__":

    main()
    