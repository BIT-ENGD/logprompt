import torch
from dataset import *
from model import *
from util import *
from tqdm import tqdm
import os
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

def transtodevice(args,batch):
    data=dict()
    data["ids"]=torch.tensor(batch["ids"]).to(args.device)
    data["mask"]=torch.tensor(batch["mask"]).to(args.device)
    data["label"]=torch.tensor(batch["label"]).to(args.device)
    return data

def inference(args,dataloader):
    config=args.config
    model=args.model.to(args.device)
    #pbar = tqdm(dataloader,desc="fewshow-log")
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps*args.warmup_steps, num_training_steps=args.max_steps)

    for epoch in range(args.epoch):
        for step,batch in enumerate(dataloader):
            batch=transtodevice(args,batch)
            loss=model(batch["ids"],batch["mask"],batch["label"])
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
    template="It was <mask>."
    args.template=template
    dataset=ds_fn(args,good,bad,int(args.shot/2),template)
    loader=get_loader(args,dataset,batch_size)
 

    inference(args,loader)



if __name__ == "__main__":

    main()
    