import  torch.utils.data  as Data

class data_item(object):
    def __init__(self,value,label):
        self.value=value
        self.label=label

  

class LogDataset(Data.Dataset):
    def __init__(self,args,data):
        self.data=list()
        tokenizer=args.tokenizer
        for item in data:
            tp_value=""
            for tp in item.value:
                tp_value+=tp +tokenizer.sep_token
            newitem=data_item(tp_value,item.label)
            self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]
    




def load_dataset(args,good,bad,number):

    data=[data_item(item,0)  for item in good[:number] ]
    data.extend([data_item(item,1) for item in bad[:number] ])
    dataset=LogDataset(args,data)

    return dataset


def collate_fn(x):
    return x
    

def get_dataloader(args,dataset,batch_size,shuffle=True,num_workers=0):
    
    return Data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn)