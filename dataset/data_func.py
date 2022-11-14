import  torch.utils.data  as Data
import torch

class data_item(object):
    def __init__(self,value,label):
        self.value=value
        self.label=label

  

class LogDataset(Data.Dataset):
    def __init__(self,args,data):
        self.data=list()
        tokenizer=args.tokenizer
        max_len=0
        all_data=list()
        for item in data:
            tp_value=""
    
            for tp in item.value:
                tp_value+=tp +tokenizer.sep_token

            if len(tp_value) >max_len:
                max_len= len(tp_value)
            all_data.append({"data":tp_value,"label":item.label})
        max_len+=len(args.template)
        for item in all_data:
            template=tokenizer(item["data"]+args.template,padding = 'max_length',max_length=max_len)
            newitem=data_item(template.data,item["label"])
            self.data.append(newitem)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]
    




def load_dataset(args,good,bad,number,template):

    data=[data_item(item,0)  for item in good[:number] ]
    data.extend([data_item(item,1) for item in bad[:number] ])
    dataset=LogDataset(args,data)

    return dataset


def collate_fn(x):
    input_ids=[ item.value["input_ids"] for item in x]
    attention_mask=[ item.value["attention_mask"] for item in x]
    labels=[ item.label for item in x]
    
    return {"ids":input_ids,"mask":attention_mask,"label":labels}
    

def get_dataloader(args,dataset,batch_size,shuffle=True,num_workers=0):
    
    return Data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn)