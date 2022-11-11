import  torch.utils.data  as Data

class data_item(object):
    def __init__(self,value,label):
        self.value=value
        self.label=label

  

class LogDataset(Data.Dataset):
    def __init__(self,data):
   
        self.data=data

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]
    




def load_dataset(good,bad,number):

    data=[data_item(item,0)  for item in good[:number] ]
    data.extend([data_item(item,1) for item in bad[:number] ])
    dataset=LogDataset(data)

    return dataset


def collate_fn(x):
    return x
    

def get_dataloader(args,dataset,batch_size,shuffle=True,num_workers=0):
    
    return Data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn)