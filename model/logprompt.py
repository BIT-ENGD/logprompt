import torch 
import torch.nn as nn
from transformers import XLNetConfig, XLNetModel,XLNetLMHeadModel,XLNetPreTrainedModel,XLNetTokenizer



class LogPromt(XLNetPreTrainedModel):
    def __init__(self,config) -> None:
        super().__init__(config)
        self.config=config
        self.model=XLNetLMHeadModel(config)
        self.loss_fn=nn.CrossEntropyLoss(reduction='mean')


    def forward(self,input_ids,attention_mask,mask_pos):
        logits=self.model(input_ids,attention_mask)
        
        return None
       # return self.loss_fn(query,class_label)




