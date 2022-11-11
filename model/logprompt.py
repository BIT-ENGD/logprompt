import torch 
from transformers import XLNetConfig, XLNetModel,XLNetLMHeadModel,XLNetPreTrainedModel,XLNetTokenizer



class LogPromt(XLNetPreTrainedModel):
    def __init__(self,config) -> None:
        super().__init__(config)
        self.config=config
        self.model=XLNetPreTrainedModel(config)


    def forward(self):
        pass 


