import argparse



def get_arg(): 
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_path",type=str,default="/data/linux/transformers/xlnet-large-cased")
    parser.add_argument("--log_dir",type=str,default="logs")
    parser.add_argument("--appname",type=str,default="logprompt")
    parser.add_argument("--dataset",type=str,default="BLG")
    parser.add_argument("--seed",type=int,default=144)
    parser.add_argument("--device",type=str,default="cuda")
    parser.add_argument("--shot",type=int,default=10)
    parser.add_argument("--epoch",type=int,default=10)
    parser.add_argument("--learning_rate", default=1e-5, type=float)  # 5e-5, bert 1e-5, t5 1e-4
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=5.0, type=float)
    
    parser.add_argument("--max_steps", default=10000, type=int)
    parser.add_argument("--warmup_steps", default=0.1, type=float)
    return parser.parse_args()