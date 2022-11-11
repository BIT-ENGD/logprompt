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
    return parser.parse_args()