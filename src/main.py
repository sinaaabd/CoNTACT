from params import get_cfg
from contact_model import CoNTACT
from contact_model import RegressionModel
from model_handler import Handler
from dti_dataloader import PreparedDataset
from torch.utils.data import DataLoader
from helpers import set_seed, mkdir, graph_collate_func
from time import time
import torch
import pandas as pd
import argparse
import os



cfg = get_cfg()
gpu_id = cfg.SETTINGS.GPU_ID 
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description="THE PROJECT TITLE")
parser.add_argument('--data', type=str, metavar='TASK', help='dataset', default='KiBA')
parser.add_argument('--load_model', type=str, metavar='TASK', help='Load an existing pre-trained binary model')
parser.add_argument('--val', type=str, metavar='TASK', help='validation set name') #'Validation'
parser.add_argument('--train', type=str, metavar='TASK', help='Training set name') #'Training'
parser.add_argument('--test', type=str, metavar='TASK', help='Test set name') #'Test'
parser.add_argument('--inference', type=str, metavar='TASK', help='Inference set name') #'Inference'
parser.add_argument('--mode', type=str, metavar='TASK', choices=["scratch", "fine-tune"], help='Fine-tune on a new binary (SpSh) dataset or train from scratch.', default='scratch')


args = parser.parse_args()

def main():
    torch.cuda.empty_cache()
    set_seed(cfg.SETTINGS.SEED)
    mkdir(f'{cfg.RESULT.OUTPUT_DIR}{args.data}')
    
    print(f"The available device: {device}")
    print(f"The request dataset: {args.data}")
    print(f"Hyperparameters: {dict(cfg)}", end="\n\n")

    data_dir = f'../datasets/{args.data}'    

    dataloader_params = {
        'batch_size'  : cfg.SETTINGS.BATCH_SIZE,
        'shuffle'     : True,
        'num_workers' : cfg.SETTINGS.NUM_WORKERS,
        'drop_last'   : True,
        'collate_fn': graph_collate_func
    }

    if((args.mode == "fine-tune") and (not args.load_model)):
        print("To fine-tune on a new SpSh dataset, you must load an existing pre-trained binary model using --load_model.")
        exit(0)
    
    validation_data_generator = None
    training_data_generator = None
    test_data_generator = None
    if(args.train):
        print('Loading Training set...')
        train_csv = os.path.join(data_dir, f'{args.train}.csv')
        train_df = pd.read_csv(train_csv)
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        train_prepared_dataset = PreparedDataset(train_df.index.values, train_df)
        training_data_generator = DataLoader(train_prepared_dataset, **dataloader_params)
    dataloader_params['shuffle'] = False
    dataloader_params['drop_last'] = False
    if(args.val):
        print('Loading Validation set...')
        validation_csv = os.path.join(data_dir, f'{args.val}.csv')
        validation_df = pd.read_csv(validation_csv)
        validation_prepared_dataset = PreparedDataset(validation_df.index.values, validation_df)
        validation_data_generator = DataLoader(validation_prepared_dataset, **dataloader_params)
    if(args.test or args.inference):
        if(args.inference):
            print('Loading Inference set...')
            test_csv = os.path.join(data_dir, f'{args.inference}.csv')
        if(args.test):
            print('Loading Test set...')
            test_csv = os.path.join(data_dir, f'{args.test}.csv')
        test_df = pd.read_csv(test_csv)
        if('bind' not in test_df.columns):
            test_df['bind'] = 1
        #test_df = test_df.iloc[:1500].reset_index(drop=True)
        #print(len(test_df))
        #test_df = test_df.rename(columns={'KiBA_score':'bind'})
        
        test_prepared_dataset = PreparedDataset(test_df.index.values, test_df)
        test_data_generator = DataLoader(test_prepared_dataset, **dataloader_params)

    model = CoNTACT(device=device, **cfg).to(device=device)
    if(args.load_model and (args.mode == "scratch")):
        print(f"The existing model '{args.load_model}' has been loaded!")
        model.load_state_dict(torch.load(f'{cfg.RESULT.OUTPUT_DIR}{args.data}/{args.load_model}'))
        model.to(device=device)

    if(args.mode == "fine-tune"):
        print(f"The existing model '{args.load_model}' has been loaded!")
        pretrained_model = CoNTACT(device=device, **cfg).to(device=device)
        pretrained_model.load_state_dict(torch.load(f'{cfg.RESULT.OUTPUT_DIR}{args.data}/{args.load_model}'))
        pretrained_model.to(device=device)
        print("Fine-tuning on a new binary dataset (SpSh).")
        model = RegressionModel(pretrained_model, device, **cfg).to(device=device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SETTINGS.LR, weight_decay=cfg.SETTINGS.WEIGHT_DECAY)
    torch.backends.cudnn.benchmark = True
    
    if(args.inference):
        for param in model.parameters():
            param.requires_grad = False
    handler = Handler(model, optimizer, device, training_data_generator, validation_data_generator, test_data_generator, args.data, **cfg)
    result = None
    if(args.train):
        include_validation = False
        include_test = False
        if(args.val):
            include_validation = True
        if(args.test):
            include_test = True
        result = handler.train(include_validation, include_test)
    elif(args.inference):
        result = handler.test(dataloader="inference")
    
    with open(os.path.join(cfg.RESULT.OUTPUT_DIR, f"{args.data}/model_architecture.txt"), "w") as wf:
        wf.write(str(model))
    with open(os.path.join(cfg.RESULT.OUTPUT_DIR, f"{args.data}/config.txt"), "w") as wf:
        wf.write(str(dict(cfg)))
    print(f"\nDirectory for saving result: {cfg.RESULT.OUTPUT_DIR}{args.data}")
    print(f'\nend...')
    return result

#Only run when the Python file executes as the main file not by calling from another python file
if(__name__ == '__main__'):
    s = time()
    result = main()
    e = time()
    print(f"Execution time: {round(e - s, 2)}s, ")