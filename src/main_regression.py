from params import get_cfg
from contact_model import CoNTACT
from contact_model import RegressionModel
from regression_model_handler import RegressionHandler
from dti_dataloader import PreparedDataset, PreparedInferenceDataset
from torch.utils.data import DataLoader
from helpers import set_seed, mkdir, graph_collate_func, graph_collate_inference_func
from time import time
import torch
import pandas as pd
import argparse
import os



cfg = get_cfg()
gpu_id = cfg.SETTINGS.GPU_ID 
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description="THE PROJECT TITLE")
parser.add_argument('--data', type=str, metavar='TASK', help='dataset', default='LA_DUDE')
parser.add_argument('--load_model', type=str, metavar='TASK', help='Load an existing regression model')
parser.add_argument('--load_bin_model', type=str, metavar='TASK', help='Load an existing binary model', default='binary_model_on_last_epoch.pth')
parser.add_argument('--train', type=str, metavar='TASK', help='Training set name') #'Training'
parser.add_argument('--test', type=str, metavar='TASK', help='Test set name') #'Test'
parser.add_argument('--val', type=str, metavar='TASK', help='Validation set name') #'Test'
parser.add_argument('--inference', type=str, metavar='TASK', help='Inference set name') #'Test'
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
    training_data_generator = None
    test_data_generator = None
    inference_data_generator = None
    val_data_generator = None
    if(args.train):
        print('Loading Training set...')
        train_csv = os.path.join(data_dir, f'{args.train}.csv')
        train_df = pd.read_csv(train_csv)
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        train_prepared_dataset = PreparedDataset(train_df.index.values, train_df)
        training_data_generator = DataLoader(train_prepared_dataset, **dataloader_params)
    dataloader_params['shuffle'] = False
    dataloader_params['drop_last'] = False
        
    if(args.inference):
        dataloader_params['collate_fn'] = graph_collate_inference_func
        print('Loading Inference set...')
        inference_csv = os.path.join(data_dir, f'{args.inference}.csv')
        inference_df = pd.read_csv(inference_csv)
        inference_prepared_dataset = PreparedInferenceDataset(inference_df.index.values, inference_df)
        inference_data_generator = DataLoader(inference_prepared_dataset, **dataloader_params)
    if(args.test):
        print('Loading Test set...')
        test_csv = os.path.join(data_dir, f'{args.test}.csv')
        test_df = pd.read_csv(test_csv)
        test_prepared_dataset = PreparedDataset(test_df.index.values, test_df)
        test_data_generator = DataLoader(test_prepared_dataset, **dataloader_params)
    if(args.val):
        print('Loading Validation set...')
        val_csv = os.path.join(data_dir, f'{args.val}.csv')
        val_df = pd.read_csv(val_csv)
        val_prepared_dataset = PreparedDataset(val_df.index.values, val_df)
        val_data_generator = DataLoader(val_prepared_dataset, **dataloader_params)
    print("Loading existing pre-trained binary model")
    pretrained_model = CoNTACT(device=device, **cfg).to(device=device)
    pretrained_model.load_state_dict(torch.load(f'{cfg.RESULT.OUTPUT_DIR}{args.data}/{args.load_bin_model}'))
    reg_model = RegressionModel(pretrained_model, device, **cfg).to(device=device)
    if(args.load_model):
        print("Loading existing pre-trained regression-based model")
        reg_model.load_state_dict(torch.load(f'{cfg.RESULT.OUTPUT_DIR}{args.data}/{args.load_model}'))
        reg_model.to(device=device)    
        
    optimizer = torch.optim.Adam(reg_model.parameters(), lr=cfg.SETTINGS.LR, weight_decay=cfg.SETTINGS.WEIGHT_DECAY)
    torch.backends.cudnn.benchmark = True
        
    handler = RegressionHandler(reg_model, optimizer, device, training_data_generator, val_data_generator, test_data_generator, inference_data_generator, args.data, **cfg)
    result = None
    if(args.train):
        include_test = False
        include_val = False
        if(args.test):
            include_test = True
        if(args.val):
            include_val = True
        result = handler.train(include_test, include_val)
    elif(args.inference):
        result = handler.inference()
    
    with open(os.path.join(cfg.RESULT.OUTPUT_DIR, f"{args.data}/reg_model_architecture.txt"), "w") as wf:
        wf.write(str(reg_model))
    with open(os.path.join(cfg.RESULT.OUTPUT_DIR, f"{args.data}/reg_config.txt"), "w") as wf:
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