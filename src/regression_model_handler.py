from contact_model import binary_cross_entropy, cross_entropy_logits
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, precision_recall_curve, precision_score, matthews_corrcoef
from prettytable import PrettyTable
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import copy
import torch
import torch.nn as nn



class RegressionHandler(object):
    def __init__(self, model, optimizer, device, train_dataloader, val_dataloader, test_dataloader, inference_dataloader, dataset_name, **config):
        self.model = model
        self.best_rmse = float('inf')
        self.best_model = None
        self.best_epoch = None
        self.optimizer = optimizer
        self.device = device
        self.epochs = config["SETTINGS"]["MAX_EPOCH"]
        self.current_epoch = 0
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.val_dataloader = val_dataloader
        self.inference_dataloader = inference_dataloader
        self.batch_size = config["SETTINGS"]["BATCH_SIZE"]
        self.step = 0
        self.lr_decay = config["SETTINGS"]["LR_DECAY"]
        self.decay_interval = config["SETTINGS"]["DECAY_INTERVAL"]
        self.use_ld = config['SETTINGS']["USE_LD"]
        self.train_loss_epoch = []
        self.test_metrics = {}
        self.config = config
        self.output_dir = config["RESULT"]["OUTPUT_DIR"] + f'{dataset_name}/'
        test_metric_header = ["# Best Epoch", "RMSE"]
        train_metric_header = ["# Epoch", "Train_loss"]
        self.test_table = PrettyTable(test_metric_header)
        self.train_table = PrettyTable(train_metric_header)
        
        
    def train(self, include_test=True, include_val=True):
        float2str = lambda x: '%0.4f' % x        
        for i in range(self.epochs):
            self.current_epoch += 1
            if(self.use_ld):
                if(self.current_epoch % self.decay_interval == 0):
                    self.optimizer.param_groups[0]['lr'] *= self.lr_decay
            train_loss = self.train_epoch()
            train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss]))
            self.train_table.add_row(train_lst)
            self.train_loss_epoch.append(train_loss)
            if(include_val):
                rmse = self.validation()
                print(f"Validation RMSE: {rmse:.4f}")
                if(rmse < self.best_rmse):
                    self.best_model = copy.deepcopy(self.model)
                    self.best_rmse = rmse
                    self.save_temporary_model()
        if(include_val):
            self.model = copy.deepcopy(self.best_model)
        if(include_test):
            rmse = self.test()
            test_lst = ["Last epoch"] + list(map(float2str, [rmse]))
            self.test_table.add_row(test_lst)
            print(f'Regression test with averaged RMSE = {str(rmse)}.')     
            self.test_metrics["RMSE"] = rmse
            self.save_result()
            return self.test_metrics
        else:
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"reg_model_on_last_epoch_noTest.pth"))  
        return 0
              

    def train_epoch(self):
        self.model.train()
        criterion = nn.MSELoss()
        loss_epoch = 0
        num_batches = len(self.train_dataloader)
        for i, (canonical_smiles, sequence, drug_features, morgan_fp, aa_encodes, bert, y) in enumerate(tqdm(self.train_dataloader)):
            self.step += 1
            drug_features, morgan_fp, aa_encodes, bert, y = drug_features.to(self.device), morgan_fp.to(self.device), aa_encodes.to(self.device), bert.to(self.device), y.float().to(self.device)
            self.optimizer.zero_grad()
            embeddings, score = self.model(drug_features, morgan_fp, aa_encodes, bert)
            score = score.squeeze(-1)
            loss = criterion(score, y)
            loss.backward()
            self.optimizer.step()
            loss_epoch += loss.item()
        loss_epoch = loss_epoch / num_batches
        print(f'Training: Epoch {str(self.current_epoch)}/{self.epochs}, Loss = {str(loss_epoch)}')
        return loss_epoch              

              
    def test(self):
        criterion = nn.MSELoss()
        float2str = lambda x: '%0.4f' % x
        test_loss = 0
        y_label, y_pred = [], []
        data_loader = self.test_dataloader
        num_batches = len(data_loader)
        df = {'drug': [], 'protein': [], 'y_pred': [], 'y_label': []}
        with torch.no_grad():
            self.model.eval()
            for i, (canonical_smiles, sequence, drug_features, morgan_fp, aa_encodes, bert, y) in enumerate(tqdm(data_loader)):
                drug_features, morgan_fp, aa_encodes, bert, y = drug_features.to(self.device), morgan_fp.to(self.device), aa_encodes.to(self.device), bert.to(self.device), y.float().to(self.device)
                #WE CAN ALSO EXTRACT THE embeddings AS BELOW: but it is necessary to change the return variables in the forward function of the model
                #embeddings, score = self.model(drug_features, morgan_fp, aa_encodes, bert)
                score = self.model(drug_features, morgan_fp, aa_encodes, bert)
                score = score.squeeze(-1)
                mse = criterion(score, y)
                rmse_value = torch.sqrt(mse)
                test_loss += rmse_value
                y_label = y_label + y.to("cpu").tolist()
                y_pred = y_pred + score.to("cpu").tolist()
                df['drug'] = df['drug'] + list(canonical_smiles) #.to('cpu').tolist()
                df['protein'] = df['protein'] + list(sequence) #.to('cpu').tolist()
        df['y_label'] = y_label
        df['y_pred'] = y_pred
        data = pd.DataFrame(df)
        data.to_csv(f'{self.output_dir}regression_result.csv', index=False)
        test_loss = test_loss / num_batches
        return test_loss


    def validation(self):
        criterion = nn.MSELoss()
        float2str = lambda x: '%0.4f' % x
        total_rmse = 0
        data_loader = self.val_dataloader
        num_batches = len(data_loader)
        with torch.no_grad():
            self.model.eval()
            for i, (canonical_smiles, sequence, drug_features, morgan_fp, aa_encodes, bert, y) in enumerate(tqdm(data_loader)):
                drug_features, morgan_fp, aa_encodes, bert, y = drug_features.to(self.device), morgan_fp.to(self.device), aa_encodes.to(self.device), bert.to(self.device), y.float().to(self.device)
                #embeddings, score = self.model(drug_features, morgan_fp, aa_encodes, bert)
                score = self.model(drug_features, morgan_fp, aa_encodes, bert)
                score = score.squeeze(-1)
                mse = criterion(score, y)
                rmse_value = torch.sqrt(mse)
                total_rmse += rmse_value
        total_rmse = total_rmse / num_batches
        return total_rmse

    
    def inference(self):
        data_loader = self.inference_dataloader
        num_batches = len(data_loader)
        y_pred = []
        #If you wish to save the CTI embeddings uncomment the following commented lines: but it is necessary to change the return variables in the forward function of the model
        #all_embeddings = []
        df = {'drug': [], 'protein': [], 'y_pred': []}
        with torch.no_grad():
            self.model.eval()
            for i, (canonical_smiles, sequence, drug_features, morgan_fp, aa_encodes, bert) in enumerate(data_loader):
                drug_features, morgan_fp, aa_encodes, bert = drug_features.to(self.device), morgan_fp.to(self.device), aa_encodes.to(self.device), bert.to(self.device)
                #embeddings, score = self.model(drug_features, morgan_fp, aa_encodes, bert)
                score = self.model(drug_features, morgan_fp, aa_encodes, bert)
                #embeddings = embeddings.cpu().numpy()
                #all_embeddings.append(embeddings)
                score = score.squeeze(-1)
                y_pred = y_pred + score.to("cpu").tolist()
                df['drug'] = df['drug'] + list(canonical_smiles) #.to('cpu').tolist()
                df['protein'] = df['protein'] + list(sequence) #.to('cpu').tolist()
        #all_embeddings = np.vstack(all_embeddings)
        #np.save(f'{self.output_dir}embeddings.npy', all_embeddings)
        df['y_pred'] = y_pred
        data = pd.DataFrame(df)
        data.to_csv(f'{self.output_dir}inference_regression_result.csv', index=False)
        return True
        
              
              
    def save_result(self):
        if(self.config["RESULT"]["SAVE_MODEL"]):
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"reg_model_epoch_{self.current_epoch}.pth"))    
        state = {}
        state["config"] = self.config
        train_prettytable_file = os.path.join(self.output_dir, "reg_train_markdowntable.txt")
        with open(train_prettytable_file, "w") as fp:
            fp.write(self.train_table.get_string()) 
        state["train_epoch_loss"] = self.train_loss_epoch
        test_prettytable_file = os.path.join(self.output_dir, "reg_test_markdowntable.txt")
        with open(test_prettytable_file, 'w') as fp:
            fp.write(self.test_table.get_string())
        state["test_metrics"] = self.test_metrics
        torch.save(state, os.path.join(self.output_dir, f"reg_result_metrics.pt"))
        
                     
            
    def save_temporary_model(self): 
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"reg_model_latest_epoch.pth"))       