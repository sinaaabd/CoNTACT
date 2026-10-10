from contact_model import binary_cross_entropy, cross_entropy_logits
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, precision_recall_curve, precision_score, matthews_corrcoef
from prettytable import PrettyTable
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import copy
import torch



class Handler(object):
    def __init__(self, model, optimizer, device, train_dataloader, val_dataloader, test_dataloader, dataset_name, **config):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.epochs = config["SETTINGS"]["MAX_EPOCH"]
        self.current_epoch = 0
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.n_class = config["SETTINGS"]["NUM_CLASS"]
        self.batch_size = config["SETTINGS"]["BATCH_SIZE"]
        self.step = 0
        self.lr_decay = config["SETTINGS"]["LR_DECAY"]
        self.decay_interval = config["SETTINGS"]["DECAY_INTERVAL"]
        self.use_ld = config['SETTINGS']["USE_LD"]
        self.best_model = None
        self.best_epoch = None
        self.best_auroc = 0
        self.best_auprc = 0
        self.best_mcc = -1.0
        self.train_loss_epoch = []
        self.train_model_loss_epoch = []
        self.val_loss_epoch, self.val_auroc_epoch = [], []
        self.test_metrics = {}
        self.config = config
        self.output_dir = config["RESULT"]["OUTPUT_DIR"] + f'{dataset_name}/'
        valid_metric_header = ["# Epoch", "AUROC", "AUPRC", "MCC", "Val_loss"]
        test_metric_header = ["# Best Epoch", "AUROC", "AUPRC", "F1", "Sensitivity", "Specificity", "Accuracy", "MCC",
                              "Threshold", "Test_loss"]
        train_metric_header = ["# Epoch", "Train_loss"]
        self.val_table = PrettyTable(valid_metric_header)
        self.test_table = PrettyTable(test_metric_header)
        self.train_table = PrettyTable(train_metric_header)
        
        
    def train(self, include_validation=True, include_test=True):
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
            if(include_validation):
                mcc, auroc, auprc, val_loss = self.test(dataloader="val")               
                val_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [auroc, auprc, mcc, val_loss]))
                self.val_table.add_row(val_lst)
                self.val_loss_epoch.append(val_loss)
                self.val_auroc_epoch.append(auroc)
                if(mcc >= self.best_mcc):
                    self.best_model = copy.deepcopy(self.model)
                    self.best_mcc = mcc
                    self.best_epoch = self.current_epoch
                    self.save_temporary_model()
                print(f'Epoch {str(self.current_epoch)}: validation loss = {str(val_loss)}, AUROC = {str(auroc)}, AUPRC = {str(auprc)}, MCC = {mcc}')
        if(include_test):
            auroc, auprc, f1, sensitivity, specificity, accuracy, test_loss, thred_optim, precision, mcc = self.test(dataloader="test", val_included=include_validation)
            test_lst = ["epoch " + str(self.best_epoch)] + list(map(float2str, [auroc, auprc, f1, sensitivity, specificity, accuracy, mcc, thred_optim, test_loss]))
            self.test_table.add_row(test_lst)
            print(f'Test on the best model (epoch {str(self.best_epoch)}): Test Loss = {str(test_loss)}, AUROC = {str(auroc)}, AUPR = {str(auprc)}, Sensitivity = {str(sensitivity)}, Specificity = {str(specificity)}, Accuracy = {str(accuracy)}, MCC = {str(mcc)}.')     
            self.test_metrics["auroc"] = auroc
            self.test_metrics["auprc"] = auprc
            self.test_metrics["test_loss"] = test_loss
            self.test_metrics["sensitivity"] = sensitivity
            self.test_metrics["specificity"] = specificity
            self.test_metrics["accuracy"] = accuracy
            self.test_metrics["thred_optim"] = thred_optim
            self.test_metrics["best_epoch"] = self.best_epoch
            self.test_metrics["F1"] = f1
            self.test_metrics["Precision"] = precision
            self.test_metrics["MCC"] = mcc
            self.save_result(include_validation)
            return self.test_metrics
        else:
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"model_on_last_epoch_noTest.pth"))  
        return 0
              

    def train_epoch(self):
        self.model.train()
        loss_epoch = 0
        num_batches = len(self.train_dataloader)
        for i, (canonical_smiles, sequence, drug_features, morgan_fp, aa_encodes, bert, y) in enumerate(tqdm(self.train_dataloader)):
            self.step += 1
            drug_features, morgan_fp, aa_encodes, bert, y = drug_features.to(self.device), morgan_fp.to(self.device), aa_encodes.to(self.device), bert.to(self.device), y.float().to(self.device)
            self.optimizer.zero_grad()
            score = self.model(drug_features, morgan_fp, aa_encodes, bert)
            if(self.n_class == 1):
                n, loss = binary_cross_entropy(score, y)
            else:
                n, loss = cross_entropy_logits(score, y)
            loss.backward()
            self.optimizer.step()
            loss_epoch += loss.item()
        loss_epoch = loss_epoch / num_batches
        print(f'Training: Epoch {str(self.current_epoch)}/{self.epochs}, Loss = {str(loss_epoch)}')
        return loss_epoch              

              
    def test(self, dataloader="test", val_included=False):
        float2str = lambda x: '%0.4f' % x
        test_loss = 0
        y_label, y_pred = [], []
        if(dataloader == "test" or dataloader == "inference"):
            data_loader = self.test_dataloader
        elif(dataloader == "val"):
            data_loader = self.val_dataloader
        else:
            raise ValueError(f"Error key value {dataloader}")
        num_batches = len(data_loader)
        df = {'drug': [], 'protein': [], 'y_pred': [], 'y_label': []}
        with torch.no_grad():
            self.model.eval()
            for i, (canonical_smiles, sequence, drug_features, morgan_fp, aa_encodes, bert, y) in enumerate(data_loader):
                drug_features, morgan_fp, aa_encodes, bert, y = drug_features.to(self.device), morgan_fp.to(self.device), aa_encodes.to(self.device), bert.to(self.device), y.float().to(self.device)
                if(dataloader == "val"):
                    score = self.model(drug_features, morgan_fp, aa_encodes, bert)
                elif(dataloader == "test"):
                    if(val_included):
                        score = self.best_model(drug_features, morgan_fp, aa_encodes, bert)
                    else:
                        score = self.model(drug_features, morgan_fp, aa_encodes, bert)
                elif(dataloader == "inference"):
                    score = self.model(drug_features, morgan_fp, aa_encodes, bert)
                if(self.n_class == 1):
                    n, loss = binary_cross_entropy(score, y)
                else:
                    n, loss = cross_entropy_logits(score, y)
                test_loss += loss.item()
                y_label = y_label + y.to("cpu").tolist()
                y_pred = y_pred + n.to("cpu").tolist()
                #if(dataloader == "test"):
                df['drug'] = df['drug'] + list(canonical_smiles) #.to('cpu').tolist()
                df['protein'] = df['protein'] + list(sequence) #.to('cpu').tolist()
        auroc = roc_auc_score(y_label, y_pred)
        auprc = average_precision_score(y_label, y_pred)
        df['y_label'] = y_label
        df['y_pred'] = y_pred
        data = pd.DataFrame(df)
        y_true = data['y_label'].values
        y_pred_proba = data['y_pred'].values
        y_pred1 = (y_pred_proba >= 0.5).astype(int)  
        mcc = matthews_corrcoef(y_true, y_pred1)        
        test_loss = test_loss / num_batches
        if(dataloader == "test" or dataloader == "inference"):
            fpr, tpr, thresholds = roc_curve(y_label, y_pred)
            prec, recall, _ = precision_recall_curve(y_label, y_pred)
            try:
                precision = tpr / (tpr + fpr)
            except RuntimeError:
                raise ('RuntimeError: the divide==0')
            f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
            thred_optim = thresholds[5:][np.argmax(f1[5:])]
            y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]
            cm1 = confusion_matrix(y_label, y_pred_s)
            accuracy = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
            sensitivity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
            specificity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
            precision1 = precision_score(y_label, y_pred_s)
            data.to_csv(f'{self.output_dir}visualization.csv', index=False)

            if(dataloader == "inference"):
                inf_metric_header = ["AUROC", "AUPRC", "F1", "Sensitivity", "Specificity", "Accuracy", "MCC"]
                self.test_table = PrettyTable(inf_metric_header)
                inf_lst = list(map(float2str, [auroc, auprc, np.max(f1[5:]), sensitivity, specificity, accuracy, mcc]))
                self.test_table.add_row(inf_lst)
                test_prettytable_file = os.path.join(self.output_dir, "inference_markdowntable.txt")
                with open(test_prettytable_file, 'w') as fp:
                    fp.write(self.test_table.get_string())
                    
            return auroc, auprc, np.max(f1[5:]), sensitivity, specificity, accuracy, test_loss, thred_optim, precision1, mcc
        else:
            return mcc, auroc, auprc, test_loss
              
              
              
    def save_result(self, include_validation=True):
        if(self.config["RESULT"]["SAVE_MODEL"]):
            if(include_validation):
                torch.save(self.best_model.state_dict(),
                           os.path.join(self.output_dir, f"best_model_epoch_{self.best_epoch}.pth"))
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"model_epoch_{self.current_epoch}.pth"))    
        state = {}
        state["config"] = self.config
        train_prettytable_file = os.path.join(self.output_dir, "train_markdowntable.txt")
        with open(train_prettytable_file, "w") as fp:
            fp.write(self.train_table.get_string()) 
        state["train_epoch_loss"] = self.train_loss_epoch
        if(include_validation):
            val_prettytable_file = os.path.join(self.output_dir, "validation_markdowntable.txt")
            with open(val_prettytable_file, 'w') as fp:
                fp.write(self.val_table.get_string())
            state["val_epoch_loss"] = self.val_loss_epoch
        test_prettytable_file = os.path.join(self.output_dir, "test_markdowntable.txt")
        with open(test_prettytable_file, 'w') as fp:
            fp.write(self.test_table.get_string())
        state["test_metrics"] = self.test_metrics
        torch.save(state, os.path.join(self.output_dir, f"result_metrics.pt"))
        
                     
            
    def save_temporary_model(self): 
        torch.save(self.best_model.state_dict(), os.path.join(self.output_dir, f"model_latest_epoch.pth"))       