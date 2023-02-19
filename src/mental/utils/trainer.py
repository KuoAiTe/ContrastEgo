import numpy as np
import torch
import copy
import torch.nn as nn
import time
from datetime import datetime
from torch.utils.data import DataLoader
from mental.utils.dataset import MedDataset
from mental.utils.utilities import to_device, compute_metrics_from_results
from mental.utils.dataclass import TrainingArguments

from torch_geometric.nn import MLP


class Trainer:
    def __init__(self, train_data, validation_data, test_data, logger, args: TrainingArguments):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.train_dataset = MedDataset(train_data)
        if validation_data is not None:
            self.validation_dataset = MedDataset(validation_data)
        self.test_dataset = MedDataset(test_data)
        self.logger = logger
        self.args = args

    def train(self, model):
        executed_at = int(datetime.timestamp(datetime.now()))

        train_model = model.to(self.device)
        train_optimizer = torch.optim.AdamW(train_model.parameters(), lr = self.args.learning_rate, weight_decay = self.args.weight_decay)
    
        train_dataloader = self.get_train_dataloader()
        best_validation_result = None
        result_with_best_val = None
        avg_run_time = []
        for epoch in range(self.args.num_train_epochs):
            loss = 0
            start_time = time.time_ns()
            for step, inputs in enumerate(train_dataloader):
                loss += self.training_step(train_model, train_optimizer, inputs)
            run_time = (time.time_ns() - start_time) / 1e9
            avg_run_time.append(run_time)
            print(f'took {run_time} sec, avg: {np.mean(avg_run_time)} sec')
            validation_result = self.validate(train_model)
            if best_validation_result == None or validation_result.auc_roc_macro > best_validation_result.auc_roc_macro:
                best_validation_result = validation_result
                #y_true, y_pred = self.predict(train_model)
                result_with_best_val = best_validation_result
                print(result_with_best_val)
            print(f'{self.logger.baseline} {self.logger.round} Epoch: #{epoch} -> loss: {loss:.4f}')
            print(f' VAL_CURR: {validation_result.f1_depressed:.3f}  VAL_CURR_AUC: {validation_result.auc_roc_macro:.3f}')
            print(f' VAL_BEST: {best_validation_result.f1_depressed:.3f}  VAL_BEST_AUC: {best_validation_result.auc_roc_macro:.3f}')
            print(f'TEST_BEST: {result_with_best_val.f1_depressed:.3f} TEST_BEST_AUC: {result_with_best_val.auc_roc_macro:.3f}')
            print(validation_result)
        self.logger.log(executed_at, epoch, result_with_best_val)
        
        return model

    def training_step(self, model, optimizer, inputs):
        model.train()
        inputs = to_device(inputs, self.device)
        optimizer.zero_grad()
        _, loss = model.compute_loss(inputs)
        loss.backward()
        optimizer.step()
        return loss.item()

    def validate(self, model):
        validation_dataloader = self.get_validation_dataloader()
        y_true = []
        y_pred = []
        for step, inputs in enumerate(validation_dataloader):
            labels, predictions = self.prediction_step(model, inputs)
            labels = labels.detach().cpu().numpy()
            predictions = predictions.detach().cpu().numpy()
            y_true.extend(labels)
            y_pred.extend(predictions)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        result = compute_metrics_from_results(y_true, y_pred)
        return result

    def predict(self, model):
        test_dataloader = self.get_test_dataloader()
        y_true = []
        y_pred = []
        for step, inputs in enumerate(test_dataloader):
            labels, predictions = self.prediction_step(model, inputs)
            labels = labels.detach().cpu().numpy()
            predictions = predictions.detach().cpu().numpy()
            y_true.extend(labels)
            y_pred.extend(predictions)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        return y_true, y_pred

    def validation_step(self, model, inputs):
        model.eval()
        inputs = to_device(inputs, self.device)
        _, validation_loss = model.compute_loss(inputs)
        return validation_loss.item()

    def prediction_step(self, model, inputs):
        model.eval()
        inputs = to_device(inputs, self.device)
        labels, predictions = model.predict(inputs)
        return labels, predictions

    def get_validation_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset = self.validation_dataset,
            batch_size = self.args.train_batch_size,
            shuffle = True,
            num_workers = 0,
            collate_fn = MedDataset.collate_fn
        )
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        return DataLoader(
            dataset = self.train_dataset,
            batch_size = self.args.train_batch_size,
            shuffle = True,
            num_workers = 0,
            collate_fn = MedDataset.collate_fn
        )

    def get_test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        return DataLoader(
            dataset = self.test_dataset,
            batch_size = self.args.test_batch_size,
            shuffle = True,
            num_workers = 0,
            collate_fn = MedDataset.collate_fn
        )
    def save(self, name, model):
        """
        print("Model's state_dict:")
        for param_tensor in self.model.state_dict():
            print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())
        
        # Print optimizer's state_dict
        print("Optimizer's state_dict:")
        for var_name in self.optimizer.state_dict():
            print(var_name, "\t", self.optimizer.state_dict()[var_name])
        """
        torch.save(model.state_dict(), f'save_models/{name}.model')
    def load(self, name, model):
        
        model.load_state_dict(torch.load(f'save_models/{name}.model'))
        """
        print("Model's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor][0])
            break
        print("Model's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor][0])
            break
        """
        
        model.load_state_dict(torch.load(f'save_models/{name}.model'))
        return model
        