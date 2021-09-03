import numpy as np
import os.path as osp
import pickle

import torch
from torch import nn
import joblib
import fsspec
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from molgpka.utils.net import GCNNet


class GraphModelTrainer:
    def __init__(self, data, batch_size=128, lr=0.0001):

        self.data_path = data
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.lr = lr
        self.model = GCNNet().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.7, patience=10, min_lr=0.00001
        )
        self.train_loader = None
        self.valid_loader = None
        self.prepare_dataset(data)

    def prepare_dataset(self, data=None):
        if data is None:
            data = self.data_path
        data = joblib.load(data)
        train_data = data["train"]
        valid_data = data["valid"]
        self.train_loader = DataLoader(train_data, batch_size=self.batch_size)
        self.valid_loader = DataLoader(valid_data, batch_size=self.batch_size)

    def train(self):
        self.model.train()
        loss_all = 0
        t_len = len(self.train_loader.dataset)
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.mse_loss(output, data.pka)
            loss.backward()
            loss_all += loss.item() * data.num_graphs
            self.optimizer.step()
        return loss_all / t_len

    def test(self, loader):
        self.model.eval()
        correct = 0
        mae = 0
        d_len = len(loader.dataset)
        for data in loader:
            data = data.to(self.device)
            output = self.model(data)

            correct += F.mse_loss(output, data.pka).item() * data.num_graphs
            mae += F.l1_loss(output, data.pka).item() * data.num_graphs
        return correct / d_len, mae / d_len

    def run(self, epochs=1000, output_path="models/weight_{epoch}.pth"):
        hist = {"loss": [], "mse": [], "mae": []}
        for epoch in range(1, epochs + 1):
            train_loss = self.train()
            mse, mae = self.test(self.valid_loader)

            hist["loss"].append(train_loss)
            hist["mse"].append(mse)
            hist["mae"].append(mae)
            if mse <= min(hist["mse"]):
                with fsspec.open(output_path.format(epoch), "rb") as IN:
                    torch.save(self.model.state_dict(), IN)
            print(
                f"Epoch: {epoch}, Train loss: {train_loss:.3}, Test_mse: {mse:.3}, Test_mae: {mae:.3}"
            )
