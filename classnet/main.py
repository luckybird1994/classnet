from classnet import ClassNet
from train import Trainer
from config import Config
from data_loader import DatasetVal
from torch.utils.data import DataLoader
import torch

if __name__=='__main__':

    val_dataset = DatasetVal()
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=8)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    net = ClassNet().to(device)
    cfg = Config()

    trainer = Trainer( 'inference',net,cfg,device)
    trainer.load_weights( 'model/model.pth')
    trainer.train( val_loader,val_dataset )
