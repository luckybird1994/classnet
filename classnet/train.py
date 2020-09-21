import torch
import os,cv2
import sys
import numpy as np
import pytorch_wasserstein

_wasserstein_loss = pytorch_wasserstein.Wasserstein( window_size=11,size_average=True )

def fm_loss( y_logits, y_true):

    y_pred = torch.sigmoid(y_logits)

    TP = ( y_pred * y_true).sum(dim=[2,3] )
    FP = ( y_pred * (1-y_true) ).sum(dim=[2,3] )
    FN = ( (1 - y_pred) *  y_true ).sum(dim=[2,3] )

    p = TP / ( TP + FP + 1e-7 )
    r = TP / ( TP + FN + 1e-7 )

    fm = 1.3 * p * r / ( 0.3 * p + r + 1e-7 )
    fm = fm.clamp( min = 1e-7, max = 1 - 1e-7 )

    return 1 - fm.mean()

class Trainer(object):

    def __init__( self,mode,model,config,device):

        assert mode in ['training', 'inference']
        self.mode = mode
        self.model = model
        self.cuda = torch.cuda.is_available()
        self.epoch = 0
        self.config = config
        self.device = device

    def train( self,val_loader,val_dataset ):

        dataloaders = {'val': val_loader}
        for phase in ['val']:

            self.model.eval()
            bar_steps = len(dataloaders[phase])
            process_bar = ShowProcess(bar_steps)

            save_sal_path = os.path.join("salmap")
            if not os.path.exists(save_sal_path):
                os.makedirs(save_sal_path)

            for i, data in enumerate(dataloaders[phase], 0):

                inputs, labels = data
                inputs, labels = inputs.to(self.device).float(), labels.to(self.device).float()

                with torch.set_grad_enabled(phase == 'train'):
                    
                    supervision1 = self.model(inputs)
                    supervision1 = torch.sigmoid(supervision1)
                    supervision1 = supervision1.clamp(0, 1)
                    output = supervision1.detach().cpu().numpy()
                    image = output[0, :, :, :]
                    image = np.transpose(image, (1, 2, 0))
                    image = image[:, :, 0]
                    image = image * 255.0
                    image = np.round(image)
                    image = np.uint8(image)
                    cv2.imwrite(os.path.join( save_sal_path,val_dataset.examples[i]["label_name"]), image)   
                process_bar.show_process()
            process_bar.close()
            
        print( "val finshed!")

    def load_weights(self, file_path):
        
        checkpoint = torch.load(file_path,map_location=self.device)
        self.model.load_state_dict(checkpoint)

class ShowProcess():


    i = 0
    max_steps = 0
    max_arrow = 50

    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.i = 0

    def show_process(self, i=None):
        if i is not None:
            self.i = i
        else:
            self.i += 1
        num_arrow = int(self.i * self.max_arrow / self.max_steps)
        num_line = self.max_arrow - num_arrow
        percent = self.i * 100.0 / self.max_steps
        process_bar = '[' + '>' * num_arrow + '-' * num_line + ']' \
                      + '%.2f' % percent + '%' + '\r'
        sys.stdout.write(process_bar)
        sys.stdout.flush()

    def close(self, words='done'):
        print('')
        self.i = 0

if __name__ == "__main__":

    input = torch.randn(  2,3,2,2)
    y_true = torch.randn( 2,3,2,2 )
    fm_loss(y_logits = input, y_true = y_true )

