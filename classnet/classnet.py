import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch.nn.init as init
import numpy as np

from resnet import resnet50

class PAM(nn.Module):

    def __init__(self, in_dim):
        super(PAM, self).__init__()

        self.conv_high = nn.Conv2d(in_channels= in_dim, out_channels= in_dim // 8, kernel_size=1)
        self.conv_low = nn.Conv2d(in_channels=in_dim, out_channels= in_dim // 8, kernel_size=1)

        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_high,x_low ):

        m_batchsize, C_high, height_high, width_high = x_high.size()
        _, C_low, height_low, width_low = x_low.size()


        proj_high = self.conv_high(x_high).view(m_batchsize, -1, width_high * height_high)
        proj_low = self.conv_low(x_low).view(m_batchsize, -1, width_low * height_low)

        proj_value_low = self.value_conv(x_low).view(m_batchsize, -1, width_low * height_low)

        proj_query = proj_high.permute(0,2,1)
        proj_key = proj_low

        energy = torch.bmm( proj_query,proj_key )
        attention = self.softmax(energy)
       
        out_high = torch.bmm( attention,proj_value_low.permute(0,2,1) )
        out_high = out_high.view(m_batchsize, C_high, height_high, width_high)
        out_high = out_high * self.gamma + x_high

        return out_high

class CAM(nn.Module):

    def __init__(self, in_dim):
        super(CAM, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self,x_high,x_low ):

        m_batchsize, C_high, height_high, width_high = x_high.size()
        _, C_low, height_low, width_low = x_low.size()

        proj_query = x_low.view(m_batchsize, C_low, -1)
        proj_key = x_high.view(m_batchsize, C_high, -1).permute(0, 2, 1)

        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        
        proj_value = x_high.view(m_batchsize, C_high, -1)
        out_low = torch.bmm(attention, proj_value)
        out_low = out_low.view(m_batchsize, C_high, height_low, height_low )
        out_low = self.gamma * out_low + x_low

        return out_low

class Fusion(nn.Module):

    def __init__(self):

        super( Fusion, self ).__init__()

        self.reducedim = nn.Sequential( nn.Conv2d(192, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True) )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x ):

        x = self.reducedim( x )
        identity = x

        x = self.conv1(x)
        x = self.conv2(x)

        x_sptial = self.conv3(x)
        x_sptial = self.sigmoid(x_sptial)

        out = x_sptial * x
        out = out + identity
        return out

class Main_Net(nn.Module):

    def __init__(self,device = torch.device( "cuda:1" if torch.cuda.is_available() else "cpu" )):

        super(Main_Net, self).__init__()

        self.device = device
        resnet_model = resnet50(pretrained=True)

        self.res_conv1 = resnet_model.conv1
        self.res_bn1 = resnet_model.bn1
        self.res_relu = resnet_model.relu
        self.res_maxpool = resnet_model.maxpool

        self.res_layer1 = resnet_model.layer1
        self.res_layer2 = resnet_model.layer2
        self.res_layer3 = resnet_model.layer3
        self.res_layer4 = resnet_model.layer4
        self.res_pool = resnet_model.avgpool

        self.squeeze5 = nn.Sequential(nn.Conv2d(2048, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze4 = nn.Sequential(nn.Conv2d(1024, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze3 = nn.Sequential(nn.Conv2d(512, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze2 = nn.Sequential(nn.Conv2d(256, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze1 = nn.Sequential(nn.Conv2d(64, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.PAM2 = PAM(in_dim=64)
        self.PAM3 = PAM(in_dim=64)
        self.PAM4 = PAM(in_dim=64)

        self.CAM2 = CAM(in_dim=64)
        self.CAM3 = CAM(in_dim=64)
        self.CAM4 = CAM(in_dim=64)

        self.fusion2 = Fusion()
        self.fusion3 = Fusion()
        self.fusion4 = Fusion()

        self.extra_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.extra_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.extra_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.extra_conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.extra_conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.extra_conv6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.extra_conv7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.extra_conv8 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.extra_conv9 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.extra_conv10 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.extra_conv11 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.extra_conv12 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.extra_conv17 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.extra_conv18 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        self.supervision5 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.supervision4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.supervision3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.supervision1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        self.decoder4_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.decoder4_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.decoder3_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.decoder3_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.decoder2_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.decoder2_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):

        Conv1_out = self.res_conv1(x)
        Conv1_out = self.res_bn1(Conv1_out)
        Conv1_out = self.res_relu(Conv1_out)

        Conv1_out_maxpool = self.res_maxpool(Conv1_out)

        res2_out = self.res_layer1(Conv1_out_maxpool)
        res3_out = self.res_layer2(res2_out)
        res4_out = self.res_layer3(res3_out)
        res5_out = self.res_layer4(res4_out)
       
        global_poo1 = self.res_pool(res5_out)
        global_poo1 = F.interpolate( global_poo1, res5_out.size()[2:], mode='bilinear', align_corners=False )
        global_poo1 = global_poo1 + res5_out

        out2h,out3h,out4h,out5v = self.squeeze2(res2_out), self.squeeze3(res3_out), self.squeeze4(res4_out), self.squeeze5(global_poo1)

        #attention_4
        down_out4v = out5v
        down_out4v = self.extra_conv1(down_out4v)
        down_out4v = self.extra_conv2(down_out4v)
        left_out4h = self.extra_conv3(out4h)
        left_out4h = self.extra_conv4(left_out4h)
        up_down_out4v = F.interpolate(down_out4v, size=left_out4h.size()[2:], mode='bilinear', align_corners=False)
        attention4_high = self.PAM4( x_high = down_out4v,x_low = left_out4h)
        attention4_low = self.CAM4( x_high = up_down_out4v,x_low = left_out4h )

        #attention_3
        down_out3v = out5v
        down_out3v = self.extra_conv5(down_out3v)
        down_out3v = self.extra_conv6(down_out3v)
        left_out3h = self.extra_conv7(out3h)
        left_out3h = self.extra_conv8(left_out3h)
        up_down_out3v = F.interpolate(down_out3v, size=left_out3h.size()[2:], mode='bilinear', align_corners=False)
        attention3_high = self.PAM3( x_high=down_out3v, x_low=left_out3h )
        attention3_low = self.CAM3( x_high=up_down_out3v, x_low=left_out3h )

        #attention_2
        down_out2v = out5v
        down_out2v = self.extra_conv9(down_out2v)
        down_out2v = self.extra_conv10(down_out2v)
        left_out2h = self.extra_conv11(out2h)
        left_out2h = self.extra_conv12(left_out2h)
        up_down_out2v = F.interpolate(down_out2v, size=left_out2h.size()[2:], mode='bilinear', align_corners=False)
        attention2_high = self.PAM2(x_high=down_out2v, x_low=left_out2h)
        attention2_low = self.CAM2(x_high=up_down_out2v, x_low=left_out2h)
     
        #decoder5
        decoder_out5v = self.extra_conv17( out5v )
        decoder_out5v = self.extra_conv18( decoder_out5v )
        supervision5 = self.supervision5( decoder_out5v )
        supervision5 = F.interpolate(supervision5, size=x.size()[2:], mode='bilinear', align_corners=False)

        #decoder4
        decoder4_down = F.interpolate(decoder_out5v, size=attention4_low.size()[2:], mode='bilinear', align_corners=False)
        decoder4_left_low = attention4_low
        decoder4_left_high = F.interpolate(attention4_high, size=attention4_low.size()[2:], mode='bilinear', align_corners=False)

        decoder4_merged = torch.cat( (decoder4_down,decoder4_left_low,decoder4_left_high),1 )
        decoder4_fusion = self.fusion4( decoder4_merged )
        decoder4 = self.decoder4_conv1(decoder4_fusion)
        decoder4 = self.decoder4_conv2(decoder4)

        supervision4 = self.supervision4(decoder4)
        supervision4 = F.interpolate(supervision4, size=x.size()[2:], mode='bilinear', align_corners=False)

        #decoder3
        decoder3_down = F.interpolate(decoder4, size=attention3_low.size()[2:], mode='bilinear', align_corners=False)
        decoder3_left_low = attention3_low
        decoder3_left_high = F.interpolate(attention3_high, size=attention3_low.size()[2:], mode='bilinear', align_corners=False)

        decoder3_merged = torch.cat((decoder3_down, decoder3_left_low, decoder3_left_high), 1)
        decoder3_fusion = self.fusion3(decoder3_merged)
        decoder3 = self.decoder3_conv1(decoder3_fusion)
        decoder3 = self.decoder3_conv2(decoder3)

        supervision3 = self.supervision3(decoder3)
        supervision3 = F.interpolate(supervision3, size=x.size()[2:], mode='bilinear', align_corners=False)

        #decoder2
        decoder2_down = F.interpolate(decoder3, size=attention2_low.size()[2:], mode='bilinear', align_corners=False)
        decoder2_left_low = attention2_low
        decoder2_left_high = F.interpolate(attention2_high, size=attention2_low.size()[2:], mode='bilinear',align_corners=False)

        decoder2_merged = torch.cat((decoder2_down, decoder2_left_low, decoder2_left_high), 1)
        decoder2_fusion = self.fusion2(decoder2_merged)
        decoder2 = self.decoder2_conv1(decoder2_fusion)
        decoder2 = self.decoder2_conv2(decoder2)

        #decoder1
        supervision1 = self.supervision1(decoder2)
        supervision1 = F.interpolate(supervision1, size=x.size()[2:], mode='bilinear', align_corners=False)

        return supervision1

def ClassNet():

    model = Main_Net()
    return model

if __name__ == "__main__":

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    net = ClassNet().to(device)
    input = torch.randn(8,3,256,256).to(device)
    net(input)





