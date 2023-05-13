import torch
import torch.nn as nn
import torch.nn.functional as F


class ResUnet(nn.Module):


    def __init__(self, in_planes, out_planes):
        super(ResUnet, self).__init__()
        
        self.bpc = 64  # basic plane count
        self.stride = 1
        self.lstm_input_size = 
        self.lstm_hidden_size =
        self.lstm_layers = 

        stride = self.stride


        # input to basic block conversion
        self.conv_input = nn.Conv2d(in_planes, self.bpc, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(64)

        # encoder inintialization 
        self.encoder_init()

        self.decoder_init()

        self.conv_output = nn.Conv2d(self.bpc, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor = 2, mode = 'nearest')


    def encoder_init(self):

        bpc = self.bpc
        stride = self.stride

        # Basic Block 1---------------------
        self.enc_conv11 = nn.Conv2d(bpc, bpc, kernel_size=3, stride=stride, padding=1, bias=False)
        self.enc_bn11 = nn.BatchNorm2d(bpc)
        self.enc_conv12 = nn.Conv2d(bpc, bpc, kernel_size=3, stride=stride, padding=1, bias=False)
        self.enc_bn12 = nn.BatchNorm2d(bpc)
        self.enc_conv13 = nn.Conv2d(bpc, bpc, kernel_size=3, stride=stride, padding=1, bias=False)
        self.enc_bn13 = nn.BatchNorm2d(bpc)
        self.enc_conv14 = nn.Conv2d(bpc, bpc, kernel_size=3, stride=stride, padding=1, bias=False)
        self.enc_bn14 = nn.BatchNorm2d(bpc)


        # Basic Block 2--------------------
        self.enc_conv21 = nn.Conv2d(bpc, 2*bpc, kernel_size=3, stride=stride, padding=1, bias=False)
        self.enc_bn21 = nn.BatchNorm2d(2*bpc)
        self.enc_conv22 = nn.Conv2d(2*bpc, 2*bpc, kernel_size=3, stride=stride, padding=1, bias=False)
        self.enc_bn22 = nn.BatchNorm2d(2*bpc)
        self.enc_conv23 = nn.Conv2d(2*bpc, 2*bpc, kernel_size=3, stride=stride, padding=1, bias=False)
        self.enc_bn23 = nn.BatchNorm2d(2*bpc)
        self.enc_conv24 = nn.Conv2d(2*bpc, 2*bpc, kernel_size=3, stride=stride, padding=1, bias=False)
        self.enc_bn24 = nn.BatchNorm2d(2*bpc)

        # Basic Block 3----------------------
        self.enc_conv31 = nn.Conv2d(2*bpc, 4*bpc, kernel_size=3, stride=stride, padding=1, bias=False)
        self.enc_bn31 = nn.BatchNorm2d(4*bpc)
        self.enc_conv32 = nn.Conv2d(4*bpc, 4*bpc, kernel_size=3, stride=stride, padding=1, bias=False)
        self.enc_bn32 = nn.BatchNorm2d(4*bpc)
        self.enc_conv33 = nn.Conv2d(4*bpc, 4*bpc, kernel_size=3, stride=stride, padding=1, bias=False)
        self.enc_bn33 = nn.BatchNorm2d(4*bpc)
        self.enc_conv34 = nn.Conv2d(4*bpc, 4*bpc, kernel_size=3, stride=stride, padding=1, bias=False)
        self.enc_bn34 = nn.BatchNorm2d(4*bpc)

        # Basic Block4-------------------------
        self.enc_conv41 = nn.Conv2d(4*bpc, 8*bpc, kernel_size=3, stride=stride, padding=1, bias=False)
        self.enc_bn41 = nn.BatchNorm2d(8*bpc)
        self.enc_conv42 = nn.Conv2d(8*bpc, 8*bpc, kernel_size=3, stride=stride, padding=1, bias=False)
        self.enc_bn42 = nn.BatchNorm2d(8*bpc)
        self.enc_conv43 = nn.Conv2d(8*bpc, 8*bpc, kernel_size=3, stride=stride, padding=1, bias=False)
        self.enc_bn43 = nn.BatchNorm2d(8*bpc)
        self.enc_conv44 = nn.Conv2d(8*bpc, 8*bpc, kernel_size=3, stride=stride, padding=1, bias=False)
        self.enc_bn44 = nn.BatchNorm2d(8*bpc)


    def lstm_cell(self):
        input_size = self.lstm_input_size
        hidden_size = self.lstm_hidden_size
        layers = self.lstm_layers

        self.lstm_layer = nn.LSTM(input_size = , hidden_size = , num_layers = )



    def decoder_init(self):

        bpc = self.bpc
        stride = self.stride

        # self.dec_conv10 = nn.Conv2d(8*bpc, 8*bpc, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.dec_bn10 = nn.BatchNorm2d(8*bpc)

        # Basic Block 1 ------------------------------
        self.dec_conv11 = nn.Conv2d(8*bpc, 4*bpc, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dec_bn11 = nn.BatchNorm2d(4*bpc)
        self.dec_conv12 = nn.Conv2d(4*bpc, 4*bpc, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dec_bn12 = nn.BatchNorm2d(4*bpc)
        self.dec_conv13 = nn.Conv2d(4*bpc, 4*bpc, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dec_bn13 = nn.BatchNorm2d(4*bpc)
        self.dec_conv14 = nn.Conv2d(4*bpc, 4*bpc, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dec_bn14 = nn.BatchNorm2d(4*bpc)


        # Basic Block 2 ----------------------------
        self.dec_conv21 = nn.Conv2d(4*bpc, 2*bpc, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dec_bn21 = nn.BatchNorm2d(2*bpc)
        self.dec_conv22 = nn.Conv2d(2*bpc, 2*bpc, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dec_bn22 = nn.BatchNorm2d(2*bpc)
        self.dec_conv23 = nn.Conv2d(2*bpc, 2*bpc, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dec_bn23 = nn.BatchNorm2d(2*bpc)
        self.dec_conv24 = nn.Conv2d(2*bpc, 2*bpc, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dec_bn24 = nn.BatchNorm2d(2*bpc)

        # Basic Block 3 -----------------------------
        self.dec_conv31 = nn.Conv2d(2*bpc, 1*bpc, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dec_bn31 = nn.BatchNorm2d(1*bpc)
        self.dec_conv32 = nn.Conv2d(1*bpc, 1*bpc, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dec_bn32 = nn.BatchNorm2d(1*bpc)
        self.dec_conv33 = nn.Conv2d(1*bpc, 1*bpc, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dec_bn33 = nn.BatchNorm2d(1*bpc)
        self.dec_conv34 = nn.Conv2d(1*bpc, 1*bpc, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dec_bn34 = nn.BatchNorm2d(1*bpc)

        # Basic Block 4 -----------------------------------
        self.dec_conv41 = nn.Conv2d(1*bpc, 1*bpc, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dec_bn41 = nn.BatchNorm2d(1*bpc)
        self.dec_conv42 = nn.Conv2d(1*bpc, 1*bpc, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dec_bn42 = nn.BatchNorm2d(1*bpc)
        self.dec_conv43 = nn.Conv2d(1*bpc, 1*bpc, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dec_bn43 = nn.BatchNorm2d(1*bpc)
        self.dec_conv44 = nn.Conv2d(1*bpc, 1*bpc, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dec_bn44 = nn.BatchNorm2d(1*bpc)


    def forward(self, x):

        # input basic conversion
        out = F.relu(self.bn_input(self.conv_input(x)))

        # -------encoder -------------------------------
        # basic block 1
        out = F.relu(self.enc_bn11(self.enc_conv11(out)))
        out = F.relu(self.enc_bn12(self.enc_conv12(out)))
        out = F.relu(self.enc_bn13(self.enc_conv13(out)))
        out = nn.MaxPool2d(out)
        out1 = F.relu(self.enc_bn14(self.enc_conv14(out)))


        # basic block 2
        out = F.relu(self.enc_bn11(self.enc_conv11(out1)))
        out = F.relu(self.enc_bn12(self.enc_conv12(out)))
        out = F.relu(self.enc_bn13(self.enc_conv13(out)))
        out = nn.MaxPool2d(out)
        out2 = F.relu(self.enc_bn14(self.enc_conv14(out)))


        #basic block 3
        out = F.relu(self.enc_bn11(self.enc_conv11(out2)))
        out = F.relu(self.enc_bn12(self.enc_conv12(out)))
        out = F.relu(self.enc_bn13(self.enc_conv13(out)))
        out = nn.MaxPool2d(out)
        out3 = F.relu(self.enc_bn14(self.enc_conv14(out)))


        #basic block 4
        out = F.relu(self.enc_bn11(self.enc_conv11(out3)))
        out = F.relu(self.enc_bn12(self.enc_conv12(out)))
        out = F.relu(self.enc_bn13(self.enc_conv13(out)))
        out = nn.MaxPool2d(out)
        out4 = F.relu(self.enc_bn14(self.enc_conv14(out)))



        lstm_out, (hn,cn) = self.lstm_layer(out4)


        # ------decoder-----------------

        out = self.upsample(lstm_out)
        # out4 = F.relu(self.dec_bn10(self.dec_conv10(out)))

        # basic block 1
        out = F.relu(self.dec_bn11(self.dec_conv11(out)))
        out = F.relu(self.dec_bn12(self.dec_conv12(out)))
        out = F.relu(self.dec_bn13(self.dec_conv13(out)))
        out = F.relu(self.dec_bn14(self.dec_conv14(out)))

        out = torch.cat((out,out3))


        # basic block 2
        out = F.relu(self.dec_bn21(self.dec_conv21(out)))
        out = F.relu(self.dec_bn22(self.dec_conv22(out)))
        out = F.relu(self.dec_bn23(self.dec_conv23(out)))
        out = F.relu(self.dec_bn24(self.dec_conv24(out)))

        out = torch.cat((out,out2))


        #basic block 3
        out = F.relu(self.dec_bn31(self.dec_conv31(out)))
        out = F.relu(self.dec_bn32(self.dec_conv32(out)))
        out = F.relu(self.dec_bn33(self.dec_conv33(out)))
        out = F.relu(self.dec_bn34(self.dec_conv34(out)))

        out = torch.cat((out,out1))


        #basic block 4
        out = F.relu(self.dec_bn41(self.dec_conv41(out)))
        out = F.relu(self.dec_bn42(self.dec_conv42(out)))
        out = F.relu(self.dec_bn43(self.dec_conv43(out)))
        out = F.relu(self.dec_bn44(self.dec_conv44(out)))

        out = self.conv_output(out)


        return out 