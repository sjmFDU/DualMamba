import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from einops import rearrange


class Order_Attention(nn.Module):
    def __init__(self, hidden_size, attention_size):
        super(Order_Attention, self).__init__()
        self.w_omega = nn.Parameter(torch.randn(hidden_size, attention_size)) # [hidden_size, attention_size]
        self.b_omega = nn.Parameter(torch.randn(attention_size)) # [attention_size]
        self.u_omega = nn.Parameter(torch.randn(attention_size)) # [attention_size]

    def forward(self, inputs):
        # inputs: [seq_len, batch_size, hidden_size]
        inputs = inputs.permute(1, 0, 2) # inputs: [batch_size, seq_len, hidden_size]
        v = torch.tanh(torch.matmul(inputs, self.w_omega) + self.b_omega) # v: [batch_size, seq_len, attention_size]
        vu = torch.matmul(v, self.u_omega) # vu: [batch_size, seq_len]
        alphas = F.softmax(vu, dim=1) # alphas: [batch_size, seq_len]
        output = inputs * alphas.unsqueeze(-1) # output: [batch_size, STEP, hidden_size]
        return output, alphas # output: [batch_size, hidden_size], alphas: [batch_size, seq_len]

class zhouEightDRNN_kamata_LSTM(nn.Module):
    @staticmethod
    def weight_init(m):
        # All weight matrices in our RNN and bias vectors are initialized with a uniform distribution, and the values of these weight matrices and bias vectors are initialized in the range [−0.1,0.1]
        if isinstance(m, (nn.Linear, nn.GRU, nn.Conv3d,nn.LSTM)):
            init.kaiming_uniform_(m.weight.data)
            init.kaiming_uniform_(m.bias.data)

    def __init__(self, input_channels, n_classes, patch_size=5):
        # The proposed network model uses a single recurrent layer that adopts our modified GRUs of size 64 with sigmoid gate activation and PRetanh activation functions for hidden representations
        super(zhouEightDRNN_kamata_LSTM, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.gru = nn.GRU(patch_size ** 2, patch_size ** 2, 1, bidirectional=False,
                          batch_first=False)  # TODO: try to change this ?
        self.gru_2 = nn.GRU(input_channels, input_channels, 1, bidirectional=False)
        self.gru_2_1 = nn.GRU(input_channels, patch_size**2, 1, bidirectional=False)
        self.gru_2_2 = nn.GRU(input_channels, patch_size**2, 1, bidirectional=False)
        self.gru_2_3 = nn.GRU(input_channels, patch_size**2, 1, bidirectional=False)
        self.gru_2_4 = nn.GRU(input_channels, patch_size**2, 1, bidirectional=False)
        self.gru_2_5 = nn.GRU(input_channels, patch_size**2, 1, bidirectional=False)
        self.gru_2_6 = nn.GRU(input_channels, patch_size**2, 1, bidirectional=False)
        self.gru_2_7 = nn.GRU(input_channels, patch_size**2, 1, bidirectional=False)
        self.gru_2_8 = nn.GRU(input_channels, patch_size**2, 1, bidirectional=False)
        self.lstm_2_1 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.lstm_2_2 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.lstm_2_3 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.lstm_2_4 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.lstm_2_5 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.lstm_2_6 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.lstm_2_7 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.lstm_2_8 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        # self.lstm_2_1 = nn.RNN(input_channels, 64, 1, bidirectional=False)
        # self.lstm_2_2 = nn.RNN(input_channels, 64, 1, bidirectional=False)
        # self.lstm_2_3 = nn.RNN(input_channels, 64, 1, bidirectional=False)
        # self.lstm_2_4 = nn.RNN(input_channels, 64, 1, bidirectional=False)
        # self.lstm_2_5 = nn.RNN(input_channels, 64, 1, bidirectional=False)
        # self.lstm_2_6 = nn.RNN(input_channels, 64, 1, bidirectional=False)
        # self.lstm_2_7 = nn.RNN(input_channels, 64, 1, bidirectional=False)
        # self.lstm_2_8 = nn.RNN(input_channels, 64, 1, bidirectional=False)
        self.lstm_stra_1 = nn.LSTM(64, 64, 1, bidirectional=False)
        # self.gru_3 = nn.GRU(input_channels, patch_size ** 2, 1, bidirectional=True)
        # self.gru_3_1 = nn.GRU(input_channels, patch_size ** 2, 1, bidirectional=True)
        # self.gru_3_2 = nn.GRU(input_channels, patch_size ** 2, 1, bidirectional=True)
        # self.gru_3_3 = nn.GRU(input_channels, patch_size ** 2, 1, bidirectional=True)
        # self.gru_3_4 = nn.GRU(input_channels, patch_size ** 2, 1, bidirectional=True)
        # self.gru_3_5 = nn.GRU(input_channels, patch_size ** 2, 1, bidirectional=True)
        # self.gru_3_6 = nn.GRU(input_channels, patch_size ** 2, 1, bidirectional=True)
        # self.gru_3_7 = nn.GRU(input_channels, patch_size ** 2 , 1, bidirectional=True)
        # self.gru_3_8 = nn.GRU(input_channels, patch_size ** 2, 1, bidirectional=True)
        self.scan_order = Order_Attention(64, 64)

        self.gru_4 = nn.GRU(64, 64, 1)
        self.lstm_4 = nn.LSTM(patch_size ** 2, 64, 1)
        self.conv = nn.Conv2d(input_channels,out_channels=input_channels, kernel_size=(3,3),stride=(3,3))
        self.transformer = nn.Transformer(nhead=16, num_encoder_layers=12)
        self.gru_bn = nn.BatchNorm1d(patch_size ** 2 * input_channels)
        self.lstm_bn_1 = nn.BatchNorm1d((64)*1)
        self.lstm_bn_1_2 = nn.BatchNorm1d((64) * (patch_size)**2)
        self.lstm_bn_2 = nn.BatchNorm1d((64)*8)
        self.lstm_bn_2_2 = nn.BatchNorm1d((64) * 8 * patch_size**2)
        self.gru_bn_2 = nn.BatchNorm1d((patch_size ** 2) * (patch_size**2))
        self.gru_bn_3 = nn.BatchNorm1d((patch_size ** 2) * (patch_size ** 2) * 2)
        self.gru_bn_4 = nn.BatchNorm1d(8 * 64)
        self.gru_bn_laststep = nn.BatchNorm1d(64)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(patch_size ** 2 * input_channels, n_classes)
        self.lstm_fc_1 = nn.Linear(64 * 1, n_classes)
        self.lstm_fc_1_2 = nn.Linear(64 * (patch_size**2), n_classes)
        self.lstm_fc_2 = nn.Linear(64*8,n_classes)
        self.lstm_fc_2_2 = nn.Linear(64 * 8 * patch_size**2, n_classes)
        self.fc_2 = nn.Linear((patch_size ** 2) * (patch_size**2), n_classes)
        self.fc_3 = nn.Linear((patch_size ** 2) * (patch_size ** 2) * 2, n_classes)
        self.fc_4 = nn.Linear(8 * 64, n_classes)
        self.fc_laststep = nn.Linear(64, n_classes)
        self.softmax = nn.Softmax()
        self.linear = nn.Linear(64,64)
        self.aux_loss_weight = 1

    def forward(self, x): #初始是第1方向
        #print('x.shape1',x.shape)
        x = x.squeeze(1)
        
        #print('x.shape2', x.shape)
        # x = self.conv(x)
        #print('x.shape3', x.shape)
        #生成第1和7
        x1_0 = x[:, :, 0, :]
        x1_1 = x[:, :, 1, :]
        x1_2 = x[:, :, 2, :]
        x1_3 = x[:, :, 3, :]
        x1_4 = x[:, :, 4, :]
        x1_1f = torch.flip(x1_1, [2])
        x1_3f = torch.flip(x1_3, [2])
        # plt.subplot(3, 4, 9).set_title('Spectral signatures in a patch')
        # direction_1_showpicture = torch.cat([x1_0, x1_1f, x1_2, x1_3f, x1_4], dim=2)
        # plt.xlabel('Band Numbers', fontdict={'size': 15}, fontweight='bold')
        # plt.ylabel('Spectral Values Values', fontdict={'size': 15},fontweight='bold')
        # plt.plot(direction_1_showpicture[0, :, :].cpu().detach().numpy())
        # direction_1 = torch.cat([x1_0, x1_1f, x1_2], dim=2)
        direction_1 = torch.cat([x1_0, x1_1f, x1_2,x1_3f,x1_4], dim=2)

        #print('d1',direction_1.shape)
        # print('d1',direction_1.shape)
        direction_7 = torch.flip(direction_1,[2])

        #生成第2和8
        x2_0 = x[:, :, :, 0]
        x2_1 = x[:, :, :, 1]
        x2_2 = x[:, :, :, 2]
        x2_3 = x[:, :, :, 3]
        x2_4 = x[:, :, :, 4]
        x2_1f = torch.flip(x2_1, [2])
        x2_3f = torch.flip(x2_3, [2])
        # direction_2 = torch.cat([x2_0, x2_1f, x2_2], dim=2)
        direction_2 = torch.cat([x2_0, x2_1f,x2_2,x2_3f,x2_4], dim=2)
        direction_8 = torch.flip(direction_2, [2])

        # 生成3和5
        x3_0 = x[:, :, 0, :]
        x3_1 = x[:, :, 1, :]
        x3_2 = x[:, :, 2, :]
        x3_3 = x[:, :, 3, :]
        x3_4 = x[:, :, 4, :]
        x3_0f = torch.flip(x3_0, [2])
        x3_2f = torch.flip(x3_2, [2])
        x3_4f = torch.flip(x3_4, [2])
        # direction_3 = torch.cat([x3_0f, x3_1, x3_2f], dim=2)
        direction_3 = torch.cat([x3_0f, x3_1, x3_2f,x3_3,x3_4f], dim=2)
        direction_5 = torch.flip(direction_3, [2])

        #生成4和6
        x4_0 = x[:, :, :, 0]
        x4_1 = x[:, :, :, 1]
        x4_2 = x[:, :, :, 2]
        x4_3 = x[:, :, :, 3]
        x4_4 = x[:, :, :, 4]
        x4_1f = torch.flip(x4_1, [2])
        x4_3f = torch.flip(x4_3, [2])
        # direction_4 = torch.cat([x4_2, x4_1f, x4_0], dim=2)
        direction_4 = torch.cat([x4_4, x4_3f, x4_2, x4_1f, x4_0], dim=2)
        # print('d4', direction_4.shape)
        direction_6 = torch.flip(direction_4, [2])
        x8r = direction_8.permute(2, 0, 1)
        x7r = direction_7.permute(2, 0, 1)
        x6r = direction_6.permute(2, 0, 1)
        x5r = direction_5.permute(2, 0, 1)
        x4r = direction_4.permute(2, 0, 1)
        x3r = direction_3.permute(2, 0, 1)
        x2r = direction_2.permute(2, 0, 1)
        x1r = direction_1.permute(2, 0, 1)
        h0_x1r = torch.zeros(1, x1r.size(1), 64).to(device='cuda')
        c0_x1r = torch.zeros(1, x1r.size(1), 64).to(device="cuda")
        x1r, x1r_hidden = self.lstm_2_1(x1r)
        x1r_laststep = x1r[-1]
        x1r_laststep = self.relu(x1r_laststep)
        x1r_laststep = torch.unsqueeze(x1r_laststep, dim=0)
        h0_x2r = torch.zeros(1, x2r.size(1), 64).to(device='cuda')
        c0_x2r = torch.zeros(1, x2r.size(1), 64).to(device="cuda")
        x2r = self.lstm_2_2(x2r)[0]
        # x2r = self.gru_2_2(x2r)[0] #把x1r经过RNN的值，作为x2r的输入
        x2r_laststep = x2r[-1]
        # x2r_laststep_2 = x2r[-1, 1, :]
        x2r_laststep = self.relu(x2r_laststep)
        x2r_laststep = torch.unsqueeze(x2r_laststep, dim=0)
        h0_x3r = torch.zeros(1, x3r.size(1), 64).to(device='cuda')
        c0_x3r = torch.zeros(1, x3r.size(1), 64).to(device="cuda")
        x3r = self.lstm_2_3(x3r)[0]
        # x3r = self.gru_2_3(x3r)[0]
        x3r_laststep = x3r[-1]
        # x3r_laststep_2 = x3r[-1, 1, :]
        x3r_laststep = self.relu(x3r_laststep)
        x3r_laststep = torch.unsqueeze(x3r_laststep, dim=0)
        h0_x4r = torch.zeros(1, x4r.size(1), 64).to(device='cuda')
        c0_x4r = torch.zeros(1, x4r.size(1), 64).to(device="cuda")
        x4r = self.lstm_2_4(x4r)[0]
        # x4r = self.gru_2_4(x4r)[0]
        x4r_laststep = x4r[-1]
        # x4r_laststep_2 = x4r[-1, 1, :]
        x4r_laststep = self.relu(x4r_laststep)
        x4r_laststep = torch.unsqueeze(x4r_laststep, dim=0)
        h0_x5r = torch.zeros(1, x5r.size(1), 64).to(device='cuda')
        c0_x5r = torch.zeros(1, x5r.size(1), 64).to(device="cuda")
        x5r = self.lstm_2_5(x5r)[0]
        # x5r = self.gru_2_5(x5r)[0]
        x5r_laststep = x5r[-1]
        # x5r_laststep_2 = x5r[-1, 1, :]
        x5r_laststep = self.relu(x5r_laststep)
        x5r_laststep = torch.unsqueeze(x5r_laststep, dim=0)
        h0_x6r = torch.zeros(1, x6r.size(1), 64).to(device='cuda')
        c0_x6r = torch.zeros(1, x6r.size(1), 64).to(device="cuda")
        x6r = self.lstm_2_6(x6r)[0]
        # x6r = self.gru_2_6(x6r)[0]
        x6r_laststep = x6r[-1]
        # x6r_laststep_2 = x6r[-1, 1, :]
        x6r_laststep = self.relu(x6r_laststep)
        x6r_laststep = torch.unsqueeze(x6r_laststep, dim=0)
        h0_x7r = torch.zeros(1, x7r.size(1), 64).to(device='cuda')
        c0_x7r = torch.zeros(1, x7r.size(1), 64).to(device="cuda")
        x7r = self.lstm_2_7(x7r)[0]
        # x7r = self.gru_2_7(x7r)[0]
        x7r_laststep = x7r[-1]
        # x7r_laststep_2 = x7r[-1, 1, :]
        x7r_laststep = self.relu(x7r_laststep)
        x7r_laststep = torch.unsqueeze(x7r_laststep, dim=0)
        h0_x8r = torch.zeros(1, x8r.size(1), 64).to(device='cuda')
        c0_x8r = torch.zeros(1, x8r.size(1), 64).to(device="cuda")
        x8r = self.lstm_2_8(x8r)[0]
        # x8r = self.gru_2_8(x8r)[0]
        x8r_laststep = x8r[-1]
        # x8r_laststep_2 = x8r[-1, 1, :]
        x8r_laststep = self.relu(x8r_laststep)
        x8r_laststep = torch.unsqueeze(x8r_laststep, dim=0)
        '----show attetntion function------------------------------------------------------'
        def showattention(inputseq):
            allpixel = inputseq[:, 1, :]
            linear1 = nn.Linear(allpixel.size(1),allpixel.size(1)).to( device='cuda')
            allpixel = linear1(allpixel)

            # centralstep = allpixel[12,:]
            laststep = inputseq[-1, 1, :]
            laststep = linear1(laststep)

            output = torch.matmul(allpixel, laststep.transpose(0,-1))

            pairdis = nn.PairwiseDistance()
            cos = nn.CosineSimilarity(dim=-1)

            output_pair = pairdis(allpixel,laststep) * -1
            # output_pair = cos(allpixel, laststep)

            softmax = nn.Softmax()
            output = softmax(output)
            output_pair = softmax(output_pair)
            output = output.unsqueeze(0)
            output_pair = output_pair.unsqueeze(0)
            #print('cos',output_pair.shape)
            return output,output_pair
        
        x_strategy_1 = torch.cat([x8r_laststep,x7r_laststep,x6r_laststep,x5r_laststep,x4r_laststep,x3r_laststep,x2r_laststep,x1r_laststep],dim=0)
        x_strategy_1 = rearrange(x_strategy_1, 'n b d -> b n d')
        x_strategy_1 = self.scan_order(x_strategy_1)
        # print('x_strategy_1', x_strategy_1.shape) #(8 , batch, 64)
        # x_strategy_1 = x_strategy_1.permute(1, 0, 2).contiguous()#(100,64,8)
        h0_last = torch.zeros(1, x_strategy_1.size(1), 64).to(device='cuda')
        c0_last = torch.zeros(1, x_strategy_1.size(1), 64).to(device="cuda")
        x_strategy_1 = self.lstm_stra_1(x_strategy_1)
        # x_strategy_1 = self.gru_4(x_strategy_1)[0]
        x_strategy_1_laststep = x_strategy_1[-1]
        # x_strategy_1_laststep_2 = x_strategy_1[-1, 1, :]
        # x_strategy_1_laststep = x_strategy_1.permute(1, 2, 0).contiguous()
        # print('x_strategy_1_laststep',x_strategy_1_laststep.shape)
        # np.save('x_strategy_1_laststep', x_strategy_1_laststep.cpu().detach().numpy(), allow_pickle=True)
        '------------------------------------------'
        'calzulate RNN attention for 8 directions'

        '-------------------------------------------------------------------------------------'
        x_strategy_1_laststep = x_strategy_1_laststep.permute(0, 1).contiguous()
        x_strategy_1_laststep = x_strategy_1_laststep.view(x_strategy_1_laststep.size(0), -1)
        # x_strategy_1_laststep = self.gru_bn_4(x_strategy_1_laststep)
        x_strategy_1_laststep = self.gru_bn_laststep(x_strategy_1_laststep)
        x_strategy_1_laststep = self.prelu(x_strategy_1_laststep)
        x_strategy_1_laststep = self.dropout(x_strategy_1_laststep)
        # x_strategy_1_laststep = self.fc_4(x_strategy_1_laststep)
        x_strategy_1_laststep = self.fc_laststep(x_strategy_1_laststep)

        x = x_strategy_1_laststep
        # 下面改变输入值，确定使用哪个方向

        return x

def msrnn(model_config):
    model = zhouEightDRNN_kamata_LSTM(**model_config)
    return model