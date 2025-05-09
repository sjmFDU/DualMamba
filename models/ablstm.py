"""
# codes: https://github.com/MeiShaohui/Attention-based-Bidirectional-LSTM-Network
# paper:
Hyperspectral Image Classification Using Attention-Based Bidirectional Long Short-Term Memory Network
Shaohui Mei, Xingang Li, Xiao Liu, Huimin Cai, and Qian Du,
IEEE TGRS, 2021
"""
import torch
from torch import nn


class BiLSTM(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, dropout: float,
                 n_class: int):
        super(BiLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=1,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True,
                           dropout=dropout,
                           bidirectional=True)
        self.out = nn.Linear(in_features=2 * hidden_size, out_features=n_class)
        # weight init
        for name, param in self.rnn.named_parameters():
            if name.startswith("weight"):
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)
        nn.init.xavier_normal_(self.out.weight)
        nn.init.constant_(self.out.bias, 0)

    def forward(self, x):
        _, (h_n, _) = self.rnn(x)
        backward_out = h_n[-1, :, :]
        forward_out = h_n[-2, :, :]
        features = torch.cat((forward_out, backward_out), 1)
        x = self.out(features)

        return x


class SpatialAttention(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def __init__(self, input_c: int, patch_size: int):
        super(SpatialAttention, self).__init__()
        self.patch_size = patch_size
        self.conv1 = nn.Conv2d(input_c,
                               2 * input_c,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=(2 * input_c))
        self.pooling1 = nn.MaxPool2d(2, stride=2)
        self.P1 = nn.PReLU()
        self.conv2 = nn.Conv2d(2 * input_c, input_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=(input_c))
        self.pooling2 = nn.MaxPool2d(2, stride=2)
        self.P2 = nn.PReLU()

        self.apply(self.weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(dim=1)
        dx = self.patch_size // 2
        identity = x[:, :, dx, dx].unsqueeze(2).unsqueeze(3)
        out = self.P2(
            self.pooling2(
                self.bn2(
                    self.conv2(self.P1(self.pooling1(self.bn1(
                        self.conv1(x))))))))
        out = identity * out + identity

        return out.permute(0, 2, 1, 3).squeeze(3)


class SpectralAttention(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def __init__(self):
        super(SpectralAttention, self).__init__()
        self.conv1 = nn.Conv1d(1, 1, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(1)
        self.ac1 = nn.PReLU()
        self.conv2 = nn.Conv1d(1, 1, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(1)
        self.ac2 = nn.PReLU()

        self.apply(self.weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.ac1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ac2(out)

        out = out * x + x

        return out.transpose(1, 2)

class ABLSTM(nn.Module):
    def __init__(self, model_config):
        super(ABLSTM, self).__init__()
        self.model_config = model_config
        self.spatial_attention = SpatialAttention(input_c=model_config['in_chans'], patch_size=model_config['patch_size'])
        self.spectral_attention = SpectralAttention()
        self.bilstm = BiLSTM(hidden_size=64, num_layers=2, dropout=0.5, n_class=model_config['n_classes'])

    def forward(self, x):
        x = self.spatial_attention(x)
        x = self.spectral_attention(x)
        x = self.bilstm(x)
        return x

    def flops(self, shape=(1, 48, 13, 13), verbose=True):
        # shape = self.__input_shape__[1:]
        import copy, fvcore.nn.flop_count as flop_count, fvcore.nn.parameter_count as parameter_count
        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input,))

        del model, input
        #return sum(Gflops.values()) * 1e9
        print(f"params {params} MFLOPs {sum(Gflops.values()) }")

def ablstm(model_config):
    return ABLSTM(model_config)

'''def ablstm(dataset, patch_size):
    model = None
    if dataset == 'sa':
        model= nn.Sequential(
            SpatialAttention(input_c=204, patch_size=patch_size), 
            SpectralAttention(), 
            BiLSTM(hidden_size=64, num_layers=2, dropout=0.5, n_class=16)
            )

    elif dataset == 'pu':
        model= nn.Sequential(
            SpatialAttention(input_c=103, patch_size=patch_size), 
            SpectralAttention(), 
            BiLSTM(hidden_size=64, num_layers=2, dropout=0.5, n_class=9)
            )  

    elif dataset == 'whulk':
        model= nn.Sequential(
            SpatialAttention(input_c=270, patch_size=patch_size), 
            SpectralAttention(), 
            BiLSTM(hidden_size=64, num_layers=2, dropout=0.5, n_class=9)
            ) 

    elif dataset == 'hrl':
        model= nn.Sequential(
            SpatialAttention(input_c=176, patch_size=patch_size), 
            SpectralAttention(), 
            BiLSTM(hidden_size=64, num_layers=2, dropout=0.5, n_class=14)
            ) 
        
    return model'''


if __name__ == '__main__':
    net = ablstm(model_config={'in_chans': 48, 'patch_size': 13, 'n_classes': 20})
    net.flops()


