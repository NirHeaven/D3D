import torch
import torch.nn as nn
from .modules import _DenseBlock, _Transition
from collections import OrderedDict
import math

class NLLSequenceLoss(nn.Module):
    """
    Custom loss function.
    Returns a loss that is the sum of all losses at each time step.
    """
    def __init__(self):
        super(NLLSequenceLoss, self).__init__()
        self.criterion = nn.NLLLoss(reduction='none')

    def forward(self, input, length, target, every_frame=False):
        loss = []
        transposed = input.transpose(0, 1).contiguous()
        for i in range(transposed.size(0)):
            loss.append(self.criterion(transposed[i,], target).unsqueeze(1))
        loss = torch.cat(loss, 1)
        mask = torch.zeros(loss.size(0), loss.size(1)).float().cuda()

        for i in range(length.size(0)):
            L = min(mask.size(1), length[i])
            if every_frame:
                mask[i, :L] = 1.0
            else:
                mask[i, L-1] = 1.0
        loss = (loss * mask).sum() / mask.sum()
        return loss
        


def _validate(modelOutput, length, labels, every_frame=False):
    labels = labels.cpu()
    averageEnergies = torch.zeros((modelOutput.size(0), modelOutput.size(-1)))
    for i in range(modelOutput.size(0)):
        if every_frame:
            averageEnergies[i] = torch.mean(modelOutput[i, :length[i]], 0)
        else:
            averageEnergies[i] = modelOutput[i, length[i] - 1]

    _, maxindices = torch.max(averageEnergies, 1)
    count = torch.sum(maxindices == labels)
    return count


class LipReading(torch.nn.Module):
    def __init__(self,  growth_rate=32, num_init_features=64, bn_size=4, drop_rate=0.2, num_classes=1000):
        super(LipReading, self).__init__()
        #block_config = (6, 12, 24, 16)
        block_config = (4, 8, 12, 8)

        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(3, num_init_features, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))),
        ]))
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):

            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)

            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm

        self.features.add_module('norm5', nn.BatchNorm3d(num_features))
        self.gru1 = nn.GRU(536*3*3, 256, bidirectional=True, batch_first=True)
        self.gru2 = nn.GRU(512, 256, bidirectional=True, batch_first=True)

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        self.loss = NLLSequenceLoss
        self._initialize_weights()

    def validator_function(self):
        return _validate

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()
        f2 = self.features(x)
        f2 = f2.permute(0, 2, 1, 3, 4).contiguous()
        B, T, C ,H ,W = f2.size()
        f2 = f2.view(B, T, -1)
        f2, _ = self.gru1(f2)
        f2, _ = self.gru2(f2)
        f2 = self.fc(f2).log_softmax(-1)
        return f2

if (__name__ == '__main__'):
    options = {"model": {'numclasses': 32}}
    data = torch.zeros((4, 3, 18, 112, 112))
    m = LipReading()
    # for k, v in m.state_dict().items():
    #     print(k)
    print(m)
    print(m(data).size())

