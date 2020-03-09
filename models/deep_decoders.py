import sys
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from fastai.text.qrnn.qrnn import QRNN
from tqdm import tqdm
# Just for debugging, later delete
import pdb


# ========== Deep Learning based Decoders ==========
# ===== RNN related =====
class StackedRNN(nn.Module):
    """
    A class for stacked RNN network.
    Can be either LSTM or GRU
    """

    def __init__(self, args):
        """
        Defining a constructor and initializing the network.
        """
        super(StackedRNN, self).__init__()

        self.args = args
        # For RNNs, to be consistent with Tensorflow, [batch, sequence, features] so batch_first=True
        if args.decode_type == 'LSTM':
            self.rnn1 = nn.LSTM(args.num_chan_eeg,
                                args.rnn_num_hidden, batch_first=True)
            self.rnn2 = nn.LSTM(args.rnn_num_hidden,
                                args.rnn_num_hidden, batch_first=True)
        elif args.decode_type == 'GRU':
            self.rnn1 = nn.GRU(args.num_chan_eeg,
                               args.rnn_num_hidden, batch_first=True)
            self.rnn2 = nn.GRU(args.rnn_num_hidden,
                               args.rnn_num_hidden, batch_first=True)
        elif args.decode_type == 'QRNN':
            self.rnn1 = QRNN(args.num_chan_eeg, args.rnn_num_hidden)
            self.rnn2 = QRNN(args.rnn_num_hidden, args.rnn_num_hidden)
        self.dropout1 = nn.Dropout(args.dropout_rate1)
        self.dropout2 = nn.Dropout(args.dropout_rate2)
        # bias=False for Linear/Conv2d when using BatchNorm?
        self.fc = nn.Linear(args.rnn_num_hidden, args.num_chan_kin)
        self.fc_qrnn = nn.Linear(
            args.tap_size*args.rnn_num_hidden, args.num_chan_kin)

        for m in self.modules():
            if isinstance(m, (nn.LSTM, nn.GRU)):
                for param in m.parameters():
                    ih = (param.data for name,
                          param in self.named_parameters() if 'weight_ih' in name)
                    hh = (param.data for name,
                          param in self.named_parameters() if 'weight_hh' in name)
                    b = (param.data for name,
                         param in self.named_parameters() if 'bias' in name)
                    for t in ih:
                        nn.init.xavier_uniform_(t)
                    for t in hh:
                        nn.init.orthogonal_(t)
                    for t in b:
                        t.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # pdb.set_trace()
                m.bias.data.zero_()

    def forward(self, input, hidden):
        """
        Defining a network architecture for forward pass.
        :param input: dimension: [num_samples x tap_size x num_features]
                      since batch_first=True, this is the right configuration where you could create a batch from
                      num_samples in the first dimension. "tap_size" means length of time samples.
        :return s: forward architecture itself. To check, just print it out.
        """
        # Since QRNN doesn't have batch_first yet, we need to do this
        if self.args.decode_type == "QRNN":
            input = torch.transpose(input, 0, 1)
        # Defining the architecture
        # 1st layer
        s, hidden = self.rnn1(input, hidden)
        s = self.dropout1(s)
        for i in range(self.args.rnn_num_stack_layers-1):
            s, hidden = self.rnn2(s, hidden)
            s = self.dropout2(s)

        # 1) Whether to consider all the outputs from hidden
        # s = s.view(-1, self.args.num_hidden)
        # s = self.fc(s)
        # 2) Or simply extract the output from the last hidden only
        # Ref: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/recurrent_neural_network/main.py

        # Just putting things back for QRNN
        if self.args.decode_type == "QRNN":
            s = torch.transpose(s, 0, 1)
            # Flatten it out to use all the output
            s = s.contiguous().view(-1, self.args.tap_size * self.args.rnn_num_hidden)
            s = self.dropout2(s)
            s = self.fc_qrnn(s)
        else:
            s = self.fc(s[:, -1, :])

        return s, hidden

    def init_hidden(self, bsize):
        """Initialization of the hidden units

        Arguments:
            bsize: batch size. The reason why you need this is when testing.
                   When testing, instead of batch, you feed in a whole dataset at once.
                   Thus exceeding the default batch size stored in argparse.
        """

        weight = next(self.parameters()).data
        if self.args.decode_type == 'LSTM':
            return (weight.new(1,
                               bsize,
                               self.args.rnn_num_hidden).normal_(mean=self.args.init_mean,
                                                                 std=self.args.init_std),
                    weight.new(1,
                               bsize,
                               self.args.rnn_num_hidden).normal_(mean=self.args.init_mean,
                                                                 std=self.args.init_std))
        elif self.args.decode_type == 'GRU':
            return weight.new(1,
                              bsize,
                              self.args.rnn_num_hidden).normal_(mean=self.args.init_mean,
                                                                std=self.args.init_std)
        elif self.args.decode_type == 'QRNN':
            return weight.new(bsize, self.args.rnn_num_hidden).normal_(mean=0, std=0.05)
        else:
            raise ValueError("There's no RNN type you specified.")


# ===== TCN related =====
# Building Temporal Convolutional Networks (TCN)
# Code from https://github.com/locuslab/TCN
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(
            n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Defining a TCN based on the above class


class TCN(nn.Module):
    def __init__(self, args):
        super(TCN, self).__init__()
        num_channels = [args.tcn_num_hidden] * args.tcn_num_layers
        self.tcn = TemporalConvNet(num_inputs=args.tap_size,
                                   num_channels=num_channels,
                                   kernel_size=args.tcn_kernel_size,
                                   dropout=args.tcn_dropout
                                   )
        self.fc = nn.Linear(num_channels[-1], args.num_chan_kin)

    def forward(self, inputs):
        s = self.tcn(inputs)
        s = self.fc(s[:, :, -1])

        return s
