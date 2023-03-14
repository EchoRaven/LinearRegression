import os.path
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch import optim

class LinearData(Dataset):
    def __init__(self, filename):
        super(LinearData, self).__init__()
        df = pd.read_csv(filename)
        self.quailty= df['quality']
        self.data = df.drop(['quality'], axis=1)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.Tensor([self.data.iloc[idx]]), \
               torch.Tensor([self.quailty[idx]])

    def getDim(self):
        return len(self.data.iloc[0])

    def getMeanStd(self):
        Mean = torch.zeros(self.data.iloc[0].shape)
        Std = torch.zeros(self.data.iloc[0].shape)
        data = torch.from_numpy(self.data.values)
        data = data.transpose(0, 1)
        for i in range(len(data)):
            Mean[i] = data[i].mean()
            Std[i] = data[i].std()
        return Mean, Std


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout = 0.1):
        super(PoswiseFeedForwardNet, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        residual = inputs # inputs : [batch_size, len_q, d_model]
        return residual + self.w_2(self.dropout(F.relu(self.w_1(inputs))))


class NerualRegression(nn.Module):
    def __init__(self,
                 d_model, #x的维度
                 d_ff, #linar层的维度
                 dropout=0.1,
                 n_layers=1,
                 M=None,
                 S=None
                 ):
        super(NerualRegression, self).__init__()
        self.M = M
        self.S = S
        self.linear = nn.Linear(d_model, 1)
        self.layers = nn.ModuleList([PoswiseFeedForwardNet(d_model=d_model,
                                                           d_ff=d_ff,
                                                           dropout=dropout)
                                     for _ in range(n_layers)])

    def forward(self, l_input):
        l_input = (l_input - self.M) / self.S
        enc_outputs = l_input
        for layer in self.layers:
            enc_outputs = layer(enc_outputs)
        return self.linear(enc_outputs)

def Loss(l_opt, l_tgt):
    num = l_opt.shape[0]
    l_tgt.squeeze()
    diff = 0
    for idx in range(num):
        diff += (l_tgt[idx]-l_opt[idx]) * (l_tgt[idx]-l_opt[idx])
    return diff/(2 * num)


def Train(Model, Data, rate, batch_size, epochs, func=optim.SGD):
    optimizer = func(Model.parameters(), lr=rate, weight_decay=0.1)
    dataLoader = DataLoader(Data, batch_size=batch_size, shuffle=True, drop_last=True)
    for e in range(epochs):
        for idx, batch in enumerate(dataLoader):
            optimizer.zero_grad()
            l_ipt, l_tgt = batch
            opt = Model(l_ipt)
            loss = Loss(opt, l_tgt)
            loss.backward()
            optimizer.step()
        if e % 10 == 0:
            print('Epoch:', '%04d' % e, 'cost =', '{:.6f}'.format(float(loss)))


if __name__ == "__main__":
    filename_ = "winequality-red.csv"
    savePath = "model_neural_deal.pth"
    datas = LinearData(filename_)
    mean, std = datas.getMeanStd()
    if not os.path.exists(savePath):
        model = NerualRegression(d_model=datas.getDim(),
                                 d_ff=16,
                                 dropout=0.1,
                                 n_layers=3,
                                 M=mean,
                                 S=std
                                 )
        Train(model, datas, rate=1e-3, batch_size=500, epochs=20000)
        torch.save(model, savePath)
    else:
        model = torch.load(savePath)
        dataLoader = DataLoader(dataset=datas, batch_size=1, shuffle=False)
        diff = 0
        num = 0
        for idx, batch in enumerate(dataLoader):
            num += 1
            ipt, tgt = batch
            print(float(model(ipt)), float(tgt))
            diff += abs(model(ipt)-tgt)
        print("diff : ", float(diff/num))
