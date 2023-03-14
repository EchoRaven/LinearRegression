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


class LinearRegression(nn.Module):
    def __init__(self,
                 dim, #x的维度
                 M=None,
                 S=None,
                 dimUp=None #数据升维
                 ):
        super(LinearRegression, self).__init__()
        # #假设为线性回归
        # Theta = torch.randn([1, dim]) #[1, dim]
        # self.Theta = nn.Parameter(Theta)
        # #偏执
        # bias = torch.randn([1, 1])
        # self.bias = nn.Parameter(bias)
        self.linear = nn.Linear(dim, 1)
        self.M = M
        self.S = S
        self.dimUp = dimUp

    def forward(self, l_input):
        # 输入为 [batch_size, 1, dim]
        # batch_size = l_input.shape[0]
        # Theta = torch.repeat_interleave(self.Theta.unsqueeze(0), repeats=batch_size, dim=0) #[batch_size, 1, dim]
        # #指数
        # mul = torch.bmm(Theta, l_input.transpose(1, 2)).squeeze() # [batch_size]
        # b = torch.repeat_interleave(self.bias, repeats=batch_size, dim=0).squeeze() # [batch_size]
        # res = mul + b # [batch_size]
        # return res
        if self.dimUp is not None:

        if self.M is not None and self.S is not None:
            l_input = (l_input-self.M)/self.S
        return self.linear(l_input)

def Loss(l_opt, l_tgt):
    num = l_opt.shape[0]
    l_tgt.squeeze()
    diff = 0
    for idx in range(num):
        diff += (l_tgt[idx]-l_opt[idx]) * (l_tgt[idx]-l_opt[idx])
    return diff/(2 * num)


def Train(Model, Data, rate, batch_size, epochs, func=optim.SGD):
    optimizer = func(Model.parameters(), lr=rate, weight_decay=0.2)
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
    savePath = "model_SGD_STD_1000_2.pth"
    datas = LinearData(filename_)
    # means, stds = datas.getMeanStd()
    if not os.path.exists(savePath):
        model = LinearRegression(dim=datas.getDim())
        Train(model, datas, rate=1e-2, batch_size=400, epochs=1000)
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
