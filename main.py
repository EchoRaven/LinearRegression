import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch import optim

class LinearData(Dataset):
    def __init__(self, l_ipt, l_opt):
        super(LinearData, self).__init__()
        self.l_ipt = l_ipt
        self.l_opt = l_opt

    def __len__(self):
        return self.l_ipt.shape[0]

    def __getitem__(self, idx):
        return torch.Tensor(self.l_ipt[idx]), \
               torch.Tensor(self.l_opt[idx])


class LinearRegression(nn.Module):
    def __init__(self,
                 dim, #x的维度
                 ):
        super(LinearRegression, self).__init__()
        #假设为线性回归
        Theta = torch.randn([1, dim]) #[1, dim]
        self.Theta = nn.Parameter(Theta)
        #偏执
        bias = torch.randn([1, 1])
        self.bias = nn.Parameter(bias)

    def forward(self, l_input):
        # 输入为 [batch_size, 1, dim]
        batch_size = l_input.shape[0]
        Theta = torch.repeat_interleave(self.Theta.unsqueeze(0), repeats=batch_size, dim=0) #[batch_size, 1, dim]
        #指数
        mul = torch.bmm(Theta, l_input.transpose(1, 2)).squeeze() # [batch_size]
        b = torch.repeat_interleave(self.bias, repeats=batch_size, dim=0).squeeze() # [batch_size]
        res = mul + b # [batch_size]
        return res

def Loss(l_opt, l_tgt):
    num = l_opt.shape[0]
    l_tgt.squeeze()
    diff = 0
    for idx in range(num):
        diff += (l_tgt[idx]-l_opt[idx]) * (l_tgt[idx]-l_opt[idx])
    return diff/(2 * num)


def Train(Model, Data, rate, batch_size, epochs, func=optim.SGD):
    optimizer = func(Model.parameters(), lr=rate)
    dataLoader = DataLoader(Data, batch_size=batch_size, shuffle=True, drop_last=True)
    for e in range(epochs):
        for idx, batch in enumerate(dataLoader):
            optimizer.zero_grad()
            l_ipt, l_tgt = batch
            output = Model(l_ipt)
            loss = Loss(output, l_tgt)
            loss.backward()
            optimizer.step()
        print(loss)


if __name__ == "__main__":
    model = LinearRegression(dim=3)
    ipt = torch.Tensor([[[1, 2, 3]],
                        [[2, 3, 3]],
                        [[5, 2, 5]]])
    opt = torch.Tensor([[15],
                        [19],
                        [29]])
    datas = LinearData(ipt, opt)
    if not os.path.exists('model.pth'):
        model(ipt)
        Train(model, datas, rate=0.0001, batch_size=2, epochs=5000)
        torch.save(model, 'model.pth')
    else:
        model = torch.load('model.pth')
        print(model(torch.Tensor([[[4, 2, 3]]])))