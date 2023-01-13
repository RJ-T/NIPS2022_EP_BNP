import torch
import torch.nn as nn
import models
from models import get_model
"""
EP/BNP
    Input:
        - sd_ori: original network state dict 
        - k: coefficient that determines the pruning threshold
        - mixed_loader/val_loader: dataset loader for EP/BNP pruning
        - args
        - num_classes: number of classes, used for defining network structure
    Output:
        Pruned model
CLP
    Input:
        - net: model to be pruned
        - u: coefficient that determines the pruning threshold
    Output:
        None (in-place modification on the model)
"""


class BatchNorm2d_gau(nn.BatchNorm2d):
    def __init__(self, num_features):
        super().__init__(num_features)
        self.batch_var = 0
        self.batch_mean = 0

    def forward(self, x):
        self.batch_var = x.var((0, 2, 3))
        self.batch_mean = x.mean((0, 2, 3))
        output = super().forward(x)
        return output


class BatchNorm2d_ent(nn.BatchNorm2d):
    def __init__(self, num_features):
        super().__init__(num_features)
        self.batch_feats = []

    def forward(self, x):
        self.batch_feats = x.reshape(x.shape[0], x.shape[1], -1).max(-1)[0].permute(1, 0).reshape(x.shape[1], -1)
        output = super().forward(x)
        return output


def batch_entropy_2(x, step_size=0.01):
    n_bars = int((x.max()-x.min())/step_size)
    # print(n_bars)
    entropy = 0
    for n in range(n_bars):
        num = ((x > x.min() + n*step_size) * (x < x.min() + (n+1)*step_size)).sum(-1)
        p = torch.true_divide(num, x.shape[-1])
        logp = -p * p.log()
        logp = torch.where(torch.isnan(logp), torch.full_like(logp, 0), logp)
        # p = p.cpu().numpy()
        # print(p)
        # print(logp)
        entropy += logp
    # print(entropy)
    return entropy


def CLP(net, u):
    params = net.state_dict()
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            std = m.running_var.sqrt()
            weight = m.weight

            channel_lips = []
            for idx in range(weight.shape[0]):
                # Combining weights of convolutions and BN
                w = conv.weight[idx].reshape(conv.weight.shape[1], -1) * (weight[idx]/std[idx]).abs()
                channel_lips.append(torch.svd(w.cpu())[1].max())
            channel_lips = torch.Tensor(channel_lips)

            index = torch.where(channel_lips>channel_lips.mean() + u*channel_lips.std())[0]

            params[name+'.weight'][index] = 0
            params[name+'.bias'][index] = 0
            print(name, index)
        
       # Convolutional layer should be followed by a BN layer by default
        elif isinstance(m, nn.Conv2d):
            conv = m

    net.load_state_dict(params)


# This version of EP uses only uses args.batch-size samples in the mixed training dataset for pruning.
# If you want to use the full training dataset for better performance, please comment the break in line 111.
def EP(sd_ori, k, mixed_loader, args, num_classes):
    net = get_model(args.model, num_classes, BatchNorm2d_ent).to(args.device)
    net.load_state_dict(sd_ori)
    net.eval()
    entrp = {}
    batch_feats_total = {}
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            batch_feats_total[name] = torch.tensor([]).cuda()
    with torch.no_grad():
        for i, data in enumerate(mixed_loader):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = net(inputs)
            for name, m in net.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    batch_feats_total[name] = torch.cat([batch_feats_total[name], m.batch_feats], 1)
            break
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            feats = batch_feats_total[name]
            feats = (feats - feats.mean(-1).reshape(-1, 1)) / feats.std(-1).reshape(-1, 1)
            entrp[name] = batch_entropy_2(feats)
    index = {}
    # print(entrp['bn1'].size())
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            entrs = entrp[name]
            idx = torch.where(entrs < (entrs.mean() - k * entrs.std()))
            index[name] = idx

    net_2 = get_model(args.model, num_classes).to(args.device)
    net_2.load_state_dict(sd_ori)

    sd = net_2.state_dict()
    pruned = 0
    for name, m in net_2.named_modules():
        if name in index.keys():
            for idx in index[name]:
                # print(name, idx)
                sd[name + '.weight'][idx] = 0
                pruned += len(idx)
    print(index)
    print('DDE pruned:', pruned)
    net_2.load_state_dict(sd)
    return net_2

# Just set args.batch_size to be the size of val_loader (500 for CIFAR-10)
def BNP(sd_ori, k, validate_loader, args, num_classes):
    net = get_model(args.model, num_classes, BatchNorm2d_gau).to(args.device)
    net.load_state_dict(sd_ori)
    index = {}
    net.eval()
    with torch.no_grad():
        for data in validate_loader:
            inputs, labels = data
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = net(inputs)
            break
            
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            var_2 = m.running_var
            var_1 = m.batch_var
            mean_2 = m.running_mean
            mean_1 = m.batch_mean
            measure = (var_2.sqrt() / var_1.sqrt()).log() + (var_1 + (mean_1 - mean_2).pow(2)) / (2 * var_2) - 1 / 2
            idx = torch.where(measure > measure.mean() + k * measure.std())
            index[name] = idx

    net_2 = get_model(args.model, num_classes).to(args.device)
    net_2.load_state_dict(sd_ori)

    sd = net_2.state_dict()
    pruned = 0

    for name, m in net_2.named_modules():
        if name in index.keys():
            for idx in index[name]:
                sd[name + '.weight'][idx] = 0
                pruned += len(idx)
    print(index)
    print('MBNS pruned:', pruned)
    net_2.load_state_dict(sd)
    return net_2
