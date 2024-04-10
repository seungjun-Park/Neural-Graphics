import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import rfftn, ifftn


# Frequency Distance loss
def FD(target, pred, dim=2, type='l2'):
    type = type.lower()
    assert type in ['l1', 'l2']
    dim = [-i for i in range(dim, 0, -1)]
    target_freq = rfftn(target, dim=dim)
    pred_freq = rfftn(target, dim=dim)

    # general form of frequency distance
    fd = (target_freq - pred_freq).abs()
    if type == 'l2':
        fd = torch.square(fd)
    fd = torch.mean(fd, dim=[-2, -1])
    fd = torch.sum(fd, dim=[-1])
    fd = torch.sum(fd) / fd.shape[0]
    return fd


# Log Frequency Distance loss
def LFD(target, pred, dim=2, type='l1'):
    fd = FD(target, pred, dim=dim, type=type)
    lfd = torch.log(fd + 1)

    return lfd


def bdcn_loss2(inputs, targets, l_weight=1.1):
    # bdcn loss with the rcf approach
    targets = targets.long()
    # mask = (targets > 0.1).float()
    mask = targets.float()
    num_positive = torch.sum((mask > 0.0).float()).float()  # >0.1
    num_negative = torch.sum((mask <= 0.0).float()).float()  # <= 0.1

    mask[mask > 0.] = 1.0 * num_negative / (num_positive + num_negative)  # 0.1
    mask[mask <= 0.] = 1.1 * num_positive / (num_positive + num_negative)  # before mask[mask <= 0.1]
    # mask[mask == 2] = 0
    inputs = torch.sigmoid(inputs)
    cost = torch.nn.BCELoss(mask, reduction='none')(inputs, targets.float())
    # cost = torch.mean(cost.float().mean((1, 2, 3))) # before sum
    cost = torch.sum(cost.float().mean((1, 2, 3)))  # before sum
    return l_weight * cost

# ------------ cats losses ----------


def bdrloss(prediction, label, radius):
    '''
    The boundary tracing loss that handles the confusing pixels.
    '''

    filt = torch.ones(1, 1, 2*radius+1, 2*radius+1).to(prediction.device)
    filt.requires_grad = False

    bdr_pred = prediction * label
    pred_bdr_sum = label * F.conv2d(bdr_pred, filt, bias=None, stride=1, padding=radius)
    texture_mask = F.conv2d(label.float(), filt, bias=None, stride=1, padding=radius)
    mask = (texture_mask != 0).float()
    mask[label == 1] = 0
    pred_texture_sum = F.conv2d(prediction * (1-label) * mask, filt, bias=None, stride=1, padding=radius)

    softmax_map = torch.clamp(pred_bdr_sum / (pred_texture_sum + pred_bdr_sum + 1e-10), 1e-10, 1 - 1e-10)
    cost = -label * torch.log(softmax_map)
    cost[label == 0] = 0

    return cost.sum()


def textureloss(prediction, label, mask_radius):
    '''
    The texture suppression loss that smooths the texture regions.
    '''
    filt1 = torch.ones(1, 1, 3, 3).to(prediction.device)
    filt1.requires_grad = False
    filt2 = torch.ones(1, 1, 2*mask_radius+1, 2*mask_radius+1).to(prediction.device)
    filt2.requires_grad = False

    pred_sums = F.conv2d(prediction.float(), filt1, bias=None, stride=1, padding=1)
    label_sums = F.conv2d(label.float(), filt2, bias=None, stride=1, padding=mask_radius)

    mask = 1 - torch.gt(label_sums, 0).float()

    loss = -torch.log(torch.clamp(1-pred_sums/9, 1e-10, 1-1e-10))
    loss[mask == 0] = 0

    return torch.sum(loss)


def cats_loss(prediction, label, weights=(1., 0., 0.)):
    # tracingLoss
    cost_weight, tex_factor, bdr_factor = weights
    balanced_w = 1.1
    label = label.float()
    prediction = prediction.float()
    with torch.no_grad():
        mask = label.clone()

        num_positive = torch.sum((mask == 1).float()).float()
        num_negative = torch.sum((mask == 0).float()).float()
        beta = num_negative / (num_positive + num_negative)
        mask[mask == 1] = beta
        mask[mask == 0] = balanced_w * (1 - beta)
        mask[mask == 2] = 0
    prediction = torch.sigmoid(prediction)
    # print('bce')
    cost = torch.sum(torch.nn.functional.binary_cross_entropy(
        prediction.float(), label.float(), weight=mask, reduce=False))
    label_w = (label != 0).float()
    # print('tex')
    textcost = textureloss(prediction.float(), label_w.float(), mask_radius=4)
    bdrcost = bdrloss(prediction.float(), label_w.float(), radius=4)

    return cost_weight * (cost + bdr_factor * bdrcost + tex_factor * textcost)


SHIFT = torch.Tensor([-.030, -.088, -.188])[None, :, None, None]
SCALE = torch.Tensor([.458, .448, .450])[None, :, None, None]


def scaling(x: torch.Tensor):
    shift = SHIFT.to(x.device)
    scale = SCALE.to(x.device)
    return (x - shift) / scale


def contents_loss(net, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

    in0_input, in1_input = (scaling(inputs), scaling(targets))
    feats0, feats1 = net(in0_input), net(in1_input)
    diffs = []

    for kk in range(5):
        diffs.append(torch.mean((feats0[kk] - feats1[kk]) ** 2, dim=[2, 3]))

    val = torch.cat(diffs, dim=1)
    val = torch.sum(val, dim=1)
    val = torch.sum(val) / val.shape[0]

    return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale