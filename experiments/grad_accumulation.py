import torch
from torch import nn
from torch.nn import Sequential
from torch.optim import Adam
from torch.nn.functional import l1_loss
from copy import deepcopy

model = Sequential(
    nn.Linear(32, 64, bias=True),
    nn.LeakyReLU(),
    nn.Linear(64, 16),
    nn.Sigmoid(),
)

inputs = torch.randn(32, 32)
targets = torch.randn(32, 16)

def gather_grad(model):
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.clone()
    return grads

def grad_batch(model):
    optimizer = Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = l1_loss(targets, outputs)
    loss.backward()
    return gather_grad(model)

def grad_batch_accum(model, chunck):
    optimizer = Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    assert inputs.shape[0] % chunck == 0
    chunck_size = inputs.shape[0] // chunck
    split_inputs = torch.split(inputs, chunck_size)
    split_targets = torch.split(targets, chunck_size)
    for idx, sin in enumerate(split_inputs):
        split_outputs = model(sin)
        loss = l1_loss(split_outputs, split_targets[idx]) / chunck
        loss.backward()
    return gather_grad(model)

grad_b = grad_batch(deepcopy(model))
grad_ba_8 = grad_batch_accum(deepcopy(model), 8)
grad_ba_16 = grad_batch_accum(deepcopy(model), 16)
grad_ba_32 = grad_batch_accum(deepcopy(model), 32)

for name in grad_b.keys():
    gb = grad_b[name]
    print(name)
    for gba in [grad_ba_8[name], grad_ba_16[name], grad_ba_32[name]]:
        diff = (gb - gba).abs().max()
        print(f'\t{diff.item():.9f}')
