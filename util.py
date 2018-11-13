import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict
import time
import math
from torch.autograd import Variable
from args import args
class AdjustLR(object):
    def __init__(self, optimizer, init_lr, sleep_epochs=5, half=5):
        super(AdjustLR, self).__init__()
        self.optimizer = optimizer
        self.sleep_epochs = sleep_epochs
        self.half = half
        self.init_lr = init_lr

    def step(self, epoch):
        if epoch >= self.sleep_epochs:
            for idx, param_group in enumerate(self.optimizer.param_groups):
                new_lr = self.init_lr[idx] * math.pow(0.5, (epoch-self.sleep_epochs+1)/float(self.half))
                param_group['lr'] = new_lr
                print('>>> reduce learning rate <<<: {}'.format(new_lr))


def reload_model(model, path=""):
    if not bool(path):
        print('train from scratch')
        return
    own_state = model.state_dict()
    state_dict = OrderedDict()
    for p in path.split(','):
        state_dict.update(torch.load(p))
    for name, param in state_dict.items():
        name = name.split('.')
        if name[0] == 'module':
            name = name[1:]
        name = '.'.join(name)
        if name == 'val_acc':
            print('current val acc: {}'.format(param))
            continue
        if name == 'test_acc':
            print('current test acc: {}'.format(param))
            continue
        if name not in own_state:
            print('layer {} skip, not exist in model'.format(name))
            continue
        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        if own_state[name].shape != param.shape:
            print('layer {} skip, shape not same in model'.format(name))
            continue
        own_state[name].copy_(param)
        print('----load param: {}----'.format(name))

def trn_epoch(model, data_loader, optimizer, epoch):
    if hasattr(model, 'module'):
        criterion = model.module.loss()
        validator = model.module.validator_function()
    else:
        criterion = model.loss()
        validator = model.validator_function()
    model.train()
    data_len = len(data_loader)
    n_samples = len(data_loader.dataset)
    running_loss, running_corrects, running_all = 0., 0., 0.
    for i_batch, sample_batch in enumerate(data_loader):
        inputs = Variable(sample_batch['temporalvolume'])
        labels = Variable(sample_batch['label']).long()
        length = Variable(sample_batch['length'])
        if args.usecuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
            length = length.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, length, labels, every_frame=args.every_frame)
        loss.backward()
        optimizer.step()
        running_loss += loss.data * inputs.size(0)
        batch_correct = validator(outputs, length, labels, every_frame=args.every_frame)
        running_corrects += batch_correct
        running_all += len(inputs)
        if i_batch == 0:
            since = time.time()
        elif i_batch % args.interval == 0 or (i_batch == data_len - 1):
            print(
                'Process: [{:5.0f}/{:5.0f} ({:.0f}%)]\tLoss batch: {:.4f}\tLoss total: {:.4f}\tAcc batch:{:.4f}\tAcc total:{:.4f}\tCost time:{:5.0f}s\tEstimated time:{:5.0f}s\r'.format(
                    running_all,
                    n_samples,
                    100. * i_batch / (data_len - 1),
                    float(loss),
                    float(running_loss) / running_all,
                    float(batch_correct) / len(inputs),
                    float(running_corrects) / running_all,
                    time.time() - since,
                    (time.time() - since) * (data_len - 1) / i_batch - (time.time() - since))),
    print('{} Epoch:\t{:2}\tLoss: {:.4f}\tAcc:{:.4f}'.format(
        'train',
        epoch,
        float(running_loss) / n_samples,
        float(running_corrects) / n_samples) + '\n')

def tst_epoch(model, data_loader, epoch, stage):
    model.eval()
    if hasattr(model, 'module'):
        validator = model.module.validator_function()
    else:
        validator = model.validator_function()
    data_len = len(data_loader)
    n_samples = len(data_loader.dataset)
    running_corrects, running_all = 0., 0.
    with torch.no_grad():
        for i_batch, sample_batch in enumerate(data_loader):
            inputs = Variable(sample_batch['temporalvolume'])
            labels = Variable(sample_batch['label']).long()
            length = Variable(sample_batch['length'])
            if args.usecuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
                length = length.cuda()

            outputs = model(inputs)
            batch_correct = validator(outputs, length, labels, every_frame=args.every_frame)
            running_corrects += batch_correct
            running_all += len(inputs)
            if i_batch == 0:
                since = time.time()
            elif i_batch % args.interval == 0 or (i_batch == data_len - 1):
                print(
                    'Process: [{:5.0f}/{:5.0f} ({:.0f}%)]\tAcc batch:{:.4f}\tAcc total:{:.4f}\tCost time:{:5.0f}s\tEstimated time:{:5.0f}s\r'.format(
                        running_all,
                        n_samples,
                        100. * i_batch / (data_len - 1),
                        float(batch_correct) / len(inputs),
                        float(running_corrects) / running_all,
                        time.time() - since,
                        (time.time() - since) * (data_len - 1) / i_batch - (time.time() - since))),
        acc = float(running_corrects) / n_samples
        print('{} Epoch:\t{:2}\tAcc:{:.4f}'.format(
            stage,
            epoch,
            acc + '\n'))
        return acc
