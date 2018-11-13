import torch
import torch.nn as nn
from args import args
import os
from model.D3D import LipReading as Model
from data.dataset import LipreadingDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from util import reload_model, AdjustLR, trn_epoch, tst_epoch

if args.usecuda:
    torch.backends.cudnn.benchmark = True

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
if args.dataset == 'LRW-1000':
    num_classes = 1000
    args.padding = 60
if args.dataset == 'LRW':
    num_classes = 500
    args.padding = 29

model = Model(drop_rate=args.dp, num_classes=num_classes)
print(model)
reload_model(model, path=args.model_path)

if len(args.gpus.split(',')) > 1:
    model = nn.DataParallel(model)
if args.usecuda:
    torch.backends.cudnn.benchmark = True
    model = model.cuda()


if args.opt.lower() == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
if args.opt.lower() == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.001, nesterov=True)

if args.train:
    scheduler = AdjustLR(optimizer, [args.lr], sleep_epochs=1, half=5)
    trn_index = os.path.join(args.index_root, 'trn.txt')
    dataset = LipreadingDataset(data_root=args.data_root, index_root=trn_index, padding=args.padding)
    trn_loader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.num_workers,
                            drop_last=False)
    trn_len = len(trn_loader)
if args.val:
    val_index = os.path.join(args.index_root, 'val.txt')
    dataset = LipreadingDataset(data_root=args.data_root, index_root=val_index, padding=args.padding, augment=False)
    val_loader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            drop_last=False)
    val_len = len(val_loader)
    tst_index = os.path.join(args.index_root, 'tst.txt')
    dataset = LipreadingDataset(data_root=args.data_root, index_root=tst_index, padding=args.padding, augment=False)
    tst_loader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            drop_last=False)
    tst_len = len(tst_loader)


if args.train:
    for epoch in range(args.s_epoch, args.epoch):
        scheduler.step(epoch)
        trn_epoch(model=model, data_loader=trn_loader, optimizer=optimizer, epoch=epoch)
        if args.val:
            val_acc = tst_epoch(model=model, data_loader=val_loader,  epoch=epoch, stage='val')
            tst_acc = tst_epoch(model=model, data_loader=tst_loader,  epoch=epoch, stage='tst')
            if hasattr(model, 'module'):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            state_dict['val_acc'] = val_acc
            state_dict['test_acc'] = tst_acc
        torch.save(state_dict, args.save_path + '/' + str(epoch + 1) + '_.pt')
if args.val:
    val_acc = tst_epoch(model=model, data_loader=val_loader,  epoch=0, stage='val')
    tst_acc = tst_epoch(model=model, data_loader=tst_loader,  epoch=0, stage='tst')

