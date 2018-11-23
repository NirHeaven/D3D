import argparse

parser = argparse.ArgumentParser('DenseNet 3D')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--epoch', type=int, default=999)
parser.add_argument('--s_epoch', type=int, default=0)
parser.add_argument('--padding', type=int, default=60)
parser.add_argument('--interval',  type=int, default=10)

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--dp', type=float, default=0.2)

parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('--opt', type=str, default='adam')
parser.add_argument('--dataset', type=str, default='LRW')
parser.add_argument('--data_root', type=str, default='')
parser.add_argument('--index_root', type=str, default='')
parser.add_argument('--model_path', type=str, default='')
parser.add_argument('--save_path', type=str, default='weights')

parser.add_argument('--load_weights', action='store_true', default=False)
parser.add_argument('--every_frame', action='store_true', default=False)
parser.add_argument('--usecuda', action='store_false', default=True)
parser.add_argument('--no_train', action='store_true', default=False)
parser.add_argument('--no_val', action='store_true', default=False)

args = parser.parse_args()
