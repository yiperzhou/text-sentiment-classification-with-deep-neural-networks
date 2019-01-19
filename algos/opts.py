import argparse

parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=50, help='number of epochs for train [default: 256]')
parser.add_argument('-batch_size', type=int, default=32, help='batch size for training [default: 64]')
parser.add_argument('-num_classes', type=int, help="classification number, 10 or 100", default=5)

parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed_dim', type=int, default=300, help='number of embedding dimension [default: 300]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')

parser.add_argument('-model', type=str, help="model folder, like ElasticNN-ResNet50", default="CNN_Text_Model")
# device
parser.add_argument('-device', type=int, default=0, help='device to use for iterate data, 0 mean gpu [default: 0]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=True, help='train or test')
parser.add_argument('-dataset', type=str, default="SST", help='dataset, SST-1, tripadvisor')
parser.add_argument('--manual-seed', default=0, type=int, metavar='N',
                    help='Manual seed (default: 0)')
parser.add_argument('--gpu', default="0", help='gpu available')
args = parser.parse_args()
