from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import load_data
import numpy as np


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

batch_size = args.batch_size
num_macro_batch = 1200
size_macro_batch = 1024
sample_size = 100
# batch_size = 128
data_type = 'train'
num_train_batch = num_macro_batch * size_macro_batch / batch_size
train_set = load_data.DataSet(num_macro_batch, size_macro_batch, sample_size, data_type)

num_macro_batch = 50
size_macro_batch = 1024
sample_size = 100
# batch_size = 128
data_type = 'FBL'
num_test_batch = num_macro_batch * size_macro_batch / batch_size
test_set = load_data.DataSet(num_macro_batch, size_macro_batch, sample_size, data_type)

channel_idx = np.arange(32)  # channels to be used


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(3200, 512)
        self.fc21 = nn.Linear(512, 20)
        self.fc22 = nn.Linear(512, 20)
        self.fc3 = nn.Linear(20, 512)
        self.fc4 = nn.Linear(512, 3200)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VAE4Layers(nn.Module):
    def __init__(self):
        super(VAE4Layers, self).__init__()
        # encoder
        self.fc1 = nn.Linear(3200, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc31 = nn.Linear(128, 20)
        self.fc32 = nn.Linear(128, 20)
        # decoder
        self.fc4 = nn.Linear(20, 128)
        self.fc5 = nn.Linear(128, 512)
        self.fc6 = nn.Linear(512, 3200)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc4(z))
        h4 = self.relu(self.fc5(h3))
        return self.sigmoid(self.fc6(h4))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE4Layers()
if args.cuda:
    model.cuda()


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x + 1e-12, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= args.batch_size * 100*32

    return BCE + KLD


optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx in range(num_train_batch):
        data_3d, label = train_set.next_batch(batch_size)
        data_2d = load_data.array_3d_to_2d(load_data.trunc_data(data_3d, channel_idx))
        data = Variable(torch.from_numpy(data_2d)).float()
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), num_train_batch*batch_size,
                100. * batch_idx / num_train_batch,
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.6f}'.format(
          epoch, train_loss / (num_train_batch*batch_size)))


def test(epoch):
    model.eval()
    test_loss = 0
    for i in range(num_test_batch):
        data_3d, label = test_set.next_batch(batch_size)
        data_2d = load_data.array_3d_to_2d(load_data.trunc_data(data_3d, channel_idx))
        data = Variable(torch.from_numpy(data_2d)).float()
        if args.cuda:
            data = data.cuda()
#        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).data[0]
        if i == 0:
          n = min(data.size(0), 8)
          comparison = torch.cat([data.view(args.batch_size, 1, 100, 32)[:n],
                                  recon_batch.view(args.batch_size, 1, 100, 32)[:n]])
          save_image(comparison.data.cpu(),
                     'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= (1.0*num_test_batch*batch_size)
    print('====> Test set loss: {:.6f}'.format(test_loss))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    sample = Variable(torch.randn(64, 20))
    if args.cuda:
       sample = sample.cuda()
    sample = model.decode(sample).cpu()
    save_image(sample.data.view(64, 1, 100, 32),
               'results/sample_' + str(epoch) + '.png')