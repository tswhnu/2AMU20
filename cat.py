import torch
import os.path
import numpy as np
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

USE_GPU = True

dtype = torch.float32  # we will be using float throughout this tutorial

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# var_x = 0.05
batch_size = 128
num_train = 50000
num_val = 10000
latent_num = 100

class onehot(object):
    def __call__(self, img):
        image = img * 3
        image = image.long()
        index1 = np.where(image < 1)
        index2 = np.where((image >= 1) & (image < 2))
        index3 = np.where((image >= 2) & (image <= 3))
        image[index1] = 0
        image[index2] = 1
        image[index3] = 2
        image = nn.functional.one_hot(image)
        image = np.squeeze(image)
        return image.permute(2, 0, 1)
# get the data set
transform = transforms.Compose([transforms.ToTensor(), onehot()])
data_train = datasets.MNIST(root="./data/", transform=transform, train=True, download=True)
data_train, data_val = torch.utils.data.random_split(data_train, [num_train, num_val])
data_test = datasets.MNIST(root="./data/", transform=transform, train=False)
data_test, _ = torch.utils.data.random_split(data_test, [32, data_test.__len__() - 32])
data_loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
data_loader_val = torch.utils.data.DataLoader(dataset=data_val, batch_size=batch_size)
data_loader_test = torch.utils.data.DataLoader(dataset=data_test, shuffle=True)




# build the model
# the encoder part

def conv(in_planes, out_planes, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                  bias=False),
        nn.LeakyReLU(0.1, inplace=True),
        nn.BatchNorm2d(out_planes)
    )


def deconv(in_planes, out_planes, kernel_size=2, stride=2):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride),
        nn.ReLU(),
        nn.BatchNorm2d(out_planes)
    )


class Encoder(nn.Module):

    def __init__(self, in_channel=3, latent_num=latent_num):
        super(Encoder, self).__init__()
        self.conv1 = conv(in_channel, 8, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = conv(8, 16, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = conv(16, 32, kernel_size=3)
        self.miu = nn.Linear(32 * 7 * 7, latent_num)
        self.var = nn.Linear(32 * 7 * 7, latent_num)
        # make sure that the var is always positive
        self.var_act = nn.Softplus()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        res = torch.flatten(x, start_dim=1)
        miu = self.miu(res)
        var = self.var_act(self.var(res))

        return miu, var


# the decoder part
class Decoder(nn.Module):
    def __init__(self, num_latent=latent_num, output_channel=3):
        super(Decoder, self).__init__()
        self.num_latent = num_latent
        self.dense = nn.Linear(num_latent, 32 * 7 * 7)
        self.conv1 = conv(32, 16, kernel_size=3)
        self.deconv1 = deconv(16, 16, 2, stride=2)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.sig = nn.Sigmoid()
        self.deconv2 = deconv(8, 8, 2, stride=2)
        self.pi = nn.Conv2d(8, output_channel, kernel_size=3, padding=1)
        self.pi_act1 = nn.ReLU()
        self.pi_act2 = nn.Sigmoid()
    def forward(self, z):
        z = self.dense(z)
        z = z.reshape((-1, 32, 7, 7))
        z = self.conv1(z)
        z = self.deconv1(z)
        z = self.sig(self.conv2(z))
        z = self.deconv2(z)
        pi = self.pi_act2(self.pi_act1(self.pi(z)))

        return pi

    def sample(self, N, convert_to_numpy=False, suppress_noise=True):

        with torch.no_grad():
            z = torch.randn(N, self.num_latent, device=device)
            pi = self.forward(z)
            x = torch.argmax(pi, dim=1)

        if convert_to_numpy:
            z = z.cpu().numpy()
            x = x.cpu().numpy()
        return x, z


# the train function for a single batch
def train(x, encoder, decoder, optimizer):
    optimizer.zero_grad()
    x = x.to(device=device, dtype=dtype)
    encoder.to(device=device)
    decoder.to(device=device)
    miu_z, var_z = encoder(x)
    batch_z = miu_z + torch.sqrt(var_z) * torch.randn(var_z.shape, device=device)

    pi = decoder(batch_z)


    log_p = torch.sum(torch.log(pi**x+0.0001))

    KL = -0.5 * torch.sum(1 + torch.log(var_z) - miu_z ** 2 - var_z)

    neg_elbo = -log_p + KL

    neg_elbo.backward()
    optimizer.step()

    return neg_elbo


def validate(x, encoder, decoder):
    x = x.to(device=device, dtype=dtype)
    encoder.to(device=device)
    decoder.to(device=device)
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        miu_z, var_z = encoder(x)
        batch_z = miu_z + torch.sqrt(var_z) * torch.randn(var_z.shape, device=device)

        pi = decoder(batch_z)
        #

        log_p = torch.sum(torch.log(pi**x+0.0001))

        KL = -0.5 * torch.sum(1 + torch.log(var_z) - miu_z ** 2 - var_z)

        neg_elbo = -log_p + KL

    return neg_elbo


def fit(encoder, decoder, encoder_dir=None, decoder_dir=None, epochs=100, lr=0.0001, save=False):
    if encoder_dir is None or decoder_dir is None:
        print("training new model")
        pass
    else:
        encoder.load_state_dict(torch.load('./test_encoder.pt'))
        decoder.load_state_dict(torch.load('./test_decoder.pt'))

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)

    plt.ion()
    plt.figure(1)
    val_elbo = [0]
    train_elbo = []
    patience = 0

    for e in range(epochs):
        print('training device:', device)
        total_elbo_train = 0

        # train the model
        for t, (x, y) in enumerate(data_loader_train):
            neg_elbo_train = train(x, encoder, decoder, optimizer)
            print('current iteration:', t)
            total_elbo_train += neg_elbo_train.item()
        mean_elbo_train = total_elbo_train / num_train
        train_elbo.append(mean_elbo_train)
        print('mean training neg elbo:', mean_elbo_train)

        # validate the model
        print('begin_validating:')
        total_elbo_val = 0

        for t, (x, y) in enumerate(data_loader_val):
            neg_elbo_val = validate(x, encoder, decoder)
            total_elbo_val += neg_elbo_val

        mean_elbo_val = total_elbo_val / num_val
        print('mean val neg elbo:', mean_elbo_val)
        val_elbo.append(mean_elbo_val)

        # check if the performance on the validation set is getting worse
        if val_elbo[-1] >= val_elbo[-2]:
            patience += 1
        else:
            patience = 0

        if patience == 4:
            print('need to stop')
            break
        # show the plot of the elbo of validation set and training set
        plt.clf()
        plt.plot(train_elbo)
        plt.plot(val_elbo)
        plt.legend(['train_neg_elbo', 'val_neg_elbo'])
        plt.show()

        # task 1d
        if e % 5 == 0:
            x, _ = decoder.sample(1)
            x = np.squeeze(x.cpu())
            x = np.array(x)
            plt.imshow(x, cmap='gray')
            plt.show()
            # plt.imshow(x)
            # plt.show()

    # save the model
    if save:
        torch.save(encoder.state_dict(), 'beta model/encoder_cat.pt')
        torch.save(decoder.state_dict(), 'beta model/decoder_cat.pt')


def perform_test():
    # load the trained model
    encoder = Encoder()
    decoder = Decoder()
    encoder.load_state_dict(torch.load('./encoder_beta.pt'))
    decoder.load_state_dict(torch.load('./decoder_beta.pt'))
    encoder.to(device).eval()
    decoder.to(device).eval()
    image = np.zeros((8 * 28, 8 * 28))

    # make sure not keep track of gradients
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader_test):
            row = i // 4
            colum = i - row * 4
            origin = x.numpy()
            origin = origin / origin.max() - origin.min()
            image[row * 28:row * 28 + 28, colum * 56:colum * 56 + 28] = origin

            x = x.to(device)
            miu_z, var_z = encoder(x)
            z = miu_z + torch.sqrt(var_z) * torch.randn(var_z.shape, device=device)
            re_x, var_x = decoder(z)
            var_x = var_x.mean()
            re_x += torch.sqrt(var_x) * torch.randn(1, device=device)
            re_x = re_x.cpu().numpy()
            re_x = re_x / re_x.max() - re_x.min()
            image[row * 28:row * 28 + 28, colum * 56 + 28:colum * 56 + 56] = re_x
        plt.imshow(image, cmap='gray')
        plt.title('sample_reconstruction')
        plt.show()


def new_samples():
    decoder = Decoder()
    decoder.load_state_dict(torch.load('./decoder_beta.pt'))
    decoder.to(device).eval()
    image = np.zeros((8 * 28, 8 * 28))
    with torch.no_grad():
        x, _ = decoder.sample(64)
        x = np.squeeze(x.cpu().numpy())
        for i in range(64):
            row = i // 8
            colum = i - row * 8
            image[row * 28:row * 28 + 28, colum * 28:colum * 28 + 28] = x[i]
        plt.imshow(image, cmap='gray')
        plt.title('new_samples')
        plt.show()


#
# perform_test()
# new_samples()
encoder = Encoder()
decoder = Decoder()
# fit(encoder, decoder, encoder_dir='./encoder_beta.pt', decoder_dir='./decoder_beta.pt', lr=0.00005, save=True)
fit(encoder, decoder, save=True)
