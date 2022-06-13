import torch
from torch import nn
from torch.nn import functional as F

class vgg_layer(nn.Module):
    def __init__(self, nin, nout):
        super(vgg_layer, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 3, 1, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True)
                )

    def forward(self, input):
        return self.main(input)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, n_channels, height, width):
        super(UnFlatten, self).__init__()
        self.n_channels = n_channels
        self.height = height
        self.width = width

    def forward(self, input):
        return input.view(
            input.size(0),
            self.n_channels,
            self.height,
            self.width)


class BaseVAE(nn.Module):
    """
    Base abstract class for the Variational Autoencoders
    """
    def __init__(self, channels=1, width=28, height=28, z_dim=2):
        """
        Constructor
        Parameters:
            channels - The number of channels for the image
            width - The width of the image in pixels
            height - The height of the image in pixels
            z_dim - The dimension of the latent space
        """
        super(BaseVAE, self).__init__()
        self.channels = channels
        self.width = width
        self.height = height
        self.z_dim = z_dim

    def getNbChannels(self):
        """
        Returns the number of channels of the handled images
        """
        return self.channels

    def getWidth(self):
        """
        Returns the width of the handled images in pixels
        """
        return self.width

    def getHeight(self):
        """
        Returns the height of the handled images in pixels
        """
        return self.height

    def getZDim(self):
        """
        Returns the dimension of the latent space of the VAE
        """
        return self.z_dim

    def flatten(self, x):
        """
        Can be used to flatten the output image. This method will only handle
        images of the original size specified for the network
        """
        return x.view(-1, self.channels * self.height * self.width)

    def unflatten(self, x):
        """
        Can be used to unflatten an image handled by the network. This method
        will only handle images of the original size specified for the network
        """
        return x.view(-1, self.channels, self.height, self.width)




class FCVAE(BaseVAE):
    """
    Fully connected Variational Autoencoder
    """
    def __init__(self, channels=1, width=28, height=28, z_dim=2):
        super(FCVAE, self).__init__(channels, width, height, z_dim)

        self.fc1 = nn.Linear(self.channels * self.width * self.height, 400)
        self.fc21 = nn.Linear(400, self.z_dim)
        self.fc22 = nn.Linear(400, self.z_dim)
        self.fc3 = nn.Linear(self.z_dim, 400)
        self.fc4 = nn.Linear(400, self.channels * self.width * self.height)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(self.flatten(x))
        z = self.reparameterize(mu, logvar)
        return self.unflatten(self.decode(z)), mu, logvar

class ConvVAE(BaseVAE):
    """
    Convolutional Variational Autoencoder
    """
    def __init__(self, channels=3, width=64, height=64, z_dim=128):
        super(ConvVAE, self).__init__(channels, width, height, z_dim)

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(self.channels, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=4, stride=1),
            nn.ReLU())

        self.encoder = nn.Sequential(
            self.encoder_conv,
            Flatten())

        dummy_input = torch.ones([1, self.channels, self.height, self.width])
        conv_size = self.encoder_conv(dummy_input).size()
        h_dim = self.encoder(dummy_input).size(1)

        self.fc1 = nn.Linear(h_dim, self.z_dim)
        self.fc2 = nn.Linear(h_dim, self.z_dim)
        self.fc3 = nn.Linear(self.z_dim, h_dim)

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(1, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.channels, kernel_size=4, stride=1),
            nn.Sigmoid())

        self.decoder = nn.Sequential(
            UnFlatten(conv_size[1], conv_size[2], conv_size[3]),
            self.decoder_conv)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        print(h.shape)
        z, mu, logvar = self.bottleneck(h)
        print(logvar.shape)
        print(mu.shape)
        print(z.shape)
        z = self.fc3(z)
        print(z.shape)
        return self.decoder(z), mu, logvar

class CVAE(BaseVAE):
    """
    Convolutional Variational Autoencoder
    """
    def __init__(self, batch_size ,channels=3, width=64, height=64, z_dim=128):
        super(CVAE, self).__init__(channels, width, height, z_dim)

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(self.channels, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=4, stride=1),
            nn.ReLU())
        self.b = batch_size
        # self.encoder = nn.Sequential(
        #     self.encoder_conv,
        #     Flatten())

        self.encoder = nn.Sequential(
            self.encoder_conv)



        dummy_input = torch.ones([1, self.channels, self.height, self.width])
        conv_size = self.encoder_conv(dummy_input).size()
        h_dim = self.encoder(dummy_input).size(1)

        self.fc1 = nn.Linear(h_dim, self.z_dim)
        self.fc2 = nn.Linear(h_dim, self.z_dim)
        self.fc3 = nn.Linear(self.z_dim, h_dim)

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(1, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.channels, kernel_size=4, stride=1),
            nn.Sigmoid())

        self.decoder = nn.Sequential(
            UnFlatten(conv_size[1], conv_size[2], conv_size[3]),
            self.decoder_conv)

        self.fc_channels = 1
        self.fc_width = 52
        self.fc_height = 52
        self.fc1 = nn.Linear(self.fc_channels * self.fc_width * self.fc_height, 400)
        self.fc21 = nn.Linear(400, self.z_dim)
        self.fc22 = nn.Linear(400, self.z_dim)
        self._fc3 = nn.Linear(self.z_dim, 400)
        self._fc4 = nn.Linear(400, self.fc_channels * self.fc_width * self.fc_height)

    def fc_encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def fc_decode(self, z):
        h3 = F.relu(self._fc3(z))
        return torch.sigmoid(self._fc4(h3))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z
    def fc_reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        
        return mu + eps*std

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        h = h.permute(0,3,2,1)
        temp = list(h.shape)
        h = torch.reshape(h,(temp[0],temp[1]*temp[2]*temp[3]))

        mu, logvar = self.fc_encode(h)
        z = self.fc_reparameterize(mu, logvar)
        z = self.fc_decode(z)
        z = z.view(-1, temp[1], temp[2],temp[3])
        z = z.permute(0,3,2,1)   
 
        return self.decoder(z), mu, logvar
