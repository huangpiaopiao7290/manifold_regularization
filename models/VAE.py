import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# 6. 定义模型（VAE + 分类器）
# ---------------------------

class Encoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)  # [B, 16, 16, 16]
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1) # [B, 32, 8, 8]
        self.fc = nn.Linear(32*8*8, 128)
        self.mu_layer = nn.Linear(128, latent_dim)
        self.logvar_layer = nn.Linear(128, latent_dim)

    def forward(self, x):
        h = F.relu(self.conv1(x))  # [B, 16, 16, 16]
        h = F.relu(self.conv2(h))  # [B, 32, 8, 8]
        h = h.view(h.size(0), -1)  # [B, 32*8*8]
        h = F.relu(self.fc(h))     # [B, 128]
        mu = self.mu_layer(h)      # [B, latent_dim]
        logvar = self.logvar_layer(h)  # [B, latent_dim]
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 32*8*8)
        self.deconv1 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)  # [B,16,16,16]
        self.deconv2 = nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1)   # [B,3,32,32]

    def forward(self, z):
        h = F.relu(self.fc(z))              # [B, 32*8*8]
        h = h.view(h.size(0), 32, 8, 8)     # [B, 32, 8, 8]
        h = F.relu(self.deconv1(h))         # [B, 16, 16, 16]
        x_recon = torch.sigmoid(self.deconv2(h))  # [B, 3, 32, 32]
        return x_recon

class Classifier(nn.Module):
    def __init__(self, latent_dim=32, num_classes=10):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, z):
        h = F.relu(self.fc1(z))  # [B, 64]
        logits = self.fc2(h)     # [B, num_classes]
        return logits

class VAE_Classifier(nn.Module):
    def __init__(self, latent_dim=32, num_classes=10):
        super(VAE_Classifier, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.classifier = Classifier(latent_dim, num_classes)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)    # [B, latent_dim]
        eps = torch.randn_like(std)      # [B, latent_dim]
        return mu + eps * std            # [B, latent_dim]

    def forward_vae(self, x):
        mu, logvar = self.encoder(x)             # [B, latent_dim], [B, latent_dim]
        z = self.reparameterize(mu, logvar)      # [B, latent_dim]
        x_recon = self.decoder(z)                # [B, 3, 32, 32]
        return x_recon, mu, logvar, z

    def forward_classifier(self, x):
        mu, logvar = self.encoder(x)             # [B, latent_dim], [B, latent_dim]
        logits = self.classifier(mu)             # [B, num_classes]
        return logits

# 实例化模型
latent_dim = 32
num_classes = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备类型
model = VAE_Classifier(latent_dim=latent_dim, num_classes=num_classes).to(device)