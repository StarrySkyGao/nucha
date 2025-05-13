
import torch
import torch.nn as nn


class VQVAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=10, num_embeddings=64, embedding_dim=10):
        super(VQVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        self.embedding = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        self.num_embeddings = num_embeddings

    def quantize(self, z_e):
        # 量化：找到最近的码本向量
        z_e_flat = z_e.view(-1, z_e.size(-1))
        distances = torch.cdist(z_e_flat, self.embedding)
        encoding_indices = torch.argmin(distances, dim=1)
        z_q = self.embedding[encoding_indices].view(z_e.size())
        return z_q, encoding_indices

    def forward(self, x):
        print(x.shape)
        z_e = self.encoder(x)
        print(z_e.shape)
        z_q, indices = self.quantize(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, z_e, z_q

def vqvae_loss(recon_x, x, z_e, z_q, beta=0.25):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    vq_loss = nn.functional.mse_loss(z_q.detach(), z_e)
    commit_loss = beta * nn.functional.mse_loss(z_e, z_q.detach())
    return recon_loss + vq_loss + commit_loss

# 示例
model = VQVAE()
x = torch.randn(32, 784)
recon_x, z_e, z_q = model(x)
loss = vqvae_loss(recon_x, x, z_e, z_q)
print(f"VQ-VAE Loss: {loss.item()}")