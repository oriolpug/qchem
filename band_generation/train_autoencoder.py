###
# Autoencoder training
###

#  Train a variational autoencoder so that then we can sample bands from the latent space

import torch
import torch.utils
import torch.utils.data
import csv
import matplotlib.pyplot as plt

class Autoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Autoencoder with one hidden layer (= latent space)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(20, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8,8)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(4,20),
            torch.nn.ReLU()
        )

        self.mu = torch.Tensor((4,))
        self.sigma = torch.Tensor((4,))
    
    def forward(self,x):    
        encoded = self.encoder(x)
        self.mu, log_var = encoded.split(4, dim=1)
        self.sigma = torch.exp(0.5*log_var)
        e = torch.randn_like(self.sigma)
        z = e * self.sigma + self.mu
        y = self.decoder(z)
        return y

    def generate(self):
        e = torch.randn_like(self.sigma)
        z = e * self.sigma + self.mu
        y = self.decoder(z)
        return y

class myDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        with open(path) as csvfile:
            reader = csv.reader(csvfile)
            data = []
            for row in reader:
                if row:
                    data.append(row)

        # Put everything into a tensor and transpose
        self.data = torch.zeros((len(data),len(data[0])))
        for i in range(len(data)):
            for j in range(len(data[0])):
                self.data[i,j] = float(data[i][j])


    def __getitem__(self, idx):
        return self.data[idx,:]
    
    def __len__(self):
        return len(self.data)

d = myDataset('fake_data.csv')

loader = torch.utils.data.DataLoader(d, batch_size=1000, shuffle=True)
model = Autoencoder()

load_mode = False
if load_mode == True:
    model.load_state_dict(torch.load('autoencoder_weights.pth'))
loss_fcn = torch.nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0)

epochs = 300
outs = []
losses = []

for epoch in range(epochs):
    for instance in loader:

        reconstructed = model(instance)
        loss = loss_fcn(reconstructed, instance)

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.detach())
    outs.append((epochs, instance, reconstructed))

plt.figure()
plt.plot(losses)
plt.xlabel("iteration"); plt.ylabel("loss"); plt.title("autoencoder performance")
plt.show()

# Save model weights
torch.save(model.state_dict(), 'autoencoder_weights.pth')