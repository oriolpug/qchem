from train_autoencoder import *
import numpy as np
import matplotlib.pyplot as plt

model = Autoencoder()
model.load_state_dict(torch.load('autoencoder_weights.pth'))

random_band = generate(model)
ks = list(np.linspace(-torch.pi, torch.pi, 1000 ))
p_eval = np.polyval(poly_band.T, ks)
plt.plot(ks, p_eval)
plt.show()