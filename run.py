import numpy as np
import matplotlib.pyplot as plt

y_hat = np.linspace(0.001, 1, 100)
loss = -np.log(y_hat)

plt.plot(y_hat, loss)
plt.title("Cross-Entropy Loss (when y=1)")
plt.xlabel("Predicted Probability")
plt.ylabel("Loss")
plt.grid(True)
plt.show()