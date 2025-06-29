# 🧠 Restoring Noisy MNIST Images Using AutoEncoders

This project demonstrates how to denoise handwritten digit images (MNIST dataset) using a lightweight **Convolutional Autoencoder** built with PyTorch.

---

## 🎯 Objective

To train an autoencoder model that can effectively reconstruct clean digit images from noisy versions using convolutional layers.

---

## 📁 Dataset

* **MNIST** (Modified National Institute of Standards and Technology database)
* Downloaded using `torchvision.datasets`
* Shape: 28x28 grayscale images

---

## 🧪 Project Workflow

### ✅ Phase 1: Data Setup & Noise Injection

* Load MNIST train, validation, and test splits using `torchvision`
* Add **Gaussian noise** with a noise factor of 0.25

### ✅ Phase 2: Model Architecture

* Define a **lightweight convolutional autoencoder** with encoder-decoder blocks
* Encoder compresses image to a latent space
* Decoder reconstructs clean image from compressed features

### ✅ Phase 3: Training

* Loss Function: Mean Squared Error (MSE)
* Optimizer: Adam
* Epochs: 20
* Metrics: **Training Loss** and **Validation PSNR (Peak Signal-to-Noise Ratio)**

### ✅ Phase 4: Evaluation

* Visualize noisy, clean (ground truth), and denoised images
* Compare outputs using PSNR

---

## 🔍 Key Components

| Component         | Description                                  |
| ----------------- | -------------------------------------------- |
| 🧪 Noise Function | Gaussian noise added with `torch.randn_like` |
| 🧠 Autoencoder    | 2 Conv2D + 2 ConvTranspose2D layers          |
| 📉 Loss           | MSE Loss for reconstruction error            |
| 📊 PSNR Metric    | Skimage metric for evaluating clarity        |

---

## 📈 Sample Visuals

* Training Loss vs Epochs
* PSNR vs Epochs
* Test Set: Noisy → Clean → Denoised Images

> 🖼️ *Tip: Include screenshots or `.png` samples from your result plots if uploading to GitHub.*

---

## ⚙️ Technologies Used

* Python 3.x
* PyTorch
* TorchVision
* NumPy, Matplotlib
* Skimage (for PSNR)

---

## 🚀 Future Extensions

* Add training on other datasets like Fashion-MNIST or CIFAR-10
* Use deeper or residual autoencoders
* Convert to a Streamlit demo for live denoising
* Explore variational autoencoders (VAEs)

---

## 📁 Project Structure

```
📦 MNIST-Autoencoder-Denoising
├── 📄 README.md
├── 📄 autoencoder_mnist.ipynb
└── 📁 assets/   # (optional) for demo images
```

---

## 🪪 License

**Apache License 2.0**

> This project is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).

---

## 🙌 Acknowledgements

* [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
* [PyTorch Autoencoder Tutorials](https://pytorch.org/tutorials/)
* Skimage PSNR documentation

---

> 💡 *Try tuning noise levels and network depth to explore performance differences.*
