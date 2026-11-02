# DL Practical – Programs & Viva Quick Notes

Run each notebook from this folder (dataset paths are `dataset/...`).
Packages: `pip install tensorflow pandas scikit-learn matplotlib seaborn pillow scipy`

| No | Notebook | Dataset |
|---|---|---|
| 1 | 01_Study_DL_Packages | – |
| 2 | 02_FFNN_MNIST / 02_FFNN_CIFAR10 | mnist-jpg / cifar-10-img |
| 3 | 03_CNN_MNIST / 03_CNN_CIFAR10 | mnist-jpg / cifar-10-img |
| 4 | 04_Autoencoder_ECG / 04_Autoencoder_CreditCard | ecg.csv / creditcard.csv |
| 5 | 05_CBOW | text paragraph inside notebook |
| 6 | 06_Transfer_Learning_VGG16 | cifar-10-img (5000 train / 1000 test) |

If the lab PC is slow: reduce `epochs=` or the `batch_size=` in `flow_from_directory` (it decides how many images are loaded).
VGG16 offline: keep `vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5` beside the notebook; otherwise it downloads automatically.

## Key viva points
- **Feedforward NN**: data flows one way (input → hidden → output), no loops. Trained with backpropagation + SGD.
- **ReLU** = max(0, x): avoids vanishing gradient. **Softmax**: turns outputs into class probabilities (sum = 1).
- **Categorical cross-entropy**: loss for multi-class one-hot labels. *Sparse* version takes integer labels.
- **SGD** updates weights: w = w − lr × gradient. **Adam** = momentum + adaptive learning rate per weight.
- **Epoch** = one full pass over the training data. **Batch size** = samples per weight update.
- **CNN**: Conv layer (filters/kernels extract features like edges) → Pooling (reduces size, keeps important features) → Flatten → Dense.
- **Dropout**: randomly switches off neurons during training → reduces overfitting.
- **Overfitting**: train accuracy high, validation accuracy low.
- **Autoencoder**: Encoder compresses input to a small latent vector; Decoder reconstructs it. Loss = MSE(input, output).
  Trained on normal data only → anomalies have high reconstruction error → error > threshold ⇒ anomaly.
- **Precision** = TP/(TP+FP), **Recall** = TP/(TP+FN). For fraud, recall matters (don't miss frauds).
- **CBOW**: predicts the middle word from surrounding context words (window = 2 → 4 context words).
  **Skip-gram** is the opposite: predicts context words from the middle word.
  Embedding layer = lookup table of word vectors; these vectors are the word embeddings.
- **Transfer learning**: reuse a model trained on a large dataset (VGG16 on ImageNet, 1000 classes).
  Freeze conv layers (generic features), train new Dense classifier, then fine-tune by unfreezing the last few layers with a small learning rate.
- **VGG16**: 16 weight layers (13 conv + 3 dense), 3×3 filters, ~138M parameters.
