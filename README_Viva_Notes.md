# DL Practical – Viva Notes

Run notebooks from this folder (datasets in `dataset/`).
Packages: `tensorflow pandas scikit-learn matplotlib seaborn pillow scipy`

| No | Notebook | Dataset |
|---|---|---|
| 1 | 01_Study_DL_Packages | – |
| 2 | 02_FFNN_MNIST / 02_FFNN_CIFAR10 | mnist-jpg / cifar-10-img |
| 3 | 03_CNN_MNIST / 03_CNN_CIFAR10 | mnist-jpg / cifar-10-img |
| 4 | 04_Autoencoder_ECG / 04_Autoencoder_CreditCard | ecg.csv / creditcard.csv |
| 5 | 05_CBOW | paragraph inside notebook |
| 6 | 06_Transfer_Learning_VGG16 | cifar-10-img (5000 / 1000) |

Results obtained: FFNN MNIST 96.6%, FFNN CIFAR 47%, CNN MNIST 99.1%, CNN CIFAR 77.6%, ECG AE 95.5%, VGG16 59%.

---

## Important Formulas

| Concept | Formula |
|---|---|
| Neuron output | y = f(Σ wᵢxᵢ + b) |
| ReLU | f(x) = max(0, x) |
| Sigmoid | σ(x) = 1 / (1 + e⁻ˣ) → output 0 to 1 |
| Tanh | (eˣ − e⁻ˣ)/(eˣ + e⁻ˣ) → output −1 to 1 |
| Softmax | P(i) = e^{zᵢ} / Σⱼ e^{zⱼ} (sum of outputs = 1) |
| MSE | (1/n) Σ (y − ŷ)² |
| Binary cross-entropy | −[y log ŷ + (1−y) log(1−ŷ)] |
| Categorical cross-entropy | −Σ yᵢ log(ŷᵢ) |
| Gradient descent update | w = w − η · ∂L/∂w (η = learning rate) |
| SGD with momentum | v = βv − η∇L ; w = w + v |
| Conv output size | ((N − F + 2P) / S) + 1 (N=input, F=filter, P=padding, S=stride) |
| Conv layer params | (F × F × C_in + 1) × C_out |
| Dense layer params | (inputs + 1) × outputs |
| Accuracy | (TP + TN) / (TP + TN + FP + FN) |
| Precision | TP / (TP + FP) |
| Recall (Sensitivity) | TP / (TP + FN) |
| F1-score | 2 × P × R / (P + R) |
| Reconstruction error | mean((x − x̂)²) |
| Normalization (pixels) | x / 255 → range 0–1 |
| MinMax scaling | (x − min) / (max − min) |
| Standard scaling | (x − mean) / std |

**Param example:** Conv2D(32, 3×3) on grayscale → (3·3·1 + 1)·32 = **320**. Dense(128) after 784 inputs → (784+1)·128 = **100,480**.
**Output size example:** 28×28 input, 3×3 filter, no padding, stride 1 → 26×26.

---

## General Concepts

**Q. What is deep learning?**
Subset of ML using neural networks with many hidden layers that learn features automatically from data.

**Q. ML vs DL?**
ML needs manual feature extraction and works with less data; DL learns features itself, needs more data and GPU.

**Q. What is a perceptron?**
Single neuron: weighted sum of inputs + bias, passed through an activation function.

**Q. Why activation functions?**
To add non-linearity; without them the whole network acts like one linear layer.

**Q. Why ReLU in hidden layers?**
Fast to compute and avoids vanishing gradient.

**Q. Softmax vs Sigmoid?**
Softmax → multi-class (probabilities sum to 1). Sigmoid → binary / multi-label.

**Q. What is backpropagation?**
Computing gradient of loss w.r.t. every weight using the chain rule, from output layer back to input.

**Q. Vanishing gradient?**
Gradients become very small in deep networks (sigmoid/tanh), so early layers stop learning. Fix: ReLU, BatchNorm, residual connections.

**Q. Epoch, batch, iteration?**
Epoch = one pass over full data. Batch = samples per update. Iterations per epoch = samples / batch size.

**Q. Learning rate?**
Step size of weight update. Too high → diverges; too low → slow training.

**Q. Optimizers?**
SGD (simple), Momentum (adds velocity), RMSprop (adaptive lr), Adam (momentum + RMSprop, most common).

**Q. Overfitting / underfitting?**
Overfitting: high train acc, low test acc. Underfitting: both low.
Fix overfitting: dropout, more data, augmentation, early stopping, L2 regularization.

**Q. Dropout?**
Randomly sets a fraction of neurons to 0 during training only; prevents co-dependence → less overfitting.

**Q. Batch normalization?**
Normalizes layer outputs per batch → faster, stable training.

**Q. Why split train/test (validation) data?**
To check performance on unseen data (generalization).

**Q. One-hot encoding?**
Label 3 → [0,0,0,1,0,0,0,0,0,0]. Used with categorical_crossentropy. Integer labels use sparse_categorical_crossentropy.

**Q. Why normalize pixels / scale features?**
Same range → faster and stable convergence.

---

## Practical 1 – Packages

**Q. TensorFlow?** Google's open-source DL library; works with tensors; supports CPU/GPU/TPU; deploy with TF Lite / TF Serving.
**Q. Keras?** High-level API (now inside TensorFlow): `Sequential`, `Dense`, `compile()`, `fit()`, `evaluate()`, `predict()`.
**Q. PyTorch?** By Meta; dynamic computation graph; popular in research; `torch.tensor`, `autograd`, `nn.Module`.
**Q. Theano?** Old library from Univ. of Montreal; symbolic math; discontinued (2017).
**Q. Tensor?** Multi-dimensional array (scalar = 0D, vector = 1D, matrix = 2D, image batch = 4D).
**Q. Static vs dynamic graph?** Static (old TF1): define then run. Dynamic (PyTorch, TF2 eager): built while running, easy debugging.
**Q. GradientTape?** TF tool to record operations and compute gradients automatically.

---

## Practical 2 – Feedforward Neural Network

**Q. What is FFNN?** Network where information moves only forward: input → hidden → output, no loops/cycles. Also called MLP.
**Q. Architecture used?** Flatten(784) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax).
**Q. Why Flatten?** Dense layer needs 1D input; 28×28 image → 784 vector.
**Q. Why 10 output neurons?** 10 classes (digits 0–9 / 10 CIFAR objects).
**Q. Why SGD?** Asked in problem statement; updates weights using gradient of mini-batches.
**Q. Why low accuracy on CIFAR-10 (47%)?** FFNN ignores spatial structure of colour images; CNN is better.
**Q. MNIST?** 70,000 handwritten digit images, 28×28 grayscale, 10 classes (60k train / 10k test).
**Q. CIFAR-10?** 60,000 colour images 32×32×3, 10 classes (50k train / 10k test).
**Q. `compile()` has?** optimizer, loss, metrics.
**Q. What does `fit()` return?** History object – loss & accuracy per epoch (used for plotting).

---

## Practical 3 – CNN

**Q. CNN?** Network with convolution layers that extract spatial features (edges → shapes → objects) from images.
**Q. Layers of CNN?** Convolution → Activation (ReLU) → Pooling → Flatten → Fully connected → Softmax.
**Q. Convolution?** Sliding a filter (kernel) over image and taking dot product → feature map.
**Q. Filter/kernel?** Small weight matrix (e.g., 3×3) learned during training; detects a feature.
**Q. Stride?** Number of pixels the filter moves each step.
**Q. Padding?** Adding zeros around image. `valid` = no padding (size shrinks), `same` = output size same as input.
**Q. Pooling?** Downsampling. Max pooling takes max in 2×2 window → halves size, reduces params, gives translation invariance.
**Q. CNN vs FFNN for images?** CNN shares weights (fewer params), keeps spatial info → higher accuracy (MNIST 99% vs 96.6%).
**Q. Feature map?** Output of a convolution filter.
**Q. Famous CNNs?** LeNet, AlexNet, VGG16, GoogLeNet/Inception, ResNet, MobileNet.
**Q. Data augmentation?** Creating more training images by rotate/flip/zoom → reduces overfitting.

---

## Practical 4 – Autoencoder Anomaly Detection

**Q. Autoencoder?** Unsupervised network that learns to reproduce its input: Encoder → Latent (bottleneck) → Decoder.
**Q. Encoder?** Compresses input to small latent vector (ECG: 140 → 32 → 16 → 8).
**Q. Decoder?** Reconstructs input from latent vector (8 → 16 → 32 → 140).
**Q. Latent space / bottleneck?** Compressed representation holding the most important features.
**Q. Loss used?** MSE between input and reconstructed output. Input = target (`fit(x, x)`).
**Q. How anomaly is detected?** Train only on normal data → normal reconstructed well (low error); anomaly gives high error. Error > threshold ⇒ anomaly.
**Q. Threshold?** ECG: mean + 1 std of train errors. Credit card: 95th percentile of errors.
**Q. Why train on normal data only?** So model never learns to reconstruct anomalies.
**Q. Autoencoder vs PCA?** PCA is linear; autoencoder can learn non-linear compression.
**Q. Types?** Undercomplete, Denoising, Sparse, Variational (VAE), Convolutional.
**Q. Applications?** Anomaly/fraud detection, denoising, dimensionality reduction, image compression.
**Q. ECG dataset?** 4998 heartbeats × 140 values; label 1 = normal, 0 = abnormal.
**Q. Credit card dataset?** 284,807 transactions, V1–V28 (PCA features) + Amount; only 492 frauds (0.17%) → highly imbalanced.
**Q. Why accuracy is misleading for fraud?** Predicting "all normal" gives 99.8% accuracy. Use recall, precision, F1, ROC-AUC.
**Q. Why linear/sigmoid in last layer?** Sigmoid when data scaled 0–1 (ECG MinMax); linear when standardized (credit card).

---

## Practical 5 – CBOW

**Q. Word embedding?** Representing words as dense vectors so similar words have similar vectors.
**Q. Word2Vec?** Shallow network (Google, 2013) to learn embeddings. Two models: CBOW and Skip-gram.
**Q. CBOW?** Predicts **target word from context words**. Context "deep ___ is part" → "learning".
**Q. Skip-gram?** Predicts **context words from target word**. Better for rare words; CBOW faster.
**Q. Window size?** Number of words on each side used as context (we used 2 → 4 context words).
**Q. Steps?** Clean text → build vocabulary (word→index) → create (context, target) pairs → Embedding → average → Dense softmax → train.
**Q. Embedding layer?** Lookup table of shape (vocab_size × embedding_dim); its weights are the word vectors.
**Q. Why average context vectors?** CBOW treats context as a "bag" – order doesn't matter.
**Q. Loss?** sparse_categorical_crossentropy (target is a word index).
**Q. Why PCA?** To reduce 10-D embeddings to 2-D for plotting.
**Q. One-hot vs embedding?** One-hot is huge & sparse, no meaning; embedding is small, dense, captures similarity.
**Q. Other embeddings?** GloVe, FastText, BERT (contextual).

---

## Practical 6 – Transfer Learning (VGG16)

**Q. Transfer learning?** Using a model pre-trained on a large dataset as a starting point for a new task.
**Q. Why?** Less data needed, faster training, better accuracy.
**Q. Steps?** Load pre-trained model (include_top=False) → freeze conv layers → add new classifier → train → fine-tune (unfreeze last layers, small lr).
**Q. `include_top=False`?** Removes VGG16's original 1000-class Dense layers so we can add our own.
**Q. Why freeze layers?** Lower layers learned general features (edges, textures); keep them, avoid destroying weights.
**Q. Fine-tuning?** Unfreezing top few conv layers and retraining with a very small learning rate (1e-5).
**Q. Why small learning rate in fine-tuning?** Large updates would destroy pre-trained weights.
**Q. VGG16?** By Oxford (Visual Geometry Group), 2014; 16 weight layers (13 conv + 3 dense); all 3×3 filters; ~138M params; input 224×224.
**Q. ImageNet?** 14 million+ images, 1000 classes; used to pre-train models.
**Q. Feature extraction vs fine-tuning?** Feature extraction: all base layers frozen. Fine-tuning: some base layers also trained.
**Q. Why only 59% accuracy?** CIFAR images are 32×32 (VGG trained on 224×224), only 5000 images, CPU training.
**Q. Image classification vs object detection?** Classification: what is in the image. Detection: what + where (bounding box) – e.g., YOLO, SSD, Faster R-CNN.
**Q. Other pre-trained models?** ResNet50, InceptionV3, MobileNet, EfficientNet, DenseNet.

---

## Common Keras Functions

| Function | Use |
|---|---|
| `Sequential([...])` | stack layers |
| `model.summary()` | layers, output shapes, params |
| `model.compile(optimizer, loss, metrics)` | configure training |
| `model.fit(x, y, epochs, batch_size, validation_data)` | train |
| `model.evaluate(x, y)` | test loss & accuracy |
| `model.predict(x)` | output probabilities |
| `np.argmax(pred)` | probability → class index |
| `ImageDataGenerator(rescale=1/255).flow_from_directory()` | load images from folders |
| `model.save("m.keras")` / `load_model()` | save / load model |
