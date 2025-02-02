This model is a CNN-based image classifier designed to distinguish between cats and dogs. It consists of three convolutional layers (16, 32, 64 filters), each followed by max-pooling for feature extraction. A dropout layer (0.2) prevents overfitting, and a fully connected (Dense) layer processes the extracted features before outputting logits for classification.

Preprocessing includes:

Image resizing (180x180), normalization (1/255), and shuffling.
Data augmentation (random flipping, rotation, zoom) for better generalization.
Training with Adam optimizer and Sparse Categorical Crossentropy loss for 66 epochs.
The trained model is evaluated on a test set and used to predict new images using softmax 
