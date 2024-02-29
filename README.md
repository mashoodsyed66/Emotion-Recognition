## Emotion Recognition Project

### Overview

The Emotion Recognition project aims to detect and classify human emotions from facial expressions using machine learning techniques. The project utilizes the FER dataset, which consists of 28,709 48x48 images depicting various emotions such as happy, sad, neutral, and angry. 

### Techniques Used

The following techniques were employed in the project:

1. **Encoding**: The target variable was encoded into numerical values to prepare it for training with machine learning algorithms.

2. **Data Augmentation**: Data augmentation techniques were applied to increase the diversity of the training dataset. Techniques such as rotation, scaling, and flipping were used to generate additional training samples from the original images.

3. **Dropout Regularization**: Dropout regularization was implemented in the CNN model to prevent overfitting. Dropout randomly drops a fraction of neurons during training, forcing the network to learn more robust features and reducing the reliance on specific neurons.

4. **Convolutional Neural Network (CNN)**: A CNN architecture was employed for image classification tasks. CNNs are particularly effective for analyzing visual data like images due to their ability to automatically learn hierarchical features.

### Model Architecture

The CNN model used for emotion recognition consists of several convolutional layers followed by max-pooling layers, batch normalization, dropout layers, and fully connected layers. The architecture was designed to effectively capture spatial features from the input images and learn discriminative representations for emotion classification.

### Training Process

The model was trained using the FER dataset, with the compiled configuration including:
- **Optimizer**: Adam optimizer with a learning rate of 0.0001.
- **Loss Function**: Sparse categorical cross-entropy.
- **Metrics**: Accuracy.

To prevent overfitting, dropout regularization was implemented during training. This technique randomly drops a fraction of neurons during training, preventing the network from relying too much on specific neurons and features.

Additionally, early stopping was employed to monitor the validation loss and stop training if there was no improvement after a specified number of epochs, restoring the model weights to those from the epoch with the best validation performance.

### Real-Time Emotion Recognition with Webcam
After training the model, we first evaluated its performance on the test dataset to assess accuracy and generalization.

After that we implemented real-time emotion recognition using the webcam, where we continuously captured frames, extracted faces, and processed them according to the model's requirements. The processed images were then fed into the trained model to obtain real-time predictions, which were displayed directly on the webcam feed.

This approach provided immediate insights into the model's ability to recognize emotions in dynamic environments, showcasing its practical utility in interactive applications.

### Results
You can view the screenshots of the model's predictions on custom images below:

[Insert screenshots here]

###Summary
The Emotion Recognition project involved preprocessing the FER dataset, including encoding the target variable, applying data augmentation techniques, implementing dropout regularization, and training a CNN model for emotion classification. 
The trained model achieved satisfactory performance in accurately recognizing and classifying human emotions from facial expressions, demonstrating the effectiveness of the employed techniques in handling the complexities of image data.
