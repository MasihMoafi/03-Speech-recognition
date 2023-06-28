## Speech Recognition using CNN and Transformers

This project aims to classify spoken digits in audio recordings using Convolutional Neural Networks (CNN). The dataset used for training and testing consists of 3,000 audio recordings of spoken digits.

# Requirements

    Python 3.x
    TensorFlow 2.x
    Librosa
    NumPy
    scikit-learn
    Matplotlib

# Dataset

The dataset used in this project is the Free Spoken Digit Dataset (FSDD), which contains audio recordings of spoken digits from 0 to 9. Each audio recording is a WAV file with a sampling rate of 8kHz. Dataset can be found here: https://www.kaggle.com/datasets/joserzapata/free-spoken-digit-dataset-fsdd

# Approach

CNN (Convolutional Neural Network)

The CNN is used to extract features from the audio recordings. The MFCC (Mel-frequency cepstral coefficients) features are calculated for each audio recording using the Librosa library. The MFCC features are then padded to have the same length and normalized to have zero mean and unit variance. The CNN architecture consists of two 1D convolutional layers with batch normalization and ReLU activation. The output of the CNN is flattened and passed through a dropout layer before reaching the output layer.

# Model Training

The model is trained using the Adam optimizer and the categorical cross-entropy loss function. Early stopping is applied to prevent overfitting. The training is performed for a maximum of 100 epochs or until early stopping criteria are met.

# Results

We've used F1 score, precision, recall and other evaluation techniques. 
