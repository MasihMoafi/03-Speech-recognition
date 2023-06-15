#Speech Recognition using CNN and Transformers

This project aims to classify spoken digits in audio recordings using a combination of Convolutional Neural Networks (CNN) and Transformers. The dataset used for training and testing consists of 3,000 audio recordings of spoken digits.
Requirements

    Python 3.x
    TensorFlow
    Librosa
    NumPy
    scikit-learn
    Matplotlib

Dataset

The dataset used in this project is the Free Spoken Digit Dataset (FSDD), which contains audio recordings of spoken digits from 0 to 9. Each audio recording is a WAV file with a sampling rate of 8kHz. The dataset is available in the recordings directory.
Approach

The classification model consists of two main parts: a CNN and a Transformer.
CNN (Convolutional Neural Network)

The CNN is used to extract features from the audio recordings. The MFCC (Mel-frequency cepstral coefficients) features are calculated for each audio recording using the Librosa library. The MFCC features are then padded to have the same length and normalized to have zero mean and unit variance. The CNN architecture consists of two 1D convolutional layers with batch normalization and ReLU activation. The output of the CNN is flattened and passed through a dropout layer before reaching the output layer.
Transformer

The Transformer is applied after the CNN to capture long-range dependencies in the audio recordings. The Transformer architecture consists of a multi-head self-attention layer followed by a feed-forward neural network. The output of the Transformer is flattened and passed through a dropout layer before reaching the output layer.
Model Training

The model is trained using the Adam optimizer and the categorical cross-entropy loss function. Early stopping is applied to prevent overfitting. The training is performed for a maximum of 100 epochs or until early stopping criteria are met.
Results

The model achieves an accuracy of 97.83% on the test set using the CNN and Transformer combination.
Usage

    Ensure that all the required libraries are installed.
    Set the directory variable to the path of the directory containing the audio recordings.
    Run the code to train and evaluate the model.
    The loss and accuracy values will be printed to the console.
    A plot showing the training and validation loss values will be displayed.
    Precision, recall, and F1-score will be calculated and printed to the console.

Conclusion

In this project, a combination of CNN and Transformers was used for audio classification. The results show that using the CNN alone achieved good performance, while adding Transformers did not significantly improve the results for this specific dataset. However, the manually implemented Transformers can still be considered as a valuable addition to a portfolio, showcasing the ability to combine different deep learning techniques for audio classification tasks.
