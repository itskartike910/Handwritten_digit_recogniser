
https://github.com/user-attachments/assets/c98809bb-8e40-4afd-873d-72ee9152baaf
# Handwritten Digit Recognizer

A machine learning project to recognize handwritten digits (0-9) using image data from the MNIST dataset. This project includes code for training, testing, and deploying a model that can classify images of digits. The model uses a Convolutional Neural Network (CNN) architecture to achieve high accuracy on digit recognition tasks.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [File Descriptions](#file-descriptions)
6. [Dataset](#dataset)
7. [Model Architecture](#model-architecture)
8. [Results](#results)
9. [Deployment](#deployment)
10. [Contributing](#contributing)
11. [License](#license)
12. [Demo](#demo)

## Project Overview

This project builds a neural network model trained on the MNIST dataset to recognize handwritten digits. By training on thousands of images, the model can classify new images with high accuracy. This project uses TensorFlow and Keras libraries and is implemented in both a Python script (`HandwrittenDigitRecognition.py`) and Jupyter notebooks for exploration and training (`MnistDataset.ipynb` and `HandwrittenDigitsRecognition.ipynb`).

## Features

- Recognizes handwritten digits (0-9) from 28x28 grayscale images.
- Trained on the MNIST dataset for reliable performance.
- Includes a trained model (`mnist_cnn_model.h5`) ready for testing and deployment.
- Uses Python’s TensorFlow and Keras libraries, allowing easy customization.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/Handwritten_Digit_Recognizer.git
   cd Handwritten_Digit_Recognizer
   ```

2. **Install Required Libraries:**
   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` includes dependencies such as TensorFlow, Keras, NumPy, and Matplotlib.

## Usage

1. **Train the Model**:
   Run the `HandwrittenDigitRecognition.py` script to train the model on the MNIST dataset.
   ```bash
   python HandwrittenDigitRecognition.py
   ```

2. **Test the Model**:
   After training, test the model using `test.py`.
   ```bash
   python test.py
   ```

3. **Explore Jupyter Notebooks**:
   Open `MnistDataset.ipynb` or `HandwrittenDigitsRecognition.ipynb` in Jupyter Notebook for interactive exploration and training.

4. **Run Deployment Script**:
   Use `deployment.py` for deploying the model to recognize new handwritten digit images.

## File Descriptions

- **HandwrittenDigitRecognition.py**: Main script for training the model on the MNIST dataset.
- **MnistDataset.ipynb**: Jupyter notebook for loading, visualizing, and pre-processing the MNIST dataset.
- **HandwrittenDigitsRecognition.ipynb**: Jupyter notebook for training and testing the CNN model.
- **test.py**: Script to test the accuracy and performance of the trained model.
- **deployment.py**: Script for deploying the model to recognize new handwritten digit images.
- **mnist_cnn_model.h5**: Saved model file, containing the trained model weights.
- **requirements.txt**: Contains a list of required Python libraries.
- **LICENSE**: The license for this project.
- **four.jpg** and **two.jpg**: Sample images used for testing the model.

## Dataset

The MNIST dataset is used in this project, containing 60,000 training images and 10,000 test images of handwritten digits (0-9). TensorFlow/Keras can directly load this dataset for training and testing.

## Model Architecture

The model is a Convolutional Neural Network (CNN) that includes:

- **Convolutional Layers**: For feature extraction.
- **Pooling Layers**: For dimensionality reduction.
- **Fully Connected Layers**: For classification of digit classes.

The model architecture and hyperparameters can be modified in `HandwrittenDigitRecognition.py` and the Jupyter notebooks.

## Results

The trained model achieves a high accuracy on the MNIST test set, typically above 98%. This can be tested using the `test.py` script, or by examining results in the notebooks.

## Deployment

The model is ready for deployment and can be tested on custom images using the `deployment.py` script. This script allows loading new images (such as `four.jpg` or `two.jpg`) and running the model to predict the digit in the image.

## Contributing

Contributions are welcome! Feel free to submit issues, fork the repository, and create pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Demo

To see the model in action, watch the [demo video](https://drive.google.com/file/d/1SKUyZ6fr9OacFVdbbP4ZJaWL9JElHF8O/view?usp=sharing). This video demonstrates how the model processes handwritten digit images and accurately predicts their labels.
