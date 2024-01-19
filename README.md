# Engineering Tools Classification leveraging two pre-trained models

This repository contains two Jupyter notebooks detailing the process and results of a deep learning project aimed at classifying engineering tools. The project utilizes transfer learning with two different architectures: MobileNetV2 and InceptionV3, and compares their performance with and without data augmentation.

## Notebooks

- `DL_Task1_Image_classification_AUG.ipynb`: This notebook demonstrates the training and evaluation of the models using data augmentation.
- `DL_Task1_Image_classification_NO_AUG.ipynb`: This notebook focuses on the performance of the models without the use of data augmentation.

## Project Overview

The purpose of this project is to explore the effectiveness of transfer learning in the context of image classification, particularly for engineering tools which can be challenging due to their similarities and fine-grained features.

## Models

- **MobileNetV2**: A lightweight deep neural network known for its efficiency on mobile devices.
- **InceptionV3**: A more complex architecture that is known for its high accuracy in image classification tasks.

## Data Augmentation

Data augmentation involves applying a series of random transformations to the training images, thereby increasing the diversity of the training data without actually collecting new data. This technique helps to reduce overfitting and improve the model's generalization capabilities.

## Results

The results indicate that data augmentation has a significant impact on the performance of the models. The notebooks include detailed performance metrics such as accuracy, precision, recall, and F1 scores, along with visualizations of the training and validation accuracy and loss.

## Usage

To run these notebooks, ensure you have Jupyter Notebook or JupyterLab installed. You can launch the notebooks by navigating to the repository's directory in your terminal and running the command `jupyter notebook`.

## Requirements

- numpy==1.26.2
- seaborn==0.13.0
- matplotlib==3.8.2
- tensorflow==2.15.0
- tensorflow-hub==0.15.0
- scikit-learn==1.3.2

Please install the dependencies using `pip install -r requirements.txt` before running the notebooks.

## Contributing

Feel free to fork this repository or submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is open source and available under the [MIT License](LICENSE).

## Contact

If you have any questions or would like to connect, reach out to me on [LinkedIn](https://www.linkedin.com/in/emmanuel-olowoboko-42205172/).

## Acknowledgements

- Special thanks to all the contributors of the TensorFlow and Keras libraries for making machine learning more accessible.
- Large part of the data used in this project were manually collected and the rest were downloaded from Google.
