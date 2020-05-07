# Face Recognition using Deep Learning with OpenCV and Keras

This project is built-in Python and it deals with Face Recognition.
The Deep Learning model is built using the Keras framework and has built using the principle of Transfer Learning. The Transfer Learning model used is the VGG16 model. The model is trainable and customizable.

## Installation

Following are the necessary packages for this project

```bash
conda install -c menpo opencv3
conda install keras
pip install Pillow
```

## Usage

Step 1: Run the Register_Face.py file

This script is for creating the data which would later be needed for training. It deals with reading images from your webcam using OpenCV and storing them as images in the 'images' folder.
```bash
python Register_Face.py
```
Step 2: Run the datagen.py file

This script is for moving the images read earlier into a folder named 'data' which would be used by the data generators during the training and process. It will readily create the training and testing folders for each person.
```bash
python datagen.py
```
Step 3: Run the train.py file

This script will carry out the training process. The weights of the trained model would then be saved.
```bash
python train.py
```
Step 3: Run the FaceRecognition.py file

This script will carry out the recognition process by loading the trained model.
```bash
python FaceRecognition.py
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
