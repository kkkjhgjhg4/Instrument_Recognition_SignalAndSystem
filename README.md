# Instrument_Recognition_SignalAndSystem
This is a repository for signal and system course in School of Aerospace Engineering, Tsinghua University. 

## Dataset
Please down IRMAS dataset here (https://zenodo.org/records/1290750#.WzCwSRyxXMU) to run the model. 
Put the Data set in following data directory relative to root directory of this repository.
```angular2html
Instrument_recognition_SignalAndSystem
|---IRMAS
    |---IRMAS-TrainingData
        |---"same directory tree as you downloaded"
    |---IRMAS_TestingData-Part1
        |---"same directory tree as you downloaded"
```

## Environment
All code run on Ubuntu18.04.6LTS, the machine has an Intel i7-6700K cpu and 16G Ram. Training process of **ALL CNN MODELS**  were sent to the NVIDIA RTX TITAN GPU by cuda. 

You can run following command in the terminal to install most of dependencies.
```angular2html
pip3 install -r requirements.txt
```
For pytorch and cuda, please refer to the official document to install in your working environment.

To test your Cuda and GPU availability by running the TestCuda.py script. It should return a message say that the task was done on cuda if everything goes right.

## My Project
My work is divided into 3 parts, and stored in 3 folders, CNN Model, SVM Model and Instrument Recognition Software. 

### SVM Model
Within the SVM Model folder, all code is included in SVM_Model.ipynb. Launch the file with jupyter notebook server block by block, then you can see the results. 
Functions including preprocessing dataset, feature extraction, model fitting and results analyzing. The extracted feature will be store in data.csv file, including data label, MFCC, RMS, etc. 

### CNN Model
Within the CNN Model folder, we have
```angular2html
|---CNN Model
    |---TITAN
        |---Deep_CNN_Custome
            |---Deep_CNN_Custome.ipynb
            |---final_dual_input_model.pth
        |---Deep_CNN_Han
        |---Deep_CNN_with_DWT
```
**Our final work is included in the Deep_CNN_Custome folder, in file Deep_CNN_Custome.ipynb. This is the model we create originally, which delivers best performance along all.**

For the Deep_CNN_Han, it was a review on a famous academic paper from Han. This is the code I've write based on his research.

### Instrument Recognition Software
This folder contains the GUI software takes audio input and predict the instrument used in it based on the CNN trained model or SVM model. 

Just run the Recognition_software.py and use the software. Remember to install PyQt5 as dependency.