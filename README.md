# Calibrating Neural Networks for Medical Image Classification: A Model Agnostic Approach
This is the official code for Histogram Binning for Multi-Label classification (HBMLC), a method for calibrating multi-label medical image classification tasks. The prediction model (ResNet-50) is trained on PyTorch version 2.1.2 with cuda enabled and torchvision version 0.16.2.
## Data Preparation

Add the paths of the training, validation and test directory in ```dataloader.py```. Add the required transformations in ```main.py```.  

## Model Prediction

### Training:

To start the training, execute ```main.py``` and set the required dataset specific configurations within the file. 
``` 
python main.py 
```
**Note:** The best model will be saved in the ```saved_data``` folder. 
### Evaluation:

In order to obtain the model predictions, execute the ``eval.py`` file by setting the path to the testing model. 

## Calibration
To calibrate the predictions, execute ```mlc_calibrate.py``` by updating the path to the predicted validation outputs, validation ground truths, test outputs and test ground truths files  in the following variables:

- ```val_preds```   :   Defines the path to predicted validation data
- ```test_preds``` : Defines the path to predicted test data
- ```val_labels``` : Defines the to validation ground truth
- ```test_labels``` : Defines the path to test ground truths

## Remarks
Currently, we are releasing the code for one dataset (ChestMNIST). The full code will be released post acceptance. 
