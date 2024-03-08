import torch 
import numpy as np 
from mlc_binning_fn import tot_ECE, tot_uncal_ECE, avg_ECE, avg_uncal_ECE, new_predictions_test_data, predictions
from sklearn.metrics import hamming_loss

#path to output and ground truths of validation and test data
val_preds= np.load('RFMiD/output_files_raw/rfmid_s42/val_384_pred.npy')
val_labels= np.load('RFMiD/output_files_raw/rfmid_s42/val_384_gt.npy')

test_preds= np.load('RFMiD/output_files_raw/rfmid_s42/test_384_pred.npy')
test_labels= np.load('RFMiD/output_files_raw/rfmid_s42/test_384_gt.npy')

val_preds= np.array(torch.sigmoid(torch.tensor(val_preds))).T
val_labels= val_labels.T

test_preds= np.array(torch.sigmoid(torch.tensor(test_preds))).T
test_labels= test_labels.T

test_size= test_preds.shape[1]

tot_calibrated_ECE= tot_ECE(val_preds, 10, val_labels, test_preds, test_labels)
tot_uncalibrated_ECE= tot_uncal_ECE(test_labels, test_preds,10)
print('Total ECE for Calibrated: ')
print(tot_calibrated_ECE)
print('Total ECE for uncalibrated: ')
print(tot_uncalibrated_ECE)




avg_calibrated_ECE= avg_ECE(val_preds, 10, val_labels, test_preds, test_labels, test_size)
avg_uncalibrated_ECE= avg_uncal_ECE(test_labels, test_preds,10, test_size)

print('Weighted average ECE for calibrated:')
print(avg_calibrated_ECE)
print('Weighted average ECE for Uncalibrated:')
print(avg_uncalibrated_ECE)

calibrated_test_preds= new_predictions_test_data(val_preds, val_labels, test_preds, 10)

print('The hamming losses are\n')
print('For uncalibrated-' ,hamming_loss(predictions(data= test_preds), test_labels))
print('For Calibrated-',hamming_loss(predictions(data= calibrated_test_preds), test_labels))