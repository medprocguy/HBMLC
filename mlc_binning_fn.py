import torch 
import numpy as np 


#function for separating predictions and labels(size is the number of data points, preds_labels is the predicted dataset)

def prediction_and_label(preds_labels):
    preds,labels =[],[]
    for item in preds_labels:
        preds.append((torch.sigmoid(item[0][0])).detach().numpy().tolist())
        labels.append(item[1][0].detach().numpy().tolist())
    return np.array(preds).T, np.array(labels).T


# def prediction_and_label(preds_labels,size):

#     preds = torch.zeros(20, size)
#     labels= torch.zeros(20, size)

#     for i in range(0, 20):
#         for j in range(0, size):
#             for k in range(0, len(preds_labels)):
#                     for l in range(0, len(preds_labels[k][0])):
#                         preds[i][j]= preds_labels[k][0][l][i]
#                         labels[i][j]= preds_labels[k][1][l][i]

#     preds= torch.sigmoid(preds)
#     preds=np.array(preds)
#     labels=np.array(labels)
#     return preds, labels

#function to determine in which bin the predictions belongs to
def bin_index(num_bins, predictions):
    bin_idx= np.zeros((predictions.shape[0], predictions.shape[1]))
    for i in range(0, predictions.shape[0]):
        bins= np.linspace(0,1, num_bins+1)
        hist, bin_edges= np.histogram(predictions[i,:], bins=bins)

        bin_idx[i,:]= np.digitize(predictions[i,:], bins[:-1] )
    return bin_idx



#values after calibration by validation data(num_classses*num_bins matrix)
def predictions_by_bin(predictions, num_bins, labels):
    pred_by_bin= np.zeros((predictions.shape[0], num_bins))
    bin_idx= bin_index(num_bins, predictions)
    for k in range(0, predictions.shape[0]):
        for i in range(1, num_bins+1):
            nu=0
            de=0
            for j in range(0,bin_idx.shape[1]):
                if bin_idx[k,:][j]== i and labels[k,:][j]==1:
                    nu=nu+1
                if bin_idx[k,:][j]==i:
                    de=de+1
            if de==0:
                pred_by_bin[k][i-1]=0
            else:
                pred_by_bin[k][i-1]= nu/de
    
    return pred_by_bin
            

#for assigning calibrated values to the predicted test probabilities(the output will only be related to test data)
def new_predictions_test_data(val_predictions,val_labels, test_predictions, num_bins):
    val_pred_by_bin= predictions_by_bin(val_predictions,num_bins, val_labels) 
    new_test_pred= np.zeros((test_predictions.shape[0], test_predictions.shape[1]))
    bins= np.linspace(0,1, num_bins+1)
    for i in range(0, test_predictions.shape[0]):
        for j in range(0, test_predictions.shape[1]):
            bin_number= np.digitize(test_predictions[i][j], bins=bins[:-1])
            new_test_pred[i][j]= val_pred_by_bin[i][bin_number-1]
    return new_test_pred


#To check the number of positive and total instances per bin in each and every bins in any data(train, validation or predict)
def positive_and_total_instances( labels, num_bins, predictions):
    
    bin_index_set= bin_index(num_bins, predictions)
    positive= np.zeros((bin_index_set.shape[0], num_bins))
    total= np.zeros((bin_index_set.shape[0], num_bins))


    for i in range(0,bin_index_set.shape[0]):
        for j in range(1, num_bins+1):
            pos=0
            tot=0
            for k in range(0,bin_index_set.shape[1]):
                if bin_index_set[i][k]==j and labels[i][k]==1:
                    pos= pos+1
                if bin_index_set[i][k]==j:
                    tot=tot+1
            positive[i][j-1]= pos
            total[i][j-1]= tot
    
    return positive, total

#the fraction of positive instances in each bin
def fraction_of_positive_instances(labels, num_bins, predictions):
    positive_inst_perbin, total_inst_perbin= positive_and_total_instances(labels, num_bins, predictions)

    fraction_positive= np.zeros((positive_inst_perbin.shape[0], positive_inst_perbin.shape[1]))
    for i in range(0, positive_inst_perbin.shape[0]):
        for j in range(0, positive_inst_perbin.shape[1]):
            if total_inst_perbin[i][j]!=0:
                fraction_positive[i][j]= positive_inst_perbin[i][j]/total_inst_perbin[i][j]
            else:
                fraction_positive[i][j]= 0
    return fraction_positive


#weighted average ECE for calibrated predictions 
def avg_ECE(validation_predictions, num_bins, validation_labels, test_predictions, test_labels, data_size):
    positive_inst_perbin, total_inst_perbin= positive_and_total_instances(test_labels, num_bins, test_predictions)
    weights= positive_inst_perbin.sum(axis=1)
    ece=np.zeros(5)
    confidence_matrix= predictions_by_bin(predictions=validation_predictions, num_bins=num_bins, labels= validation_labels)
    positive_fractions_of_test_data= fraction_of_positive_instances(test_labels, num_bins, test_predictions)
    for i in range(0, positive_fractions_of_test_data.shape[0]):
        for j in range(0, num_bins):
            if total_inst_perbin[i][j]!=0:
                ece[i]= ece[i]+ (total_inst_perbin[i][j]/test_predictions.shape[1])* abs(positive_fractions_of_test_data[i][j]- confidence_matrix[i][j])
    
    return np.sum(weights*ece)/data_size 


def avg_bin_conf(predictions, num_bins):
    bin_idx= bin_index(num_bins= num_bins, predictions= predictions )
    avg_pred= np.zeros((bin_idx.shape[0], num_bins ))
    for i in range(0, bin_idx.shape[0]):
        for j in range(1, num_bins+1):
            count=0
            cuml_prob=0
            for k in range(0, bin_idx.shape[1]):
                if bin_idx[i][k]==j:
                    count=count+1
                    cuml_prob= cuml_prob+predictions[i][k]
            if count!=0:
                avg_pred[i][j-1]= cuml_prob/count
            else:
                avg_pred[i][j-1]=0
    return avg_pred
                    
#weighted average ECE for uncalibrated predictions 

def avg_uncal_ECE(labels, predictions, num_bins, data_size):
    positive_inst_perbin, total_inst_perbin= positive_and_total_instances(labels, num_bins, predictions)
    weights= positive_inst_perbin.sum(axis=1)
    ece=np.zeros(5)
    avg_conf= avg_bin_conf(predictions= predictions, num_bins= num_bins)
    positive_frac= fraction_of_positive_instances(labels, num_bins, predictions)
    for i in range(0, avg_conf.shape[0]):
        for j in range(0, avg_conf.shape[1]):
            if total_inst_perbin[i][j]!=0:
                ece[i]= ece[i]+ (total_inst_perbin[i][j]/predictions.shape[1])*abs(avg_conf[i][j]- positive_frac[i][j])
    return np.sum(weights*ece)/data_size

#Total ECE after calibration

def tot_ECE(validation_predictions, num_bins, validation_labels, test_predictions, test_labels):
    positive_inst_perbin, total_inst_perbin= positive_and_total_instances(test_labels, num_bins, test_predictions)
    #weights= positive_inst_perbin.sum(axis=1)
    ece=np.zeros(5)
    confidence_matrix= predictions_by_bin(predictions=validation_predictions, num_bins=num_bins, labels= validation_labels)
    positive_fractions_of_test_data= fraction_of_positive_instances(test_labels, num_bins, test_predictions)
    for i in range(0, positive_fractions_of_test_data.shape[0]):
        for j in range(0, num_bins):
            if total_inst_perbin[i][j]!=0:
                ece[i]= ece[i]+ (total_inst_perbin[i][j]/test_predictions.shape[1])* abs(positive_fractions_of_test_data[i][j]- confidence_matrix[i][j])
    
    return ece.sum()

#Total ECE before calibration
def tot_uncal_ECE(labels, predictions, num_bins):
    positive_inst_perbin, total_inst_perbin= positive_and_total_instances(labels, num_bins, predictions)
    #weights= positive_inst_perbin.sum(axis=1)
    ece=np.zeros(5)
    avg_conf= avg_bin_conf(predictions= predictions, num_bins= num_bins)
    positive_frac= fraction_of_positive_instances(labels, num_bins, predictions)
    for i in range(0, avg_conf.shape[0]):
        for j in range(0, avg_conf.shape[1]):
            if total_inst_perbin[i][j]!=0:
                ece[i]= ece[i]+ (total_inst_perbin[i][j]/predictions.shape[1])*abs(avg_conf[i][j]- positive_frac[i][j])
    return ece.sum()



def predictions(data):
    return np.round(data)