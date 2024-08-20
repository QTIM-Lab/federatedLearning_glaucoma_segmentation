# federatedLearning_glaucoma_segmentation
 Federated Learning Optic Disc and Cup Segmentation Model for Glaucoma Monitoring in Color Fundus Photographs(CFPs)

## The 3 FL pipelines are:

### Pipeline1: Global Validation
1. 1 training round per site -> FedAvg -> FedAvg validated on GlobalVal (val for all sites)
2. .py to run --> './py_code_files/fl_train_glaucoma_seg.py'

### Pipeline2: Global Validation - weighted
1. same as pipeline 1 with weightedVal val feature (since number of validation samples affects how accurately you can gauge the model's performance and how you should weight each site's contribution to the global validation metrics), in each site, during calculation of val_loss:

val_loss for each site = n/N * val_loss,
n==no. of val samples in the site, N== total no. of val samples for all sites

2. .py file to run --> './py_code_files/globalVal_weighted_val_code_cuda_optimized.py'
   
### Pipeline3: Onsite Validation
1. n training epochs per site + val with early stopping -> sending each site’s best performing model on it’s own val set to the global model -> global model val_loss=mean(val_losses of the individual sites)
2. .py to run --> './py_code_files/fl_train_glaucoma_seg_training_epochs_cudaOptimized.py'




