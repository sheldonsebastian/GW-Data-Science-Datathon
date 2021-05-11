## README: 

### Final Report:
https://sheldonsebastian.github.io/GW-Data-Science-Datathon/

### Folder Structure:

| Path | Description | 
|------|-------------|
| common | contains utility functions to clean data, perform evaluation, modelling, etc. |
| images | contains all saved images |
| input_data | contains the input data, cleaned train-test-validation data and feature-target NumPy arrays |
| model_trainer | contains model training and feature importance notebooks |
| 0_preprocessing_eda.ipynb | contains cleaning, preprocessing, stratified split and feature-target separator code |
| 1_final_report.ipynb | FINAL REPORT |

### Steps to replicate project:

1. Download data from [here](https://opendata.dc.gov/datasets/70248b73c20f46b0a5ee895fc91d6222/data).
2. Run 0_preprocessing_eda.ipynb to preprocess data and create stratified train-test-holdout splits.
3. Run all scripts in model_trainer folder to create all the respective models
