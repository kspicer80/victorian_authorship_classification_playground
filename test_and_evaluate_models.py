import load_and_shuffle_data
import train_models
#import pytest
import matplotlib.pyplot as plt

def test_train_ngram_model():
    #windows_data_dir = r"C:\Users\KSpicer\Desktop\victorian_era_authorship_attribution_project\dataset"
    mac_data_dir = '/Users/spicy.kev/Desktop/victorian_authorship'
    data_file_name = "victorian_training_data.csv"
    columns = (0, 1)
    data = load_and_shuffle_data.load_victorian_dataset(mac_data_dir, data_file_name, columns, 0.2)
    acc, loss = train_models.train_ngram_model(data, .007)
    
test_train_ngram_model()
