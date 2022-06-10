import load_and_shuffle_data
import train_models
import pytest

def test_train_ngram_model():
    data_dir = r"C:\Users\KSpicer\Desktop\victorian_era_authorship_attribution_project\dataset"
    data_file_name = r"training_data.csv"
    columns = (0, 1)
    data = load_and_shuffle_data.load_victorian_dataset(data_dir, data_file_name, columns, 0.2)
    acc, loss = train_models.train_ngram_model(data)
    assert acc == pytest.approx(0.80, 0.02)
    assert loss == pytest.approx(0.24, 0.02)
    print("The accuracy of the model is: " + acc)
    print("\n")
    print("The loss of the model is: " + loss)

test_train_ngram_model()
