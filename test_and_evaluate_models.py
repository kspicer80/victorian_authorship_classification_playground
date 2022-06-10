import load_and_shuffle_data
import train_models
#import pytest

def test_train_ngram_model():
    data_dir = r"C:\Users\KSpicer\Desktop\victorian_era_authorship_attribution_project\dataset"
    data_file_name = r"training_data.csv"
    columns = (0, 1)
    data = load_and_shuffle_data.load_victorian_dataset(data_dir, data_file_name, columns, 0.2)
    acc, loss = train_models.train_ngram_model(data)
    print(f"The accuracy of the model is: {acc}")
    print("\n")
    print(f"The loss of the model is: {loss}")

def plot_model_loss(model_name, string_1='loss', string_2='val_loss'):
    plt.plot(model_name.history[string_1])
    plt.plot(model_name.history[string_2])
    plt.title('model loss')
    plt.ylabel(string_1)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def plot_model_accuracy(model_name, string_1='accuracy', string_2='val_accuracy'):
    plt.plot(model_name.history[string_1])
    plt.plot(model_name.history[string_2])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='lower right')
    plt.show()

test_train_ngram_model()
