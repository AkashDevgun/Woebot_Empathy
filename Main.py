# Importing Main Libraries
import torch

torch.cuda.current_device()
import numpy as np
from torch.utils.data import DataLoader
from preprocess_load_embedding import EmpathyDataset, DataPreprocessing
from sklearn.model_selection import train_test_split
import torch.backends.cudnn as cudnn
from models_LSTMs import LSTM_fixed_len, LSTM_variable_input, LSTM_glove_vecs
from train_test_lossCriterion import train, get_optimizer_criterion_scheduler
import warnings

warnings.filterwarnings('ignore')
np.random.seed(1)

# Model was trained using GPU in CUDA Environment
print("Cuda Available: {}".format(torch.cuda.is_available()))

# File Names
labelled_message_file = "/media/HDD_2TB.1/machine-learning-engineer/labeled_messages.csv"
empathies_file = "/media/HDD_2TB.1/machine-learning-engineer/empathies.csv"


# Method that calls to train different types of LSTMs
def train_lstms(model, num_epochs, learning_rate, loss_weights, device, train_queue, valid_queue):
    model = model.to(device)
    criterion, optimizer, scheduler = get_optimizer_criterion_scheduler(model, num_epochs, learning_rate,
                                                                        loss_weights, device)

    for epoch in range(num_epochs):
        scheduler.step()
        train(model, device, train_queue, valid_queue, optimizer, epoch, criterion)


# Main Method
def main():
    # Object 'Data' is created by Class Named -> 'DataPreprocessing' with file names as parameters
    data = DataPreprocessing(labelled_message_file, empathies_file)

    # Method Call describes number of words in Corpus, messages lengths
    data.describe_counts()

    # Method call for MultiLabel Encoding, for weighted label weights to handle imbalance
    output_size, loss_weights = data.label_binarizer_get_weights()

    # CAll to Get X and Y
    X = data.get_X_data()
    y = data.get_Y_data()
    print('\n')

    # Train and Test Split
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3)

    # Baseline Classifier using SVC to calculate Accuracy and Area Under Curve Scores
    print("***** Baseline AUC Scores *****")
    acc_svm, roc_svm = data.modelling("SVC", X_train, X_valid, y_train, y_valid)
    print("SVM Modelling --> Validation Acc. : %.3f, Validation AUC Score : %.3f" % (acc_svm, roc_svm))
    acc_RF, roc_RF = data.modelling("RandomForest", X_train, X_valid, y_train, y_valid)

    print("***** Statistical Method better then Baseline *****")
    # Baseline Classifier using SVC to calculate Accuracy and Area Under Curve Scores
    print("Random Forest Modelling --> Validation Acc. : %.3f, Validation AUC Score : %.3f" % (acc_RF, roc_RF))

    # Class 'EmpathyDataset' Called for train and valid dataset to load while run time during training and testing
    train_ds = EmpathyDataset(X_train, y_train)
    valid_ds = EmpathyDataset(X_valid, y_valid)

    vocab_size = len(data.words)
    num_epochs = 1001
    batch_size = 1000
    learning_rate = 0.3

    # Data Loader for train and test
    train_queue = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_queue = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    # CUDA Environment Settings
    torch.cuda.set_device(0)
    cudnn.benchmark = True
    torch.manual_seed(1)
    cudnn.enabled = True
    torch.cuda.manual_seed(1)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # LSTMs models with fixed length Input, variable length Input, using StandFord Glove Representations
    print('\n')
    print('-----------LSTMs Fixed Length Input--------------')
    model1 = LSTM_fixed_len(vocab_size, 48, 96, output_size)
    train_lstms(model1, num_epochs, learning_rate, loss_weights, device, train_queue, valid_queue)

    print('\n')
    print('-----------LSTMs Variable Length Input--------------')
    model2 = LSTM_variable_input(vocab_size, 48, 96, output_size)
    train_lstms(model2, num_epochs, learning_rate, loss_weights, device, train_queue, valid_queue)

    print('\n')
    print('-----------LSTMs with Glove Representation of Input--------------')
    word_vecs = data.load_glove_vectors()
    pretrained_weights, vocab, vocab2index = data.get_emb_matrix(word_vecs)
    model3 = LSTM_glove_vecs(vocab_size, 50, 96, pretrained_weights, output_size)
    train_lstms(model3, num_epochs, learning_rate, loss_weights, device, train_queue, valid_queue)


if __name__ == '__main__':
    main()
