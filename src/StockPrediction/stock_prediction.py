import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.StockPrediction.nnconfig import NNConfig
from src.StockPrediction.prediction_data import PredictionData
from src.StockPrediction.strategy_calculator import StrategyCalculator
from src.helper_functions import format_time, get_logger

logger = get_logger(LOG_NAME="stock_predictor")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def plot_error(loss):
    plt.figure(figsize=(10, 5))
    plt.title('Training Loss')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    plt.plot(loss)
    plt.show()


def save_model(model, training_loss, config):
    torch.save(model.state_dict(), config.model_save_location)
    file = open(config.stock_prediction_var_path, mode="w+")
    file.write(str(training_loss) + "\n" + str(config.get_current_parameters()))
    file.close()


def load_var(config):
    file = Path(config.stock_prediction_var_path)
    if file.is_file():
        file = open(config.stock_prediction_var_path, mode="r")
        least_loss = float(file.readline())
        params = eval(file.readline())
        file.close()
    else:
        params = None
        least_loss = 100
    return least_loss, params


def model_training(model, optimizer, dataloader, num_epochs, loss_function, config, do_plot):
    loss_values = []
    training_predictions = []
    training_targets = []
    t0 = time.time()

    for epoch in range(0, num_epochs):

        total_loss = 0

        model.train()

        for batch in dataloader:
            batch_input = batch["input_seq"].to(device)
            batch_target = batch["target"].to(device)

            optimizer.zero_grad()  # Prevent accumulation of gradients between passes

            y_pred = model(batch_input)

            if epoch == (num_epochs - 1):  # Saves the training predictions for the last epoch
                training_predictions.extend(y_pred)
                training_targets.extend(batch_target)

            # single_loss = loss_function(y_pred, batch_target)
            single_loss = torch.sqrt(loss_function(y_pred, batch_target))

            total_loss += single_loss.item()

            single_loss.backward()  # Performs backpropagation to calculated and store gradients

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.weight_clipping)

            optimizer.step()  # Updates model weights based of stored gradients

        avg_loss = total_loss / len(dataloader)

        loss_values.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            elapsed = format_time(time.time() - t0)
            eta = format_time(((time.time() - t0) / (epoch + 1)) * config.num_epochs - (time.time() - t0))
            logger.debug("epoch %d of %d ----- Elapsed time: %s ETA: %s", epoch + 1, num_epochs, elapsed, eta)
            logger.debug("training loss: %f" % avg_loss)

    if do_plot:
        plot_error(loss_values)
        logger.info("training loss: %f" % loss_values[-1])

    return training_predictions, training_targets


def model_testing(model, dataloader):
    predictions = []
    targets = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch_input = batch["input_seq"].to(device)
            batch_target = batch["target"].to(device)

            outputs = model(batch_input)
            predictions.extend(outputs)
            targets.extend(batch_target)

    predictions = torch.stack(predictions).cpu()
    targets = torch.stack(targets).cpu()

    return predictions, targets


class StockSentimentDataset(Dataset):
    def __init__(self, data_pre_loader: PredictionData, training=False, validation=False, testing=False, fold=None):
        self.inputs, self.targets = data_pre_loader.get_model_data(training, validation, testing, fold)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        input_seq = self.inputs[:][index]
        target_price = self.targets[index]
        return {
            "input_seq": input_seq.float(),
            "target": target_price.float()
        }


class StockLSTMNet(nn.Module):
    def __init__(self, config: NNConfig):
        super(StockLSTMNet, self).__init__()
        self.config = config
        self.input_size = config.get_num_features()
        self.output_size = config.get_num_outputs()
        self.num_layers = config.num_layers
        self.hidden_layer_size = config.hidden_layer_size

        if self.config.use_CNN:
            self.cnn_model = nn.Sequential(
                nn.Conv1d(in_channels=self.input_size, out_channels=self.config.conv_outputs,
                          kernel_size=self.config.conv_kernel, stride=self.config.conv_stride),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=self.config.maxpool_kernel, stride=self.config.maxpool_stride),
                nn.Flatten(),
                nn.Linear(in_features=self.config.get_cnn_linear_size(), out_features=self.config.model_hidden_outputs))
            self.lstm_linear = nn.Linear(in_features=self.hidden_layer_size * self.config.get_size_multiplier(),
                                         out_features=self.config.model_hidden_outputs)
            self.combined_linear = nn.Linear(in_features=self.config.model_hidden_outputs * 2,
                                             out_features=self.output_size)
        else:
            self.lstm_linear = nn.Linear(in_features=self.hidden_layer_size * self.config.get_size_multiplier(),
                                         out_features=self.output_size)

        if config.split_sentiment:
            self.stock_lstm = nn.LSTM(input_size=config.get_num_features(False, False),
                                      hidden_size=self.hidden_layer_size,
                                      num_layers=self.num_layers,
                                      bidirectional=config.bidirectional,
                                      dropout=config.dropout)
            self.sentiment_lstm = nn.LSTM(input_size=config.get_num_features(True, False),
                                          hidden_size=self.hidden_layer_size,
                                          num_layers=self.num_layers,
                                          bidirectional=config.bidirectional,
                                          dropout=config.dropout)
        else:
            self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_layer_size,
                                num_layers=self.num_layers, bidirectional=config.bidirectional,
                                dropout=config.dropout)

    def forward(self, input_sequence):
        h_in, c_in = self.generate_layers(input_sequence)
        input_formatted = input_sequence.transpose(1, 0)
        if self.config.split_sentiment:
            stock_lstm_output, (h_out, c_out) = self.stock_lstm(self.split_input(input_formatted, False),
                                                                (h_in, c_in))
            sentiment_lstm_output, (h_out, c_out) = self.sentiment_lstm(self.split_input(input_formatted, True),
                                                                        (h_in, c_in))
            lstm_output = torch.cat([stock_lstm_output[-1, :, :], sentiment_lstm_output[-1, :, :]], dim=1)

        else:
            lstm_output, (h_out, c_out) = self.lstm(input_formatted, (h_in, c_in))
            lstm_output = lstm_output[-1, :, :].view(input_sequence.size(0), -1)

        predictions = self.lstm_linear(lstm_output)

        if self.config.use_CNN:
            cnn_input_formatted = input_sequence.transpose(1, 2)
            cnn_predictions = self.cnn_model(cnn_input_formatted)
            combined_input = torch.cat([predictions, cnn_predictions], dim=1)
            predictions = self.combined_linear(combined_input)

        return predictions

    '''
    Splits the input sequence into sentiment and stock features
    '''

    def split_input(self, input_sequence, sentiment):
        inputs = []
        for index in self.config.get_feature_indexes(sentiment):
            inputs.append(input_sequence[:, :, index])
        input = torch.stack(inputs, dim=2)
        return input

    '''
    Initialises hidden and cell states of the correct size for the model
    '''

    def generate_layers(self, input_sequence):
        h_in = torch.zeros(self.num_layers * self.config.get_size_multiplier(True), input_sequence.size(0),
                           self.hidden_layer_size).type(torch.FloatTensor).to(device)
        c_in = torch.zeros(self.num_layers * self.config.get_size_multiplier(True), input_sequence.size(0),
                           self.hidden_layer_size).type(torch.FloatTensor).to(device)
        return h_in, c_in


def train_model(config):
    data_loader = PredictionData(config)
    loss_function = config.loss_function
    validation_losses = []
    logger.info("Training for \n%s", config.get_current_parameters())

    t0 = time.time()
    model = StockLSTMNet(config)
    optimizer = torch.optim.Adam(model.parameters(), config.learning_rate)
    for i in range(config.folds):
        for name, module in model.named_children():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

        model.to(device)

        training_dataset = StockSentimentDataset(data_loader, training=True, fold=i)
        train_dataloader = DataLoader(training_dataset, batch_size=config.batch_size)

        model_training(model, optimizer, train_dataloader, config.num_epochs, loss_function, config, do_plot=False)

        validation_dataset = StockSentimentDataset(data_loader, validation=True, fold=i)
        validation_dataloader = DataLoader(validation_dataset, batch_size=1)

        model_outputs, target_outputs = model_testing(model, validation_dataloader)
        loss = torch.sqrt(loss_function(model_outputs, target_outputs))
        validation_losses.append(loss)
        logger.info("Validation fold: %d loss: %f", i + 1, loss)

    # Calculates average validation loss
    total_loss = 0
    for loss in validation_losses:
        total_loss += loss
    total_loss = float(total_loss / len(validation_losses))

    elapsed = format_time(time.time() - t0)
    logger.info("Validation took: %s average loss: %f", elapsed, total_loss)

    # Saves model to file if average validation loss beats stored value
    least_loss, _ = load_var(config)
    if config.save_model and total_loss < least_loss:
        logger.info("Saving model to %s" % config.model_save_location)
        save_model(model, total_loss, config)

    return total_loss, validation_losses


def test_model(config, data_loader):
    loss_function = config.loss_function

    training_dataset = StockSentimentDataset(data_loader, training=True, testing=True)
    train_dataloader = DataLoader(training_dataset, batch_size=config.batch_size)

    model = StockLSTMNet(config)
    if config.load_model:
        logger.debug("Loading model from %s" % config.model_save_location)
        model.load_state_dict(torch.load(config.model_save_location))
        model.to(device)
        training_pred, training_targets = model_testing(model, train_dataloader)
        training_pred = training_pred.flatten()
        training_targets = training_targets.flatten()
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        model.to(device)
        training_pred, training_targets = model_training(model, optimizer, train_dataloader, config.num_epochs,
                                                         loss_function, config, do_plot=True)

    t_training_pred, training_percentages = data_loader.inverse_normalization(training_pred, start_index=0,
                                                                              end_index=len(training_pred))
    training_targets, training_target_percentages = data_loader.inverse_normalization(training_targets, start_index=0,
                                                                                      end_index=len(training_targets))

    testing_dataset = StockSentimentDataset(data_loader, testing=True)
    test_dataloader = DataLoader(testing_dataset, batch_size=1)

    model_outputs, target_outputs = model_testing(model, test_dataloader)
    testing_loss = torch.sqrt(loss_function(model_outputs, target_outputs))
    logger.info("Testing Loss: %f", testing_loss)

    t_pred_outputs, testing_percentages = data_loader.inverse_normalization(model_outputs.flatten(), start_index=-(
            len(model_outputs) + config.prediction_offset), end_index=-config.prediction_offset)
    t_target_outputs, target_percentages = data_loader.inverse_normalization(target_outputs.flatten(), start_index=-(
            len(target_outputs) + config.prediction_offset), end_index=-config.prediction_offset)

    x_test = np.arange(len(training_dataset) + config.prediction_offset,
                       len(training_dataset) + len(testing_dataset) + config.prediction_offset, 1)
    bx_test = np.arange(len(training_dataset), len(training_dataset) + len(testing_dataset) + config.prediction_offset,
                        1)
    x_train = np.arange(config.prediction_offset, len(training_dataset) + config.prediction_offset, 1)
    x_full = np.arange(config.prediction_offset,
                       len(training_dataset) + len(testing_dataset) + config.prediction_offset, 1)

    return {
        "train_target_percentages": training_target_percentages,
        "train_percentages": training_percentages,
        "test_percentages": testing_percentages,
        "target_percentages": target_percentages,
        "true_training_pred": t_training_pred,
        "train_targets": training_targets,
        "true_predicted_outputs": t_pred_outputs,
        "true_target_outputs": t_target_outputs,
        "x_train": x_train,
        "x_test": x_test,
        "x_full": x_full,
        "bx_test": bx_test,
        "test_loss": testing_loss
    }


def plot_results(results_dict, signals, config: NNConfig, prediction_data, plot_signals=False):
    buy_signals = []
    sell_signals = []
    for index, signal in enumerate(signals):
        if signal[0] == "Buy":
            buy_signals.append(index)
        elif signal[0] == "Sell":
            sell_signals.append(index)

    # Plot training plus testing stock prices
    plt.figure(figsize=(10, 5))
    plt.title('Share price per 30 min interval')
    plt.ylabel('Share Price ($)')
    plt.xlabel('Time Intervals (1 interval = 30 minutes)')
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    corrected_base_targets = results_dict["train_targets"] + results_dict["true_target_outputs"]
    plt.plot(results_dict["x_full"], corrected_base_targets, label="Targets")
    plt.plot(results_dict["x_train"], results_dict["true_training_pred"], label="Training Predictions")
    plt.plot(results_dict["x_test"], results_dict["true_predicted_outputs"], label="Test Predictions")
    plt.legend(loc="upper left")
    plt.show()

    close_prices = prediction_data.get_prices(True, False, len(results_dict["true_predicted_outputs"]))

    # Plot testing stock prices
    plt.figure(figsize=(10, 5))
    plt.title('Test Results')
    plt.ylabel('Share Price ($)')
    plt.xlabel('Time Intervals (1 interval = 30 minutes)')
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    corrected_base_targets = results_dict["train_targets"][-config.prediction_offset:] + results_dict[
        "true_target_outputs"]
    plt.plot(results_dict["bx_test"], corrected_base_targets, label="Target")
    plt.plot(results_dict["x_test"], results_dict["true_predicted_outputs"], label="Predictions")
    if plot_signals:
        plt.plot(results_dict["bx_test"][buy_signals], [corrected_base_targets[i] for i in buy_signals], "g^",
                 label="Buy Signals")
        plt.plot(results_dict["bx_test"][sell_signals], [corrected_base_targets[i] for i in sell_signals], "rs",
                 label="Sell Signals")
    plt.legend(loc="upper left")
    plt.show()

    if config.percentage_change:
        # Plot training percentage change
        plt.figure(figsize=(10, 5))
        plt.title('Training - Percentage change per 30 min interval')
        plt.ylabel('Share price percentage change')
        plt.xlabel('Time Intervals (1 interval = 30 minutes)')
        plt.grid(True)
        plt.autoscale(axis='x', tight=True)
        plt.plot(results_dict["train_target_percentages"], label="Training Targets")
        plt.plot(results_dict["train_percentages"], label="Training Predictions")
        plt.legend(loc="upper left")
        plt.show()

        # Plot testing percentage change
        plt.figure(figsize=(10, 5))
        plt.title('Testing - Percentage change per 30 min interval')
        plt.ylabel('Share price percentage change')
        plt.xlabel('Time Intervals (1 interval = 30 minutes)')
        plt.grid(True)
        plt.autoscale(axis='x', tight=True)
        plt.plot(results_dict["target_percentages"], label="Testing Targets")
        plt.plot(results_dict["test_percentages"], label="Testing Predictions")
        plt.legend(loc="upper left")
        plt.show()

    if config.predict_moving_average:
        # Plot testing stock prices
        plt.figure(figsize=(10, 5))
        plt.title('Test Results')
        plt.ylabel('Share Price ($)')
        plt.xlabel('Time Intervals (1 interval = 30 minutes)')
        plt.grid(True)
        plt.autoscale(axis='x', tight=True)
        corrected_base_targets = results_dict["train_targets"][-config.prediction_offset:] + results_dict[
            "true_target_outputs"]
        plt.plot(results_dict["bx_test"], corrected_base_targets, label="Target")
        plt.plot(results_dict["x_test"], results_dict["true_predicted_outputs"], label="Predictions")
        plt.plot(results_dict["bx_test"], close_prices, label="Close Prices")
        if plot_signals:
            plt.plot(results_dict["bx_test"][buy_signals], [corrected_base_targets[i] for i in buy_signals], "g^",
                     label="Buy Signals")
            plt.plot(results_dict["bx_test"][sell_signals], [corrected_base_targets[i] for i in sell_signals], "rs",
                     label="Sell Signals")
        plt.legend(loc="upper left")
        plt.show()

    print(signals)


def save_results(config, param_dict, loss, fold_losses):
    param_dict["loss"] = loss
    param_dict["fold_losses"] = str(fold_losses)
    param_dict["features"] = str(param_dict["features"])
    df = pd.DataFrame(param_dict, index=[0])
    df.to_csv(config.hyperparameter_save_location, mode="a", header=None, index=False)


def start():
    lowest_loss = 9999

    config = NNConfig()
    param_dict_list = config.generate_param_dict()
    if config.load_model:
        _, best_param = load_var(config)
        if best_param is None:
            logger.error("Load model set in config but no model to load")
            exit(-1)
    elif len(param_dict_list) != 1:
        t0 = time.time()
        for i, param_dict in enumerate(param_dict_list):
            config.set_parameters(param_dict)
            logger.info("Parameter set: %d/%d", i + 1, len(param_dict_list))
            validation_loss, fold_losses = train_model(config)
            save_results(config, dict(param_dict), validation_loss, fold_losses)
            if validation_loss < lowest_loss:
                lowest_loss = validation_loss
                best_param = param_dict
            elapsed = format_time(time.time() - t0)
            eta = format_time(((time.time() - t0) / (i + 1)) * (len(param_dict_list) - (i + 1)))
            logger.info("Elapsed: %s eta: %s", elapsed, eta)

        logger.info("Best parameters were: %s", str(best_param))
        logger.info("Validation Loss: %f", lowest_loss)
    else:
        best_param = param_dict_list[0]

    config.set_parameters(best_param)
    logger.info("Testing for \n%s", config.get_current_parameters())
    data_loader = PredictionData(config)
    best_result = test_model(config, data_loader)

    strategy_calc = StrategyCalculator(config, data_loader)
    buy_and_hold_return = strategy_calc.get_buy_and_hold_return(best_result["true_target_outputs"])
    model_return, signals = strategy_calc.get_model_return(best_result["true_predicted_outputs"])
    logger.info("B&H return: %f", buy_and_hold_return)
    logger.info("Model return: %f", model_return)

    plot_results(best_result, signals, config, data_loader)


start()
