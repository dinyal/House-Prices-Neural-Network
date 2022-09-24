import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
from torch import dropout, nn
from torch.optim.lr_scheduler import ExponentialLR
import pickle
import numpy as np
from numpy.random import default_rng
import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
# Was needed to store outputs in json
# import graphs


class Regressor(nn.Module):
    def __init__(
        self,
        x=None,
        nb_epoch=300,
        size_of_batches=64,
        hidden_layer_2=40,
        hidden_layer_3=16
    ):

        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """
        Initialise the model.

        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape
                (batch_size, input_size), used to compute the size
                of the network.
            - nb_epoch {int} -- number of epoch to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        super().__init__()

        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.input_size = 12
        self.hidden_layer_2 = hidden_layer_2
        self.hidden_layer_3 = hidden_layer_3
        self.output_size = 1
        self.nb_epoch = nb_epoch
        self.size_of_batches = size_of_batches

        self.param_grid = {
            "nb_epoch": [300, 400, 500],
            "size_of_batches": [32, 64, 128],
            "hidden_layer_2": [10, 20, 40],
            "hidden_layer_3": [8, 16, 24],

        }

        # will trigger grid search to report negative RMSE
        # as the gridsearch function chooses based on highest
        # reported value from score. 
        self.in_grid_search = False

        # store for the model that we want to create
        self.model = None

        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def instantiate_model(self):
        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_layer_2),
            nn.ReLU(),
            nn.Linear(self.hidden_layer_2, self.hidden_layer_3),
            nn.ReLU(),
            nn.Linear(self.hidden_layer_3, self.output_size),
        )

    def shuffle_data(self, x, y):
        shuffled_indices = default_rng().permutation(len(x))
        x_shuffled = x[shuffled_indices]
        y_shuffled = y[shuffled_indices]
        return x_shuffled, y_shuffled

    # get_params method implemented in estimator to make gridsearchCV function
    def get_params(self, deep=True):
        return {
            "nb_epoch": self.nb_epoch,
            "size_of_batches": self.size_of_batches,
            "hidden_layer_2": self.hidden_layer_2,
            "hidden_layer_3": self.hidden_layer_3
        }

    # set params method for gridsearchCV function
    def set_params(
        self,
        nb_epoch,
        size_of_batches,
        hidden_layer_2,
        hidden_layer_3
    ):

        self.nb_epoch = nb_epoch
        self.size_of_batches = size_of_batches
        self.hidden_layer_2 = hidden_layer_2
        self.hidden_layer_3 = hidden_layer_3
        self.in_grid_search = True

        return self

    def _preprocessor(self, x, y=None, training=False):
        """
        Preprocess input of the network.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size).
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # This filters out certain columns in Score, but not in predict.
        if y is not None:
            # merge the two, drop the outlying house values
            combined = pd.merge(x, y, left_index=True, right_index=True)
            combined = combined[combined["median_house_value"] != 500001]

            # create our x and y again
            x = combined.drop(columns=["median_house_value"])
            y = pd.DataFrame(combined["median_house_value"])

        # We're going to fill the nan values with the average value of the column
        x = x.fillna(x.mean())
        # getting the per-household values as these are more insightful
        x["total_rooms"] = x["total_rooms"] / x["households"]
        x["total_bedrooms"] = x["total_bedrooms"] / x["households"]
        x["population"] = x["population"] / x["households"]

        # We believe that the households column confers no useful information for the model
        # so we're dropping it, having used it to get per-household figures elsewhere
        x.drop(columns=["households"], inplace=True)

        x.rename(
            columns={
                "total_rooms": "rooms_per_household",
                "total_bedrooms": "bedrooms_per_household",
                "population": "residents_per_household",
            },
            inplace=True,
        )

        # add one hot encoding to allow the model to work with text categories
        # First create a list of strings containing all possible values in the ocean_proximity column
        # this is so that if a dataset doesn't contain some of the values by chance, the 
        # processed matrix still contains the right number of columns

        op_categories = ['area_<1H OCEAN', 'area_INLAND', 'area_ISLAND', 'area_NEAR BAY', 'area_NEAR OCEAN']

        one_hot_area = pd.get_dummies(x["ocean_proximity"], prefix="area")
        one_hot_area = one_hot_area.reindex(columns=op_categories, fill_value=0)

        x = x.drop(["ocean_proximity"], axis=1)

        # Scale x
        if training:
            x = self.x_scaler.fit_transform(x)
        else:
            x = self.x_scaler.transform(x)

        # Scale y
        if y is not None and training:
            y = self.y_scaler.fit_transform(y)

        elif y is not None:
            y = self.y_scaler.transform(y)

        x = pd.DataFrame(x)
        one_hot_area.reset_index(inplace=True)
        one_hot_area = one_hot_area.drop(["index"], axis=1)

        x = pd.merge(x, one_hot_area, left_index=True, right_index=True)
        x = x.to_numpy()

        # Return preprocessed x and y
        return x, y

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y=y, training=True)  # Do not forget
        # set input size of neural network
        self.input_size = X.shape[1]
        self.instantiate_model()

        # separate out a validation set to use for checking loss at each epoch.
        x_train, x_val, y_train, y_val = train_test_split(
            X, Y, test_size=0.25, random_state=42
        )

        # choose model optimiser, lr decay function, and loss function
        optimiser = torch.optim.Adam(self.model.parameters())

        # Initially chose to use a scheduler for learning rate decay, but removed
        # after using Adam optimiser
        # scheduler = ExponentialLR(optimiser, gamma=0.9)  # for lr decay, now deprecated

        # Select chosen loss function
        criterion = torch.nn.MSELoss()

        # calculate the number of batches required based on desired batch size
        if len(x_train) <= self.size_of_batches:
            number_of_batches = 1
        else:
            number_of_batches = len(x_train) // self.size_of_batches

        # keep track of minimum loss and stop if exceeds a certain
        # number of epochs without reducing loss
        min_loss = float("inf")
        early_stop_counter = 0

        for epoch in range(self.nb_epoch):
            # shuffle split the dataset into specific number of batches
            x_train, y_train = self.shuffle_data(x_train, y_train)

            # create batches to be iterated through
            X_batches = np.array_split(x_train, number_of_batches)
            Y_batches = np.array_split(y_train, number_of_batches)

            for batch_no in range(number_of_batches):
                X_batches[batch_no] = torch.from_numpy(X_batches[batch_no]).float()
                Y_batches[batch_no] = torch.from_numpy(Y_batches[batch_no]).float()

                # Reset the gradients
                optimiser.zero_grad()
                # forward pass
                y_hat = self.model(X_batches[batch_no])
                # compute loss
                loss = criterion(y_hat, Y_batches[batch_no])
                # Backward pass (compute the gradients)
                loss.backward()
                # update parameters
                optimiser.step()

            # Initially chose to use a scheduler for learning rate decay, but removed
            # after using Adam optimiser
            # scheduler.step()  # this was for the learning rate decay, now deprecated

            # Check loss on validation set.
            x_val_tensor = torch.from_numpy(x_val).float()
            y_val_tensor = torch.from_numpy(y_val).float()

            y_predictions = self.model(x_val_tensor)

            epoch_loss = criterion(y_predictions, y_val_tensor)

            #if epoch % 10 == 0:
                #print("Epoch ", epoch, f" Loss: {epoch_loss:.4f}")

           
            # Check for improving loss on validation set. If hit earlys stopping limit, 
            # terminate loop
            if epoch_loss < min_loss:
                min_loss = epoch_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter == 50:
                return self

        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).

        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        X, _ = self._preprocessor(x, training=False)  # Do not forget
        X = torch.from_numpy(X).float()

        normalised_y_predictions = self.model(X)

        # denormalise the predictions, to get something useful for comparisons
        y_predictions = self.y_scaler.inverse_transform(
            normalised_y_predictions.detach().numpy()
        )

        return y_predictions

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        y_hat = pd.DataFrame(self.predict(x))
        rmse_loss = mean_squared_error(y, y_hat) ** 0.5

        #=======================================================================
        #               FOR PLOTTING. COMMENT OUT AFTER USE
        #=======================================================================
        #result = {'epochs': self.nb_epoch, 'batchSize' : self.size_of_batches, 
        #'neurons': [self.hidden_layer_2, self.hidden_layer_3], 'regressionError' : rmse_loss }

        #graphs.PlotDataCollector.saveTuningResults(result, "resultsB.json")
        #=======================================================================
        #                       END PLOT
        #=======================================================================
        
        # Printout to see results and for which hyperparameters
        print("Finished tuning. Results: ")
        print(
            "   With params set to: Epochs: ",
            self.nb_epoch,
            ", Batch Size: ",
            self.size_of_batches,
            ", and others: ",
            self.hidden_layer_2,
            self.hidden_layer_3,
        )
        
        if self.in_grid_search == False:
            print("       Score on test set: ", rmse_loss)
            return rmse_loss
        else:
            print("       Score on test set: ", rmse_loss*-1)
            return rmse_loss*-1

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model):
    """
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open("part2_model.pickle", "wb") as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor():
    """
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open("part2_model.pickle", "rb") as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def RegressorHyperParameterSearch(x_train, y_train):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.

    Returns:
        The function should return your optimised hyper-parameters.

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    regressor = Regressor(x_train)

    # Perform 4 fold cross validation to minimise overfitting. 
    hyperparameter_tuner = GridSearchCV(regressor, regressor.param_grid, n_jobs=-1, cv=4)

    hyperparameter_tuner.fit(x_train, y_train)
    print(hyperparameter_tuner.best_params_)

    # Save the best estimator with the best hyperparameters
    best_model = hyperparameter_tuner.best_estimator_

    save_regressor(best_model)

    return  hyperparameter_tuner.best_params_

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


def example_main():
    

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv")

    # Spliting input and output
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42
    )

    # Create the regressor model
    regressor = Regressor(x_train)

    # fit the model based on our held out training set
    regressor.fit(x_train, y_train)

    # Error
    error = regressor.score(x_test, y_test)
    print("\nRegressor error: {}\n".format(error))

    # RegressorHyperParameterSearch(x_train, y_train)
    best_regressor = load_regressor()
    print("In saved best model, grid search: ", best_regressor.in_grid_search)
    print("In saved best model, params: ", best_regressor.nb_epoch, best_regressor.hidden_layer_2, best_regressor.hidden_layer_3)
    error = best_regressor.score(x_test, y_test)
    print("\nBest Regressor error: {}\n".format(error))

if __name__ == "__main__":
    example_main()
