import time

from tensorflow import convert_to_tensor
import gc
import tensorflow as tf
from tensorflow import keras

from numerapi import NumerAPI
from halo import Halo
from utils import *


start = time.time()
napi = NumerAPI()
spinner = Halo(text='', spinner='dots')
current_round = napi.get_current_round(tournament=8)  # tournament 8 is the primary Numerai Tournament

download_data()

# read the feature metadata and get the "small" feature set (We will need a data center grade server if we want to
# process all the Numerai features!)
with open("features.json", "r") as f:
    feature_metadata = json.load(f)
features = feature_metadata["feature_sets"]["medium"]

# read the training and validation data given the predefined features stored in parquets as pandas DataFrames
training_data, validation_data = read_learning_data(features)
# extract feature matrix and target vector used for training
X_train = training_data.filter(like='feature_', axis='columns')
y_train = training_data[TARGET_COL]
# extract feature matrix and target vector used for validation
X_val = validation_data.filter(like='feature_', axis='columns')
y_val = validation_data[TARGET_COL]
# "garbage collection" (gc) gets rid of unused data and frees up memory
gc.collect()


########################################################################################################################
# define and train your model here using the loaded data sets!
model_name = "nn"
NN_model = keras.models.Sequential()
num_feature_neutralization = 50
input_dimension = len(features)

initializer = tf.keras.initializers.HeNormal()

# The Input Layer :
NN_model.add(keras.layers.Dense(input_dimension, kernel_initializer=initializer, input_dim = len(features), activation='selu'))
NN_model.add(keras.layers.Dropout(0.2))

# The Hidden Layers :
layer_neurons = input_dimension
while layer_neurons > 4:
    if layer_neurons > 100:
        dropout = 0.3
        layer_neurons //= 2
    elif layer_neurons > 10:
        dropout = 0.2
        layer_neurons //= 4
    else:
        dropout = 0.1
        layer_neurons //= 4
    NN_model.add(keras.layers.Dense(layer_neurons, kernel_initializer=initializer, activation='selu', bias_regularizer=keras.regularizers.l2(1e-5)))
    NN_model.add(keras.layers.Dropout(dropout))

# The Output Layer :
NN_model.add(keras.layers.Dense(1, kernel_initializer='random_uniform', activation='linear'))

# Compile the network :
NN_model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])

spinner.start('Training model')
X_train = convert_to_tensor(X_train)

# Early stopping callback function
es = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=2000, restore_best_weights=True)

# Train the neural network
NN_model.fit(X_train, y_train, epochs=10000, batch_size=X_train.shape[0]//100, shuffle=True, callbacks=[es])
# Save weights as checkpoint
NN_model.save_weights('./checkpoint')


spinner.succeed()
gc.collect()

########################################################################################################################


spinner.start('Predicting on validation data')
# here we insert the predictions back into the validation data set, as we need to use the validation data set to
# perform feature neutralization later
X_val = convert_to_tensor(X_val)
validation_data.loc[:, f"preds_{model_name}"] = NN_model.predict(X_val)
spinner.succeed()
gc.collect()

spinner.start('Neutralizing to risky features')
# neutralize our predictions to the k riskiest features in the training set
neutralize_riskiest_features(training_data, validation_data, features, model_name, k=num_feature_neutralization)
spinner.succeed()
gc.collect()

print('Exporting Predictions to csv...')
validation_data["prediction"] = validation_data[f"preds_{model_name}_neutral_riskiest_{num_feature_neutralization}"]\
    .rank(pct=True)
validation_data["prediction"].to_csv(f"validation_predictions_{model_name}.csv")
print('Done!')

print(f'Time elapsed: {(time.time() - start) / 60} mins')

