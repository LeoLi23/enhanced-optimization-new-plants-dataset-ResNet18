import keras_tuner as kt
from tensorflow import keras as keras 
from keras.callbacks import EarlyStopping
from models import ResNet18Model  # Assuming ResNet18Model is in models.py

# Define the build model function that the tuner will use
def build_model(hp):
    model = ResNet18Model(input_shape=(256, 256, 3), classes=38)
    model.set_hyperparameters(hp)
    return model.build()

# Create a tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=3,
    executions_per_trial=1,
    directory='random_search_2',
    project_name='plant_disease_classification_2'
)

# Function to perform hyperparameter tuning
def tune_hyperparameters(train_gen, valid_gen):
    # Display search space summary
    tuner.search_space_summary()

    # Perform the hyperparameter search
    tuner.search(train_gen,
                 validation_data=valid_gen,
                 epochs=5,
                 callbacks=[EarlyStopping(monitor='val_loss', patience=3)])

    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Print all the best hyperparameters
    print("Best Hyperparameters Found:")
    for hyperparam, value in best_hps.values.items():
        print(f"{hyperparam}: {value}")

    # Build the model with the best hyperparameters
    best_model = build_model(best_hps)
    return best_model
