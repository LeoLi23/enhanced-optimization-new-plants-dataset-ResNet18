from models import ResNet18
from preprocess import create_train_valid_test_data
from tensorflow import keras as keras
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import hyper_tune

def train():
    # Load the preprocessed training, validation, and test data
    train_gen, valid_gen, _ = create_train_valid_test_data()

    # Instantiate and compile the ResNet18 model
    model = hyper_tune.tune_hyperparameters(train_gen, valid_gen)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)
    # Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Fit the model
    history = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=10,
        callbacks=[reduce_lr, early_stopping]
    )
    
    # Save Model
    model.save('model_history_checkpoints/plant_disease_resnet18_model_epoch_10_v2.h5')
    # Save History 
    import pickle

    with open('model_epoch_10_v2_history.pkl', 'wb') as file:
        pickle.dump(history.history, file)
    print("History object saved.")
    
    '''
    from keras.models import load_model
    import pickle

    # Load the model
    model = load_model('plant_disease_resnet18_model_epoch_10_v2.h5')
    print("Model loaded successfully.")

    # Load the history
    with open('model_epoch_10_v2_history.pkl', 'rb') as file:
        history = pickle.load(file)
    print("History object loaded successfully.")
    '''

if __name__ == '__main__':
    train()
