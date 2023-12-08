import os
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.utils import resample
from tensorflow import keras as keras 
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import json
import pickle 
from preprocess import create_train_valid_test_data

# Load the model
def load_trained_model(model_path):
    model = load_model(model_path)
    print("Model loaded successfully.")
    return model

# Load training history
def load_training_history(history_path):
    with open(history_path, 'rb') as file:
        history = pickle.load(file)
    print("History object loaded successfully.")
    return history

# Print training and validation losses and accuracies
def print_training_summary(history):
    for epoch in range(len(history['accuracy'])):
        print(f"Epoch {epoch+1}/{len(history['accuracy'])}")
        print(f"Training loss: {history['loss'][epoch]}")
        print(f"Training accuracy: {history['accuracy'][epoch]}")
        print(f"Validation loss: {history['val_loss'][epoch]}")
        print(f"Validation accuracy: {history['val_accuracy'][epoch]}")

# [Include other functions like TrainingHistoryPlotter, GradCAM, etc.]
class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        # store the model, the class index used to measure the class
        # activation map, and the layer to be used when visualizing
        # the class activation map
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        for layer in reversed(self.model.layers):
            # check to see if the layer has a 4D output
            if len(layer.output_shape) == 4:
                return layer.name
        # otherwise, we could not find a 4D layer so the GradCAM
        # algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")


    def compute_heatmap(self, image, eps=1e-8):
        # construct our gradient model by supplying (1) the inputs
        # to our pre-trained model, (2) the output of the (presumably)
        # final 4D layer in the network, and (3) the output of the
        # softmax activations from the model
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output, self.model.output])

        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass the
            # image through the gradient model, and grab the loss
            # associated with the specific class index
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)

            loss = predictions[:, tf.argmax(predictions[0])]

        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)

        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
        # return the resulting heatmap to the calling function
        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5,
                        colormap=cv2.COLORMAP_VIRIDIS):
        # apply the supplied color map to the heatmap and then
        # overlay the heatmap on the input image
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        # return a 2-tuple of the color mapped heatmap and the output,
        # overlaid image
        return (heatmap, output)
    def __init__(self, model, classIdx, layerName=None):
        # store the model, the class index used to measure the class
        # activation map, and the layer to be used when visualizing
        # the class activation map
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        for layer in reversed(self.model.layers):
            # check to see if the layer has a 4D output
            if len(layer.output_shape) == 4:
                return layer.name
        # otherwise, we could not find a 4D layer so the GradCAM
        # algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")


    def compute_heatmap(self, image, eps=1e-8):
        # construct our gradient model by supplying (1) the inputs
        # to our pre-trained model, (2) the output of the (presumably)
        # final 4D layer in the network, and (3) the output of the
        # softmax activations from the model
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output, self.model.output])

        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass the
            # image through the gradient model, and grab the loss
            # associated with the specific class index
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)

            loss = predictions[:, tf.argmax(predictions[0])]

        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)

        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
        # return the resulting heatmap to the calling function
        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5,
                        colormap=cv2.COLORMAP_VIRIDIS):
        # apply the supplied color map to the heatmap and then
        # overlay the heatmap on the input image
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        # return a 2-tuple of the color mapped heatmap and the output,
        # overlaid image
        return (heatmap, output)

class TrainingHistoryPlotter:
    def __init__(self, history):
        """
        Initialize the plotter with training history.
        :param history: A history object from the training session.
        """
        self.history = history

    def plot_accuracy(self):
        """
        Plots the training and validation accuracy.
        """
        plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
        plt.plot(self.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()  # Adjust the layout
        plt.show()

    def plot_loss(self):
        """
        Plots the training and validation loss.
        """
        plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
        plt.plot(self.history['loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()  # Adjust the layout
        plt.show()

    # If learning rate was recorded, add a method to plot it
    def plot_learning_rate(self):
        """
        Plots the learning rate over epochs (if it's part of the history).
        """
        if 'lr' in self.history:
            plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
            plt.plot(self.history['lr'], label='Learning Rate')
            plt.title('Learning Rate over Time')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.legend()
            plt.tight_layout()  # Adjust the layout
            plt.show()
        else:
            print("Learning rate not found in history.")

# Load and preprocess a single image
def load_and_preprocess_image(img_path, target_size=(256, 256)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch
    img_array /= 255.0  # Rescale the image
    return img_array, img

# Helper function to check if the prediction is correct
def isCorrect(str1, str2):
    it = iter(str2)
    return all(c in it for c in str1)

# Helper function to check if the prediction is correct based on the filename
def normalizeFileName(filename, predicted_class):
    # Normalize strings for comparison: lowercase and replace underscores with spaces
    normalized_filename = filename.lower().replace('.jpg', '').replace('.jpeg', '')
    normalized_filename = normalized_filename[:-1]
    normalized_predicted_class = predicted_class.lower().replace('_', ' ').replace(' ','')
    return normalized_filename, normalized_predicted_class

# Function to plot images and their predicted classes
def plot_predictions(images, filenames, predictions, max_images_per_grid=9):
    """Plots images with their predictions in multiple 3x3 grids, with correct predictions in green and incorrect in red."""
    num_images = len(images)
    num_grids = math.ceil(num_images / max_images_per_grid)
    successCount = 0

    for grid_num in range(num_grids):
        start_idx = grid_num * max_images_per_grid
        end_idx = start_idx + max_images_per_grid
        subset_images = images[start_idx:end_idx]
        subset_filenames = filenames[start_idx:end_idx]
        subset_predictions = predictions[start_idx:end_idx]

        num_cols = 3
        num_rows = math.ceil(len(subset_images) / num_cols)

        plt.figure(figsize=(3 * num_cols, 2 * num_rows))
        for i, (img, filename, pred) in enumerate(zip(subset_images, subset_filenames, subset_predictions)):
            plt.subplot(num_rows, num_cols, i + 1)
            plt.imshow(img)
            normalized_filename, normalized_predicted_class = normalizeFileName(filename, pred)
            correct = isCorrect(normalized_filename, normalized_predicted_class)
            if correct:
                successCount += 1

            #print(f"filename: {normalized_filename}, pred: {normalized_predicted_class}")
            title_color = 'green' if correct else 'red'
            plt.title(f"Pred: {pred}\n(True: {normalized_filename})", color=title_color, fontsize=8)
            plt.axis('off')
        plt.tight_layout()
        plt.show()
        print('\n\n')

    return successCount

def plot_confidence_intervals(accuracies):
    """
    Plots the bootstrap confidence intervals for accuracies.

    :param accuracies: List or array of accuracy values.
    :param n_boots: Number of bootstrap samples to draw.
    :param confidence_level: Confidence level for the intervals.
    """
    bootstrap_accuracies = []
    n_boots=1000
    confidence_level=0.95
    # Perform bootstrap sampling
    for _ in range(n_boots):
        boot = resample(accuracies)
        boot_mean = np.mean(boot)
        bootstrap_accuracies.append(boot_mean)

    # Calculate the percentiles for the confidence interval
    alpha = 1 - confidence_level
    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)
    confidence_lower = np.percentile(bootstrap_accuracies, lower_percentile)
    confidence_upper = np.percentile(bootstrap_accuracies, upper_percentile)

    print(f"95% Confidence interval for the accuracy: [{confidence_lower:.2f}, {confidence_upper:.2f}]")

    # Plotting
    plt.hist(bootstrap_accuracies, bins=30, color='skyblue', alpha=0.7, edgecolor='black')
    plt.axvline(x=confidence_lower, color='red', linestyle='dashed', linewidth=2, label=f'Lower 95% CI ({confidence_lower:.2f})')
    plt.axvline(x=confidence_upper, color='green', linestyle='dashed', linewidth=2, label=f'Upper 95% CI ({confidence_upper:.2f})')
    plt.title('Bootstrap Accuracies and Confidence Intervals')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    
# Compute and plot confusion matrix
def plot_confusion_matrix(valid_gen):
    # Confusion Matrix

    # Collect all the true labels and predictions
    true_labels = []
    predictions = []

    for images, labels in valid_gen:
        preds = model.predict(images)
        preds = np.argmax(preds, axis=1)

        # If labels are already in integer form, no need for argmax
        if labels.ndim > 1:
            true = np.argmax(labels, axis=1)
        else:
            true = labels

        true_labels.extend(true)
        predictions.extend(preds)


    # Compute confusion matrix
    conf_mat = confusion_matrix(true_labels, predictions)
    fig, ax = plt.subplots(figsize=(10,10))  # Adjust size as needed
    sns.heatmap(conf_mat, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, cbar=False)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    plt.show()

# Plot ROC curves
def plot_roc_curves(valid_gen, classes_names):
    # [Function implementation]
    true_labels = []
    pred_probs = []

    for images, labels in valid_gen:
        probs = model.predict(images)
        if labels.ndim > 1:
            true = np.argmax(labels, axis=1)
        else:
            true = labels

        true_labels.extend(true)
        pred_probs.extend(probs)
    
    true_labels = np.array(true_labels)
    pred_probs = np.array(pred_probs)

    true_labels_bin = label_binarize(true_labels, classes=range(len(classes_names)))

    # Calculate ROC-AUC for each class
    roc_auc_scores = {}
    for i, class_name in enumerate(classes_names):
        # Ensure we're indexing properly
        class_true_labels = true_labels_bin[:, i]
        class_pred_probs = pred_probs[:, i]

        # Calculate ROC-AUC for the current class
        roc_auc_scores[class_name] = roc_auc_score(class_true_labels, class_pred_probs)

    # Print ROC-AUC scores for each class
    for class_name, roc_auc in roc_auc_scores.items():
        print(f"{class_name}: {roc_auc:.3f}")

    # You can also calculate the micro-average and macro-average ROC-AUC across all classes
    micro_avg_roc_auc = roc_auc_score(true_labels_bin, pred_probs, average="micro")
    macro_avg_roc_auc = roc_auc_score(true_labels_bin, pred_probs, average="macro")

    print(f"Micro-average ROC-AUC: {micro_avg_roc_auc:.3f}")
    print(f"Macro-average ROC-AUC: {macro_avg_roc_auc:.3f}")
    
    # Assuming true_labels_bin and pred_probs are already defined and are numpy arrays
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Calculate ROC curve and ROC area for each class
    for i in range(len(classes_names)):
        fpr[i], tpr[i], _ = roc_curve(true_labels_bin[:, i], pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(true_labels_bin.ravel(), pred_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    #print(fpr)
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes_names))]))

    # Then interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes_names)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(classes_names)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    # Plot each individual ROC curve
    for i in range(len(classes_names)):
        plt.plot(fpr[i], tpr[i], lw=1)

    # Plot micro and macro average ROC curves
    plt.plot(fpr["micro"], tpr["micro"], label='Micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]), color='deeppink', linestyle=':', linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"], label='Macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]), color='navy', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve')

    # Place a legend to the right of the plot
    plt.legend(loc="lower right", bbox_to_anchor=(1.25, 0.3))

    plt.show()

# Load trial data and plot changes in hyperparameters
def plot_hyperparameter_changes():
    # Function to load the trial data from JSON files
    def load_trial_data(file_paths):
        trial_data = []
        for file_path in file_paths:
            with open(file_path, 'r') as file:
                data = json.load(file)
                trial_data.append(data)
        return trial_data

    # Assuming the JSON files are saved in the following paths
    file_paths = ['trial0.json', 'trial2.json', 'trial3.json']

    # Load the trial data
    data = load_trial_data(file_paths)
    #print(data)

    # Plot the change of hyperparameters over time
    plt.figure(figsize=(20, 6))

    # Assuming the JSON files have a structure that includes 'hyperparameters' and their 'values'
    # Extract hyperparameter values for plotting
    learning_rates = [trial['hyperparameters']['values']['learning_rate'] for trial in data]
    dropouts = [trial['hyperparameters']['values']['dropout_rate'] for trial in data]
    l2_regs = [trial['hyperparameters']['values']['l2_reg'] for trial in data]

    # Plotting each hyperparameter change
    plt.subplot(1, 3, 1)
    plt.plot(learning_rates, label='Learning Rate')
    plt.xlabel('Trial')
    plt.ylabel('Value')
    plt.title('Learning Rate over Trials')
    plt.grid()
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(dropouts, label='Dropout Rate')
    plt.xlabel('Trial')
    plt.ylabel('Value')
    plt.title('Dropout Rate over Trials')
    plt.grid()
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(l2_regs, label='L2 Regularization')
    plt.xlabel('Trial')
    plt.ylabel('Value')
    plt.title('L2 Regularization over Trials')
    plt.grid()
    plt.legend()

    plt.tight_layout()

    plt.show()

def plot_Grad_CAM():
    layer_name = 'conv2d_119'

    # Load and preprocess an image
    img_path = '/content/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/Corn_(maize)___Northern_Leaf_Blight/0118e276-ee7b-4fed-961d-057590ae9f7f___RS_NLB 4666 copy 2.jpg'
    img_array, img = load_and_preprocess_image(img_path)
    pred = model.predict(img_array)
    pred_class = classes_names[np.argmax(pred)]

    # Instantiate a GradCAM object
    gradcam = GradCAM(model, 0, layer_name)

    # Compute heatmap
    heatmap = gradcam.compute_heatmap(img_array)

    # Use OpenCV to load the original image
    orig = cv2.imread(img_path)

    # Resize heatmap to the size of the original image
    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))

    # Apply the heatmap to the original image
    (heatmap, output) = gradcam.overlay_heatmap(heatmap, orig, alpha=0.5)

    # Display results
    plt.imshow(output)
    plt.show()

# Main execution block
if __name__ == '__main__':
    model_path = 'plant_disease_resnet18_model_epoch_10_v2.h5'
    history_path = 'model_epoch_10_v2_history.pkl'
    test_images_dir = 'new-plant-diseases-dataset/test/test/'
    train_gen, valid_gen, classes_names = create_train_valid_test_data()[-1]

    model = load_trained_model(model_path)
    history = load_training_history(history_path)
    print_training_summary(history)

    plotter = TrainingHistoryPlotter(history)
    plotter.plot_accuracy()
    plotter.plot_loss()

    # [Add other functionalities such as prediction visualization, Grad-CAM, etc.]
    # [Ensure to call appropriate functions and handle data as needed]
    # Load and predict images, then collect their actual data and predictions
    test_images = []
    test_preds = []
    fileNames = []
    successCount = 0
    test_images_dir = 'new-plant-diseases-dataset/test/test/'
    
    for img_name in os.listdir(test_images_dir):
        if img_name.lower().endswith('.jpg') or img_name.lower().endswith('.jpeg'):
            fileNames.append(img_name)
            img_path = os.path.join(test_images_dir, img_name)
            img_array, img = load_and_preprocess_image(img_path)
            pred = model.predict(img_array)
            pred_class = classes_names[np.argmax(pred)]
            test_images.append(img)
            test_preds.append(pred_class)
    successCount = plot_predictions(test_images, fileNames, test_preds)
    print(f"Success Rate: {successCount} / {len(fileNames)}")
    
    plot_Grad_CAM()
    plot_hyperparameter_changes()
    plot_confusion_matrix(valid_gen=valid_gen)
    plot_roc_curves(valid_gen=valid_gen, classes_names=classes_names)
    plot_confidence_intervals(accuracies=history['val_accuracy'])
    
    
    