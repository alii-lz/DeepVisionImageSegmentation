# Import modules
from image_tools import create_split, convert_label_to_image
import os
import cv2
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Tensorflow modules
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Cropping2D, Dropout, Conv2DTranspose
from tensorflow.keras.metrics import MeanIoU, OneHotMeanIoU, OneHotIoU
from tensorflow.keras.callbacks import Callback
from keras.utils import to_categorical


model_path = 'segmentation_unet_model.keras'

original_image_height = 1512
original_image_width = 2016

# Downsample the image 3x
image_height = int(1512 / 3)
image_width = int(2016 / 3)

num_classes=18

images_path = "PathToDataSet"

# Unet model with a default size of the full images with the 3 channels for RGB
def Unet(input_size=(1512, 2016, 3), num_classes=num_classes, num_filters_multiplier=4):
    
    # Intialize a keras tensor
    inputs = Input(input_size)
    
    # __Encoder__
    # Encoder Layer 1
    # Apply a 2D convolution with 64 filters, each of size 3x3
    # Dropout layer to randomly set 10% of the neurons to 0 so we can prevent overfitting
    conv1 = Conv2D(16 * num_filters_multiplier, 3, activation='relu', kernel_initializer='he_normal',padding='same')(inputs)
    conv1 = Dropout(0.1)(conv1)
    conv1 = Conv2D(16 * num_filters_multiplier, 3, activation='relu', kernel_initializer='he_normal', padding='same')(conv1)
    
    # Encoder Layer 2
    # Downsample input by taking the maximum value over a 3x3 pool size
    pool1 = MaxPooling2D(pool_size=(3, 3))(conv1)
    conv2 = Conv2D(32 * num_filters_multiplier, 3, activation='relu', kernel_initializer='he_normal', padding='same')(pool1)
    conv2 = Dropout(0.1)(conv2)
    conv2 = Conv2D(32 * num_filters_multiplier, 3, activation='relu', kernel_initializer='he_normal', padding='same')(conv2)

    # Encoder Layer 3
    # reason for increasing dropout: as the network goes deeper and has more filters
    # it's capacity to learn complex features increases. So by gradually increasing
    # dropout rate, we add more regularization in the deeper layers, preventing overfitting
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(64 * num_filters_multiplier, 3, activation='relu', kernel_initializer='he_normal', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(64 * num_filters_multiplier, 3, activation='relu', kernel_initializer='he_normal', padding='same')(conv3)

    # Encoder Layer 4
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(128 * num_filters_multiplier, 3, activation='relu', kernel_initializer='he_normal', padding='same')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(128 * num_filters_multiplier, 3, activation='relu', kernel_initializer='he_normal', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # Bottleneck
    conv5 = Conv2D(256 * num_filters_multiplier, 3, activation='relu', kernel_initializer='he_normal', padding='same')(pool4)
    conv5 = Dropout(0.3)(conv5)
    conv5 = Conv2D(256 * num_filters_multiplier, 3, activation='relu', kernel_initializer='he_normal', padding='same')(conv5)
    
    # __Decoder__
    # Decoder Layer 1
    # Upsample input by a factor of 2 in both dimensions
    # Concatenates upsampled features with the corresponding encoder features to retain high resolution features
    up6 = Conv2DTranspose(128 * num_filters_multiplier, (2, 2), strides=(2, 2), padding='same')(conv5)
    up6 = concatenate([up6, conv4])
    conv6 = Conv2D(128 * num_filters_multiplier, 3, activation='relu', kernel_initializer='he_normal', padding='same')(up6)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(128 * num_filters_multiplier, 3, activation='relu', kernel_initializer='he_normal', padding='same')(conv6)
    
    # Decoder Layer 2
    up7 = Conv2DTranspose(64 * num_filters_multiplier, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = concatenate([up7, Cropping2D(cropping=((0, abs(conv3.shape[1] - up7.shape[1])), (0, abs(conv3.shape[2] - up7.shape[2]))))(conv3)])
    conv7 = Conv2D(64 * num_filters_multiplier, 3, activation='relu', kernel_initializer='he_normal', padding='same')(up7)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(64 * num_filters_multiplier, 3, activation='relu', kernel_initializer='he_normal', padding='same')(conv7)
    
    # Decoder Layer 3
    up8 = Conv2DTranspose(32 * num_filters_multiplier, (2, 2), strides=(2, 2), padding='same')(conv7)
    up8 = concatenate([up8, Cropping2D(cropping=((0, abs(conv2.shape[1] - up8.shape[1])), (0, abs(conv2.shape[2] - up8.shape[2]))))(conv2)])
    conv8 = Conv2D(32 * num_filters_multiplier, 3, activation='relu', kernel_initializer='he_normal', padding='same')(up8)
    conv8 = Dropout(0.1)(conv8)
    conv8 = Conv2D(32 * num_filters_multiplier, 3, activation='relu', kernel_initializer='he_normal', padding='same')(conv8)
    
    # Decoder Layer 4
    up9 = Conv2DTranspose(16 * num_filters_multiplier, (3, 3), strides=(3, 3), padding='same')(conv8)
    up9 = concatenate([up9, Cropping2D(cropping=((0, abs(conv1.shape[1] - up9.shape[1])), (0, abs(conv1.shape[2] - up9.shape[2]))))(conv1)])
    conv9 = Conv2D(16 * num_filters_multiplier, 3, activation='relu', kernel_initializer='he_normal', padding='same')(up9)
    conv9 = Dropout(0.1)(conv9)
    conv9 = Conv2D(16 * num_filters_multiplier, 3, activation='relu', kernel_initializer='he_normal', padding='same')(conv9)
    
    # Convert feature map to probabilities for each class
    outputs = Conv2D(num_classes, 1, activation='softmax')(conv9)
    
    # Define the U-net architecture from input to output
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Custom loss function
def combined_dice_focal_loss(y_true, y_pred):
    focal_loss_fn = tf.keras.losses.CategoricalFocalCrossentropy()
    dice_loss_fn = tf.keras.losses.Dice()
    
    focal_loss = focal_loss_fn(y_true, y_pred)
    dice_loss = dice_loss_fn(y_true, y_pred)
    
    return focal_loss + dice_loss

def load_images(image_paths):
    images = []
    for image_path in image_paths:
        single_img = Image.open(image_path).convert('RGB')
        single_img = np.reshape(single_img,(original_image_height, original_image_width,3)) 
        # Downsample
        single_img = cv2.resize(single_img, (image_width, image_height), interpolation=cv2.INTER_AREA)
        single_img = single_img/255.
        images.append(single_img)

    return np.array(images)

def load_labels(label_paths):
    labels = []
    for label_path in label_paths:
        single_label = Image.open(label_path)
        single_label = np.reshape(single_label,(original_image_height, original_image_width, 1)) 
        # Downsample
        single_label = cv2.resize(single_label, (image_width, image_height), interpolation=cv2.INTER_AREA)
        # Subtract 1 from label since the wildscenes dataset has them saved from 1 - 18
        single_label = single_label - 1 
        labels.append(single_label)

    return np.array(labels)

# Runs after each epoch to run tests and print images
class TestModelCallback(Callback):
    def __init__(self, image_test_set, label_test_set, num_test):
        super().__init__()
        self.image_test_set = image_test_set[0:num_test]
        self.label_test_set = label_test_set[0:num_test]

    def on_epoch_end(self, epoch, logs=None):
        test_model(self.model, self.image_test_set, self.label_test_set, f"Epoch {epoch}")

# Saves model every 5 epochs
class SaveModelCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        if not os.path.exists("model_saves/"):
            os.makedirs("model_saves/")
        # Save the model
        if epoch % 5 == 0:
            save_path = f"model_saves/epoch{epoch}.keras"
            self.model.save(save_path)
            print(f"Model saved to {model_path}")

def load_custom_model(path):

    print("Trying load focal_dice")
    segmentation_unet = load_model(path, custom_objects={'combined_dice_focal_loss': combined_dice_focal_loss})
    original_loss = combined_dice_focal_loss
    print("Loading focal_dice")
    
    selected_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16 ,17] # Only include classes tested in wildscenes
    metrics = ['accuracy', OneHotIoU(num_classes=num_classes, target_class_ids=selected_classes, name="M_IoU")]
    for class_num in selected_classes:
        metrics.append(OneHotIoU(num_classes=num_classes, target_class_ids=[class_num], name=f"IoU_{class_num}"))

    # Recompile new metrics since original trained models didn't include them
    segmentation_unet.compile(optimizer=tf.keras.optimizers.Adam(), loss=original_loss, metrics=metrics)
    
    return segmentation_unet

def continue_model_train(model, image_training_set, image_test_set, label_training_set, label_test_set, num_saved_images, batch_size, epochs, initial_epoch):
    # Convert to one-hot encoding
    label_training_set_cat = to_categorical(label_training_set, num_classes=num_classes)
    label_test_set_cat = to_categorical(label_test_set, num_classes=num_classes)

    # Create callback for testing the model after each epoch
    test_model_callback = TestModelCallback(image_test_set, label_test_set, num_saved_images)

    # Create callback for saving the model
    save_model_callback = SaveModelCallback()

    # Train the model
    print("Training Model")
    with tf.device('/GPU:0'):
        results = model.fit(image_training_set, label_training_set_cat, batch_size=batch_size, epochs=epochs, validation_data=(image_test_set, label_test_set_cat), callbacks=[test_model_callback, save_model_callback], initial_epoch=initial_epoch)

    # Save the model
    model.save(model_path)
    print(f"Model saved to {model_path}")

def initial_model_train(image_training_set, image_test_set, label_training_set, label_test_set, num_saved_images, custom_learning_rate, batch_size, epochs):
    # Convert to one-hot encoding
    label_training_set_cat = to_categorical(label_training_set, num_classes=num_classes)
    label_test_set_cat = to_categorical(label_test_set, num_classes=num_classes)

    # Create callback for testing the model after each epoch
    test_model_callback = TestModelCallback(image_test_set, label_test_set, num_saved_images)

    # Create callback for saving the model
    save_model_callback = SaveModelCallback()

    # Create the model
    selected_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16 ,17]
    metrics = ['accuracy', MeanIoU(num_classes=num_classes), OneHotMeanIoU(num_classes=num_classes)]
    segmentation_unet = Unet(input_size=(image_height, image_width, 3), num_classes=18, num_filters_multiplier=4)
    segmentation_unet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=custom_learning_rate), loss=combined_dice_focal_loss, metrics=metrics)

    # Train the model
    print("Training Model")
    # Specified CPU since testing on M3 macbook
    with tf.device('/CPU:0'):
        results = segmentation_unet.fit(image_training_set, label_training_set_cat, batch_size=batch_size, epochs=epochs, validation_data=(image_test_set, label_test_set_cat), callbacks=[test_model_callback, save_model_callback])

    # Save the model
    segmentation_unet.save(model_path)
    print(f"Model saved to {model_path}")

def test_model(model, image_test_set, label_test_set, test_name):
    # Make predictions on the test set
    predictions = model.predict(image_test_set)

    # Store true and predicted labels in lists
    y_true = []
    y_pred = []

    # Save the predicted labels as coloured images
    for i in range(len(predictions)):
        image_dir = f"test/image{i}"
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        
        # Find predicted labels from predctions
        predicted_label = np.argmax(predictions[i], axis=-1)
        y_true.append(label_test_set[i])
        y_pred.append(predicted_label)
        
        # Convert predicted labels to a coloured image
        colour_image = convert_label_to_image(predicted_label)
        colour_image = cv2.cvtColor(colour_image, cv2.COLOR_RGB2BGR)
        file_path = os.path.join(image_dir, f"{test_name}.png")
        cv2.imwrite(file_path, colour_image)

    # Save the labels as ground truth
    ground_truth_dir = f"test/ground_truth"
    if not os.path.exists(ground_truth_dir):
        os.makedirs(ground_truth_dir)

    for i in range(len(label_test_set)):
        ground_truth_image = convert_label_to_image(label_test_set[i].squeeze())
        ground_truth_image = cv2.cvtColor(ground_truth_image, cv2.COLOR_RGB2BGR)
        file_path = os.path.join(ground_truth_dir, f"ground_truth_{i}.png")
        cv2.imwrite(file_path, ground_truth_image)

    # Flatten for confusion matrix calculation
    y_true = np.concatenate(y_true).ravel()
    y_pred = np.concatenate(y_pred).ravel()

    # Compute confusion matrix
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes)

    # Normalize confusion matrix
    confusion_mtx = confusion_mtx / tf.reduce_sum(confusion_mtx, axis=1)[:, np.newaxis]

    # Plot the Confusion Matrix
    plt.figure(figsize=(16, 12))
    sns.heatmap(confusion_mtx, annot=True, fmt='.2f', cmap='viridis')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'{test_name} Confusion Matrix')

    # Save the plot instead of showing it
    heatmap_dir = f"test/heatmaps"
    if not os.path.exists(heatmap_dir):
        os.makedirs(heatmap_dir)
    heatmap_path = os.path.join(heatmap_dir, test_name)
    plt.savefig(heatmap_path)
    plt.close()

def main():
    # Manual Entry of args
    new_or_continue_or_test = "final"
    num_train_images = 5
    num_test_images = 100
    batch_size = 16
    custom_learning_rate = 0.001
    epochs = 70
    initial_epoch = 0

    num_saved_images = 10
    num_loaded_images = 640
    num_loaded_labels = 64

    if new_or_continue_or_test != "MULTI":
        # Get training data
        print("Loading Images")
        image_training_set_paths, image_test_set_paths, label_training_set_paths, label_test_set_paths = create_split(images_path, num_train_images, num_test_images, 0)
        image_training_set = load_images(image_training_set_paths)
        image_test_set = load_images(image_test_set_paths)
        label_training_set = load_labels(label_training_set_paths)
        label_test_set = load_labels(label_test_set_paths)

        if new_or_continue_or_test.lower() == "new":
            print("Creating Model")
            initial_model_train(image_training_set, image_test_set, label_training_set, label_test_set, num_saved_images, custom_learning_rate, batch_size, epochs)
        elif new_or_continue_or_test.lower() == "continue":
            print("Loading Model")
            model = load_custom_model(model_path)
            continue_model_train(model, image_training_set, image_test_set, label_training_set, label_test_set, num_saved_images, batch_size, epochs, initial_epoch)
        else:
            print("Loading Model")
            model = load_custom_model(model_path)
            test_model(model, image_test_set, label_test_set, f"final_test")
    else:
        # Used for training the final models with splits of 10 image sets
        print("Loading Images")
        image_training_set_paths, image_test_set_paths, label_training_set_paths, label_test_set_paths = create_split(images_path, num_train_images, num_test_images, 0)
        num_sub_runs = math.ceil(num_train_images/num_loaded_images)
        run_epoch_size = math.ceil(epochs / num_sub_runs)
        print(f"Number of sub runs: {num_sub_runs}")
        print(f"Num epoch size: {run_epoch_size}")
        for i in range (num_sub_runs):
            lower_train_index = i * num_loaded_images
            upper_train_index = (i + 1) * num_loaded_images
            lower_test_index = i * num_loaded_labels
            upper_test_index = (i + 1) * num_loaded_labels
            print(f"Loading images: {lower_train_index + 1} to {upper_train_index}")
            print(f"Loading labels: {lower_test_index + 1} to {upper_test_index}")
            image_training_set = load_images(image_training_set_paths[lower_train_index:upper_train_index])
            image_test_set = load_images(image_test_set_paths[lower_test_index:upper_test_index])
            label_training_set = load_labels(label_training_set_paths[lower_train_index:upper_train_index])
            label_test_set = load_labels(label_test_set_paths[lower_test_index:upper_test_index])

            initial_epoch = run_epoch_size * i
            final_epoch = run_epoch_size * (i + 1)

            if i == 0:
                print("Creating Model")
                initial_model_train(image_training_set, image_test_set, label_training_set, label_test_set, num_saved_images, custom_learning_rate, batch_size, final_epoch)
            else:
                print("Loading Model")
                model = load_custom_model(model_path)
                continue_model_train(model, image_training_set, image_test_set, label_training_set, label_test_set, num_saved_images, batch_size, final_epoch, initial_epoch)

            # Rename the ground truth directory
            ground_truth_dir = f"test/ground_truth"
            new_ground_truth_dir = f"test/ground_truth_subrun_{i + 1}"
            if os.path.exists(ground_truth_dir):
                os.rename(ground_truth_dir, new_ground_truth_dir)


if __name__ == '__main__':
    main()