# Import modules
from image_tools import convert_label_to_image, return_untrained_images
from Unet_Training import load_images, load_labels, load_custom_model
import os
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Tensorflow modules
import tensorflow as tf
from keras.utils import to_categorical

num_classes=18

def final_evaluation(model_paths, num_test_images, num_print_images, batch_size, results_dir="test_results"):
    print("Loading Images")
    new_test_image_paths, new_test_label_paths = return_untrained_images()
    image_test_set = load_images(new_test_image_paths[:num_test_images])
    label_test_set = load_labels(new_test_label_paths[:num_test_images])
    label_test_set_cat = to_categorical(label_test_set, num_classes=num_classes)

    print(model_paths)

    for path in model_paths:
        print(f"Testing {path}")
        # Load the model
        model = load_custom_model(path)
        
        # Extract model name from the path
        model_name = os.path.basename(path).split('.')[0]
        model_results_dir = os.path.join(results_dir, model_name)
        
        if not os.path.exists(model_results_dir):
            os.makedirs(model_results_dir)
        
        # Evaluate the model
        print("Running Eval")
        results = model.evaluate(image_test_set, label_test_set_cat, batch_size=batch_size, return_dict=True)
        print(f"Evaluation results for {model_name}: {results}")
        
        # Save evaluation results to a file
        results_file_path = os.path.join(model_results_dir, f"{model_name}_evaluation_results.txt")
        with open(results_file_path, "w") as results_file:
            results_file.write(f"Evaluation results for {model_name}:\n{results}\n")
        
        # Make predictions on the test set
        print("Running Prediction")
        predictions = model.predict(image_test_set, batch_size=batch_size)

        # Store true and predicted labels in lists
        y_true = []
        y_pred = []

        print("Saving Results")
        # Save the predicted labels as coloured images
        for i in range(len(predictions)):    
            # Find predicted labels from predictions
            predicted_label = np.argmax(predictions[i], axis=-1)
            y_true.append(label_test_set[i])
            y_pred.append(predicted_label)
            
            # Convert predicted labels to a coloured image
            if i < num_print_images:
                colour_image = convert_label_to_image(predicted_label)
                colour_image = cv2.cvtColor(colour_image, cv2.COLOR_RGB2BGR)
                file_path = os.path.join(model_results_dir, f"{i}.png")
                cv2.imwrite(file_path, colour_image)

        # Save the labels as ground truth
        ground_truth_dir = os.path.join(model_results_dir, "ground_truth")
        if not os.path.exists(ground_truth_dir):
            os.makedirs(ground_truth_dir)

        for i in range(num_print_images):
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
        plt.title(f'{model_name} Confusion Matrix')

        # Save the plot instead of showing it
        heatmap_path = os.path.join(model_results_dir, f"{model_name}_confusion_matrix.png")
        plt.savefig(heatmap_path)
        plt.close()

def main():
    # Manual Entry of args
    # Plase place the final model in the models/ folder
    # Print images must be less than test images
    num_test_images = 100 # All unseen by model
    num_print_images = 10
    batch_size = 16

    print("Running Final Tests")
    model_dir = "models/"
    model_paths = [os.path.join(model_dir, file) for file in os.listdir(model_dir) if file.endswith('.keras')]
    final_evaluation(model_paths, num_test_images, num_print_images, batch_size)


if __name__ == '__main__':
    main()