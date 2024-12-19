# Description: Creates a random split of the data such that there is an equal amount from each of the 5 sequences
# Input: folder path to 2d wildscenes dataset (e.g. "PathToDataSet"),
# Input: size of traiing, test and random state of consistent results
# Output: imageTrainingSet, imageTestSet, labelTrainingSet, labelTestSet (all in path names)
def create_split(folder_path, train_size, test_size, random_state):
    import os
    import numpy as np
    from sklearn.model_selection import StratifiedShuffleSplit

    image_sequences = []
    image_paths = []
    label_paths = []

    # Add all non hidden folders to a list
    sequences = []
    for folder in os.listdir(folder_path):
        if not folder.startswith('.'):
            sequences.append(folder)

    # Add all images and labels to a list
    sequence_num = 0
    for sequence in sequences:
        sequence_images_path = os.path.join(folder_path, sequence, "image")
        sequence_labels_path = os.path.join(folder_path, sequence, "indexLabel")
        for image in os.listdir(sequence_images_path):
            image_path = os.path.join(sequence_images_path, image)
            label_path = os.path.join(sequence_labels_path, image)
            image_sequences.append(sequence_num)
            image_paths.append(image_path)
            label_paths.append(label_path)
        sequence_num += 1

    image_paths = np.array(image_paths)
    label_paths = np.array(label_paths)

    # Split the data such that there is a balanced amount from each sequence
    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_size, test_size=test_size, random_state=random_state)
    for train_index, test_index in sss.split(image_paths, image_sequences):
        imageTrainingSet = image_paths[train_index]
        imageTestSet = image_paths[test_index]
        labelTrainingSet = label_paths[train_index]
        labelTestSet = label_paths[test_index]

    return imageTrainingSet, imageTestSet, labelTrainingSet, labelTestSet

def convert_label_to_image(labelled_image):
    import numpy as np
    colour_dict = {
        0: (0, 0, 0), # Asphalt
        1: (97, 178, 88), # Dirt
        2: (250, 226, 80), # Mud
        3: (56, 128, 194), # Water
        4: (133, 41, 174), # Gravel
        5: (125, 237, 238), # Other-terrain
        6: (221, 69, 223), # Tree trunk
        7: (217, 244, 97), # Foliage
        8: (211, 53, 79), # Bush
        9: (55, 126, 127), # Fence
        10: (161, 113, 54), # Structure
        11: (0, 0, 0), # Pole
        12: (0, 0, 0), # Vehicle
        13: (189, 253, 200), # Rock
        14: (128, 128, 38), # Log
        15: (250, 190, 190), # Other-object
        16: (0, 0, 123), # Sky
        17: (128, 128, 128) # Grass
    }    

    height, width = labelled_image.shape
    output_image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            output_image[i, j] = colour_dict[labelled_image[i, j]]

    return output_image

def elements_in_first_only(list1, list2, list3):
    # Convert list2 and list3 to sets for faster lookup
    set2 = set(list2)
    set3 = set(list3)
    
    # Use a list comprehension to filter elements in list1 that are not in set2 or set3
    result = [element for element in list1 if element not in set2 and element not in set3]
    return result

# Since we used the 640, 64 split for most models and 6400, 640 split for the final 2 models this 
# function extracts the images that neither of thsoe models were trained on.
def return_untrained_images():
    images_path = "PathToDataSet"
    image_training_set_paths1, image_test_set_paths1, label_training_set_paths1, label_test_set_paths1 = create_split(images_path, 640, 64, 0)
    image_training_set_paths2, image_test_set_paths2, label_training_set_paths2, label_test_set_paths2 = create_split(images_path, 6400, 640, 0)
    image_training_set_paths, image_paths_all, label_training_set_paths, label_paths_all = create_split(images_path, 5, 9301, 0)
    new_images = elements_in_first_only(image_paths_all, image_training_set_paths1, image_training_set_paths2)
    new_labels = elements_in_first_only(label_paths_all, label_training_set_paths1, label_training_set_paths2)
    return new_images, new_labels

def main():
    create_split("PathToDataSet", 10, 5, 0)


if __name__ == '__main__':
    main()