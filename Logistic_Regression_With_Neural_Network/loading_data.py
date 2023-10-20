import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import image

def load_dataset(data_directory, image_size=(64, 64), test_size=0.2, random_seed=None):
    # Get the list of class names from subdirectories
    class_names = os.listdir(data_directory)
    
    # Initialize lists to store data
    X = []
    y = []

    for class_name in class_names:
        class_path = os.path.join(data_directory, class_name)
        for i in os.listdir(class_path):
            ncx=os.path.join(class_path,i)
            for image_name in os.listdir(ncx):
                img = image.load_img(os.path.join(ncx, image_name), target_size=image_size)
                img = image.img_to_array(img)
                
                # Append image data and class label
                X.append(img)
                y.append(class_name)

    # Encode class labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

    # Convert lists to NumPy arrays
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # Normalize pixel values to be between 0 and 1
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return X_train, y_train, X_test, y_test, class_names

# # Specify the path to your dataset directory
# data_directory = "log_dataset"

# # Load the dataset
# X_train, y_train, X_test, y_test, class_names = load_dataset(data_directory)
