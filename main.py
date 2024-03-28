from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import shutil
import pickle,joblib

def load_and_preprocess_images(directory, target_size=(150, 150),ext=".jpg"):
    """
    Load images from a directory, resize them, and normalize the pixel values.

    Parameters:
    - directory: Path to the directory containing the images.
    - target_size: Desired size of the images as (height, width).

    Returns:
    - A NumPy array containing the preprocessed images.
    """
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(ext):  # Assuming the images are in JPEG format
            # Load the image
            img_path = os.path.join(directory, filename)
            img = Image.open(img_path).convert('RGB')  # Ensure images are in RGB format

            # Resize the image
            img = img.resize(target_size)

            # Convert the image to a NumPy array and normalize pixel values
            img_array = np.array(img) / 255.0  # Normalize to 0-1 range

            # Append to the list of images
            images.append(img_array)

    # Convert the list of images to a NumPy array
    return np.array(images)


def build_and_compile_model(conv_layers, dense_units, learning_rate,op_activation):
    model = models.Sequential()
    model.add(layers.Conv2D(conv_layers[0], (3, 3), activation='relu', input_shape=(150,150,3)))
    model.add(layers.MaxPooling2D((2, 2)))
    for layer_size in conv_layers[1:]:
        model.add(layers.Conv2D(layer_size, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(dense_units, activation='relu'))
    model.add(layers.Dense(2, activation=op_activation))#'softmax'))

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def create_dataset_x_y(class1_path,class2_path):
    # Load, reshape, and normalize the images
    preprocessed_images1 = load_and_preprocess_images(class1_path)[:200]
    preprocessed_images2 = load_and_preprocess_images(class2_path)[:200]

    full_dataset = np.concatenate((preprocessed_images1,preprocessed_images2),axis=0)
    print("Shape of the preprocessed images array:", preprocessed_images1.shape)
    print("Shape of the preprocessed images array 2:", preprocessed_images2.shape)
    print("Shape of the preprocessed images array:", full_dataset.shape)

    # create target variable for above classes
    y1=np.ones(750)[:200] 
    y2=np.zeros(2500)[:200]
    y=np.concatenate((y1,y2),axis=0)
    return full_dataset,y

def main(source_image,retrain=0,ext=".jpg"):
    if retrain:
        # 
        # Directory containing the images
        directory1 = r'.\Healthy-Defective-Fruits-main\apple_images\apple_real_images\bruise_defect'
        directory2 = r'.\Healthy-Defective-Fruits-main\apple_images\apple_real_images\fresh'
        full_dataset,y=create_dataset_x_y(directory1,directory2)

        ## train test split
        # Assuming `full_dataset` is your features and `y` is your labels
        x_train, x_test, y_train, y_test = train_test_split(full_dataset, y, test_size=0.2, random_state=44, stratify=y)  # Note `stratify=y` here
        print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

        # Hyperparameters sets to try
        hyperparameters = [
            {'conv_layers': [32, 64, 64], 'dense_units': 64, 'learning_rate': 1e-2,'op_activation':'softmax'},
            {'conv_layers': [32, 64, 64], 'dense_units': 64, 'learning_rate': 1e-2,'op_activation':'sigmoid'},
            # Add more hyperparameter configurations here
        ]

        best_accuracy = 0
        best_hyperparameters = None

        for params in hyperparameters:
            model = build_and_compile_model(**params)
            history = model.fit(x_train, y_train, epochs=1, validation_split=0.2, verbose=1)
            accuracy = max(history.history['val_accuracy'])  # Get the best validation accuracy
            print(f"Params: {params}, Accuracy: {accuracy}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_hyperparameters = params

        print(f"Best Hyperparameters: {best_hyperparameters}, Accuracy: {best_accuracy}")

        model = build_and_compile_model(**best_hyperparameters)
        print(model.summary())
        model.fit(x_train, y_train, epochs=1, validation_split=0.2, verbose=1)

        # Evaluate the model
        model.evaluate(x_test, y_test)
        model.save('fruit_defect_model.h5')
        
    else:
        model = tf.keras.models.load_model('fruit_defect_model.h5')
        
    # Creating a temp folder
    folder_name = "temp"
    os.makedirs(folder_name)
    # Copy an image file to that folder
    destination_folder = folder_name
    shutil.copy(source_image, destination_folder)
    predict_input=load_and_preprocess_images(destination_folder,ext=ext)
        # Delete the folder
    os.system(f"rmdir /s /q {folder_name}")
    
    predictions=model.predict(predict_input)
    
    return np.argmax(predictions,axis=1)


if __name__ == "__main__":
    source_image = "./Healthy-Defective-Fruits-main/apple_images/apple_synthethic_images/bruise_defect/G_SINT_0011.png" # Provide the path to your source image
    output=main(source_image,retrain=0,ext=".png")
    print(output)
    output = "defect" if output[0]==1 else "fresh"
    print(output)