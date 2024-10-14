# projet_cnn.py

import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # Utilise un backend compatible sans interface graphique
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

# Chargement et préparation des données
def load_and_prepare_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train_onehot = to_categorical(y_train, 10)
    y_test_onehot = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test), y_train_onehot, y_test_onehot

# Visualisation des images
def plot_sample_images(x, y, class_names, num_images=5, filename='sample_images.png'):
    plt.figure(figsize=(10, 2))
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(x[i])
        plt.title(class_names[int(y[i])])
        plt.axis('off')
    plt.savefig(filename)
    plt.close()

# Analyse de la distribution des classes
def plot_class_distribution(y, class_names, filename='class_distribution.png'):
    plt.figure(figsize=(8, 4))
    plt.hist(y, bins=np.arange(11) - 0.5, edgecolor='black', rwidth=0.8)
    plt.xticks(np.arange(10), class_names, rotation=45)
    plt.title('Distribution des classes dans l\'ensemble d\'entraînement')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Définition du modèle LeNet5
def create_lenet5_model(input_shape=(32, 32, 3), num_classes=10):
    model = Sequential([
        Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(16, kernel_size=(5, 5), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(120, activation='relu'),
        Dense(84, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

def create_vgg_model(num_blocks, input_shape=(32, 32, 3), num_classes=10):
    model = Sequential()
    for i in range(num_blocks):
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape if i == 0 else None))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def compile_and_train(model, x_train, y_train, x_test, y_test, optimizer, epochs=10):
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), batch_size=64)
    return history

# Fonction pour tracer les courbes de performance
def plot_history(history, title, filename):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Main function
def main():
    # Chargement des données
    (x_train, y_train), (x_test, y_test), y_train_onehot, y_test_onehot = load_and_prepare_data()

    # Affichage d'exemples d'images
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    plot_sample_images(x_train, y_train, class_names)

    # Affichage de la distribution des classes
    plot_class_distribution(y_train, class_names)

    # Liste des optimiseurs à tester
    optimizers = [
        ('SGD', lambda: SGD(learning_rate=0.01, momentum=0.9)),
        ('Adam', Adam),
        ('RMSprop', RMSprop)
    ]

    # Définition des modèles
    models = [
        ('LeNet5', create_lenet5_model),
        ('VGG1', lambda: create_vgg_model(num_blocks=1)),
        ('VGG2', lambda: create_vgg_model(num_blocks=2)),
        ('VGG3', lambda: create_vgg_model(num_blocks=3))
    ]

    # Entraînement et comparaison des modèles avec différents optimiseurs
    for model_name, model_func in models:
        for opt_name, optimizer_func in optimizers:
            print(f"Training {model_name} with {opt_name} optimizer...")
            model = model_func()  # Créer une nouvelle instance du modèle
            optimizer = optimizer_func()  # Créer une nouvelle instance de l'optimiseur
            history = compile_and_train(model, x_train, y_train_onehot, x_test, y_test_onehot, optimizer, epochs=10)
            plot_history(history, f'{model_name} - {opt_name}', f'{model_name.lower()}_{opt_name.lower()}_history.png')

    print("Training and evaluation completed. Check the generated plot files for results.")

# Exécution du script principal
if __name__ == '__main__':
    main()