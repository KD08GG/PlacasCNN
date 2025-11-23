# train_classifier.py
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import CLASSIFIER_CONFIG, create_directories, CLASSIFIER_MODEL_DIR
from pathlib import Path

def build_model(img_size=32, num_classes=36):
    inp = layers.Input((img_size, img_size,1))
    x = layers.Conv2D(32,3,activation='relu',padding='same')(inp)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64,3,activation='relu',padding='same')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(128,3,activation='relu',padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inp, out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train(data_dir, img_size=32, batch=64, epochs=50, save_path=None):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(data_dir)
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.15,
                                 rotation_range=8, width_shift_range=0.1, height_shift_range=0.1)
    train_gen = datagen.flow_from_directory(
        str(data_dir), target_size=(img_size,img_size), color_mode='grayscale',
        batch_size=batch, class_mode='categorical', subset='training'
    )
    val_gen = datagen.flow_from_directory(
        str(data_dir), target_size=(img_size,img_size), color_mode='grayscale',
        batch_size=batch, class_mode='categorical', subset='validation'
    )
    model = build_model(img_size=img_size, num_classes=CLASSIFIER_CONFIG["num_classes"])
    model.summary()
    model.fit(train_gen, validation_data=val_gen, epochs=epochs)
    save_path = Path(save_path or (CLASSIFIER_MODEL_DIR / "classifier.h5"))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    print(f"Modelo guardado en {save_path}")
    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="carpeta con subcarpetas por clase")
    parser.add_argument("--epochs", type=int, default=CLASSIFIER_CONFIG["epochs"])
    args = parser.parse_args()
    create_directories()
    train(args.data, img_size=CLASSIFIER_CONFIG["img_size"], epochs=args.epochs)
