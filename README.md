## 📁 Project Structure
```
faceexpressionrecognition/
│
├── data/                       # dataset goes here (not included)
│
├── models/
│   └── model.h5                # saved after training
│
├── notebooks/
│   └── training_notebook.ipynb
│
├── results/
│   ├── training_curves.png
│   └── confusion_matrix.png
│
├── src/
│   ├── dataloader.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── requirements.txt
└── README.md
```

---
# 📌 FILE CONTENTS

## 📄 requirements.txt
```txt
tensorflow
numpy
matplotlib
opencv-python
scikit-learn
pillow
```

---

## 📄 src/dataloader.py
```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(data_path, img_size=(48, 48), batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_data = train_datagen.flow_from_directory(
        data_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_data = train_datagen.flow_from_directory(
        data_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_data, val_data
```

---

## 📄 src/model.py
```python
from tensorflow.keras import layers, models

def build_model(input_shape=(48, 48, 3), num_classes=7):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
```

---

## 📄 src/train.py
```python
import matplotlib.pyplot as plt
from model import build_model
from dataloader import load_data

DATA_PATH = "../data/train"

train_data, val_data = load_data(DATA_PATH)
model = build_model(num_classes=train_data.num_classes)

history = model.fit(train_data, validation_data=val_data, epochs=25)

model.save("../models/model.h5")

# Plot Curves
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.savefig('../results/training_curves.png')
plt.close()
```

---

## 📄 src/evaluate.py
```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
from dataloader import load_data

DATA_PATH = "../data/train"
_, val_data = load_data(DATA_PATH)

model = load_model("../models/model.h5")

pred = model.predict(val_data)
pred_labels = np.argmax(pred, axis=1)
true_labels = val_data.classes

cm = confusion_matrix(true_labels, pred_labels)
fig = ConfusionMatrixDisplay(cm).plot()
plt.savefig('../results/confusion_matrix.png')
```

---

## 📄 src/predict.py
```python
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import sys

model = load_model("../models/model.h5")
label_map = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

img_path = sys.argv[1]
img = cv2.imread(img_path)
img = cv2.resize(img, (48, 48))
img = img.astype('float32')/255
img = np.expand_dims(img, axis=0)

pred = model.predict(img)
print("Predicted Emotion:", label_map[np.argmax(pred)])
```
