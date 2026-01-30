import tensorflow as tf

IMG_SIZE = 48
BATCH_SIZE = 32
EPOCHS = 15
NUM_CLASSES = 7   

# =====================
# LOAD DATASET
# =====================
train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

# NORMALISASI
train_ds = train_ds.map(lambda x, y: (x/255.0, y))
test_ds = test_ds.map(lambda x, y: (x/255.0, y))

# =====================
# MODEL CNN
# =====================
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(48,48,3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

# =====================
# COMPILE
# =====================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =====================
# TRAIN
# =====================
model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS
)

# =====================
# SAVE MODEL
model.save("emotion_model.h5")
print("emotion_model.h5 berhasil dibuat!")