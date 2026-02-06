import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import os

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS_STAGE1 = 25
EPOCHS_STAGE2 = 25
SEED = 42

DATASET_DIR = r"C:\Users\DELL\Desktop\projects\wall crack - civil\dataset"
STAGE1_MODEL_PATH = "stage1_shrinkage_vs_structural_model_f.keras"
STAGE2_MODEL_PATH = "stage2_settlement_vs_vertical_model_f.keras"

tf.random.set_seed(SEED)
np.random.seed(SEED)

AUTOTUNE = tf.data.AUTOTUNE

train_raw = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_raw = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

class_names = train_raw.class_names
print("Classes:", class_names)

IDX_SHRINKAGE = class_names.index("shrinkage")
IDX_SETTLEMENT = class_names.index("settlement")
IDX_VERTICAL = class_names.index("vertical")

def stage1_map(x, y):
    y = tf.where(y == IDX_SHRINKAGE, 0, 1)
    return x, y

train_s1 = train_raw.map(stage1_map).shuffle(1000, seed=SEED).cache().prefetch(AUTOTUNE)
val_s1 = val_raw.map(stage1_map).cache().prefetch(AUTOTUNE)

y_train_s1 = np.concatenate([y.numpy() for _, y in train_s1])
cw_s1 = dict(enumerate(compute_class_weight("balanced", classes=np.unique(y_train_s1), y=y_train_s1)))

base_s1 = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_s1.trainable = False

inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = tf.keras.applications.efficientnet.preprocess_input(inp)
x = base_s1(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)
out = tf.keras.layers.Dense(2, activation="softmax")(x)

stage1_model = tf.keras.Model(inp, out)
stage1_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n Training Stage 1 (Shrinkage vs Structural)")
stage1_model.fit(
    train_s1,
    validation_data=val_s1,
    epochs=EPOCHS_STAGE1,
    class_weight=cw_s1,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)]
)

stage1_model.save(STAGE1_MODEL_PATH)
print(" Stage 1 model saved")

def is_structural(x, y):
    return tf.logical_or(tf.equal(y, IDX_SETTLEMENT), tf.equal(y, IDX_VERTICAL))

def stage2_map(x, y):
    y = tf.where(y == IDX_SETTLEMENT, 0, 1)
    return x, y

train_s2_filtered = train_raw.unbatch().filter(is_structural).map(stage2_map).cache()
y_train_s2 = np.array([y.numpy() for _, y in train_s2_filtered])
cw_s2 = dict(enumerate(compute_class_weight("balanced", classes=np.unique(y_train_s2), y=y_train_s2)))

train_s2 = (
    train_s2_filtered
    .shuffle(1000, seed=SEED)
    .batch(BATCH_SIZE)
    .repeat()
    .prefetch(AUTOTUNE)
)

val_s2 = (
    val_raw.unbatch()
    .filter(is_structural)
    .map(stage2_map)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTOTUNE)
)

steps_per_epoch = np.ceil(len(y_train_s2) / BATCH_SIZE).astype(int)

base_s2 = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
for layer in base_s2.layers[:int(len(base_s2.layers) * 0.7)]:
    layer.trainable = False

inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = tf.keras.applications.efficientnet.preprocess_input(inp)
x = base_s2(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.4)(x)
out = tf.keras.layers.Dense(2, activation="softmax")(x)

stage2_model = tf.keras.Model(inp, out)
stage2_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n Training Stage 2 (Settlement vs Vertical)")
stage2_model.fit(
    train_s2,
    validation_data=val_s2,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS_STAGE2,
    class_weight=cw_s2,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)]
)

stage2_model.save(STAGE2_MODEL_PATH)
print(" Stage 2 model saved")

stage1_model = tf.keras.models.load_model(STAGE1_MODEL_PATH)
stage2_model = tf.keras.models.load_model(STAGE2_MODEL_PATH)

def predict_crack(image):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.expand_dims(image, 0)

    s1 = np.argmax(stage1_model.predict(image), axis=1)[0]
    if s1 == 0:
        return "shrinkage"
    else:
        s2 = np.argmax(stage2_model.predict(image), axis=1)[0]
        return "settlement" if s2 == 0 else "vertical"
