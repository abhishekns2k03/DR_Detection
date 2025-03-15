from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data Augmentation
data_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Assuming train_data is a generator created using flow_from_directory
train_data = data_gen.flow_from_directory(
    'path_to_train_data',
    target_size=(600, 600),
    batch_size=32,
    class_mode='categorical'
)

# Fine-tuning the model
with tpu_strategy.scope():
    model.trainable = True  # Unfreeze the model for fine-tuning
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Lower learning rate for fine-tuning
                  metrics=["accuracy"])

history = model.fit(train_data, epochs=25, validation_data=val_data, callbacks=[reduce_lr, early_stop])