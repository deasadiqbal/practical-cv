"""
pretrained embedding for image vision task
pretained embedding are used to initialize the embedding layer of the model 
"""

import tensorflow_hub as hub
huburl = "https://tfhub.dev/google/imagenet/\
 mobilenet_v2_100_224/feature_vector/4"
hub_layer = hub.KerasLayer(handle = huburl, input_shape=(224, 224, 3), trainable=False,#layer not trainable and should be assumed to be pretrained.
                            name='pretrained_mobilenet_embedding')

#lets use it into a model
layers = [
    hub.kerasLayer(handle = huburl, input_shape=(224,224,3),
                   trainable=False, name='pretrained_mobilenet_embedding'),
    tf.keras.layers.Dense(units = 16, activation='relu', name = 'hidden_layer'),
    tf.keras.layers.Dense(units = len(CLASS_NAME), activation='softmax', name = 'output_layer')
]

model = tf.keras.Sequential(layers=layers, name='flower_classification_model')
model.summary()
'''this is the output from .summary=>
Total params: 2,278,565
Trainable params: 20,581
Non-trainable : 2,257,984
'''

# this is full implementation of the model
import tensorflow_hub as hub
import os
# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

# parameterize to the values in the previous cell
def train_and_evaluate(batch_size = 32,
                       lrate = 0.001,
                       l1 = 0.,
                       l2 = 0.,
                       num_hidden = 16):
  regularizer = tf.keras.regularizers.l1_l2(l1, l2)

  train_dataset = (tf.data.TextLineDataset(
      "gs://practical-ml-vision-book/flowers_5_jpeg/flower_photos/train_set.csv").
      map(decode_csv)).batch(batch_size)

  eval_dataset = (tf.data.TextLineDataset(
      "gs://practical-ml-vision-book/flowers_5_jpeg/flower_photos/eval_set.csv").
      map(decode_csv)).batch(32) # this doesn't matter

  layers = [
      hub.KerasLayer(
          "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4",
          input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
          trainable=False,
          name='mobilenet_embedding'),
      tf.keras.layers.Dense(num_hidden,
                            kernel_regularizer=regularizer, 
                            activation='relu',
                            name='dense_hidden'),
      tf.keras.layers.Dense(len(CLASS_NAMES), 
                            kernel_regularizer=regularizer,
                            activation='softmax',
                            name='flower_prob')
  ]

  model = tf.keras.Sequential(layers, name='flower_classification')
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lrate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=False),
                metrics=['accuracy'])
  print(model.summary())
  history = model.fit(train_dataset, validation_data=eval_dataset, epochs=5)
  training_plot(['loss', 'accuracy'], history)
  return model


#pretrained model in keras
import tensorflow as tf
from keras_adamw import AdamW

preTrained_model = tf.keras.applications.MobileNetV2(input_shape=[244,244,3],
                                                     weights='Imagenet',
                                                     include_top=False)
preTrained_model.trainable = False

model = tf.keras.Sequential([
  # convert image format from int [0,255]
  # to the format expected by this model
  tf.keras.layers.Lambda(
    lambda data: tf.keras.applications.mobilenet.preprocess_input(
      tf.cast(data, dtype=tf.float32)
    )
    """
    The tf.keras.layers.Lambda function in TensorFlow is used to create
    a custom layer that applies a given function to the input data. In the provided code snippet,
    tf.keras.applications.mobilenet.preprocess_input is the function being applied to the data.
    """
  ),
  input_shape = [244,244,3],
  preTrained_model,
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(256, activation = 'relu'),
  tf.keras.layers.Dense(2, activation = 'softmax')
])

mult = 0.4 # for pretrained layers
mult_by_layer={ 
            # Clasification head
            'flower_prob': 1.0,
            'flower_dense': 1.0,
            # Pretrained layers
            'block_1_': 0.02 * mult,
            'block_2_': 0.04 * mult,
            'block_3_': 0.06 * mult,
            'block_4_': 0.08 * mult,
            'block_5_': 0.1 * mult,
            'block_6_': 0.15 * mult,
            'block_7_': 0.2 * mult,
            'block_8_': 0.25 * mult,
            'block_9_': 0.3 * mult,
            'block_10_': 0.35 * mult,
            'block_11_': 0.4 * mult,
            'block_12_': 0.5 * mult,
            'block_13_': 0.6 * mult,
            'block_14_': 0.7 * mult,
            'block_15_': 0.8 * mult,
            'block_16_': 0.9 * mult,
            # these layers do not have stable identifiers in tf.keras.applications.MobileNetV2
            'conv': 0.5 * mult,
            'Conv': 0.5 * mult
    }
optimizer = AdamW(lr=0.001, model=model, lr_multipliers=mult_by_layer)
    
model.compile(
    #optimizer='adam',
    optimizer=optimizer,
    loss = 'sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy'],
    steps_per_execution=8
)
model.fit(train_dataset, validation_data=val_dataset, epochs = 10)
