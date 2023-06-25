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