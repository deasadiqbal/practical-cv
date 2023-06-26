#dataset function that use to create dataset for our model
import tensorflow as tf
IMG_SIZE = [244,244,3]
BATCH_SIZE = 16
AUTO = tf.data.experimental.AUTOTUNE
TRAINING_FILENAME = 'your training data file'
VALIDATION_FILENAME = 'your val data file'

def decode_model(img_data):
    img = tf.image.decode_jpeg(img_data, channels=3)
    img = tf.image.reshape(img, [IMG_SIZE])
    return img
    """
    this function is used to decode the image, the image is decode using decode_jpeg,
    we use decode_jpeg because the image is in jpeg format, the image is decode to 3 channel
    """

def tf_records(example):
    TFRECDS_FORMAT ={
        'image': tf.io.FixedLenFeature([], tf.string), #fixed length feature for image
        'class': tf.io.FixedLenFeature([], tf.int64) #fixed length feature for class
    },
    example = tf.io.parse_single_example(example, TFRECDS_FORMAT), #parse_single_exm is used to parse the example
    image = decode_model(example['image']),
    label = tf.cast(example['class'], tf.int32)
    return image, label
    """
    This function use to decode image and label from tfrecords file, here fixed length feature is use to
    define the length of the feature, in this case image and class label. 
    tf.io.parse_single_example is use to parse the example from tfrecords file
    tf.cast is used to cast the label to int32
    """

def load_dataset(filename, ordered = False):
    ignore_order = tf.data.Options() #the option to ignore order
    if not ordered: #if not ordered then disable order, if ordered then enable order
        ignore_order.experimental_deterministic = False #disable order, increase speed 
    dataset = tf.data.TFRecordDataset(filename, num_parallel_reads=AUTO) #read the tfrecords file
    dataset = dataset.with_options(ignore_order) #apply the option
    dataset = dataset.map(tf_records, num_parallel_calls=AUTO) #map the tfrecords file
    return dataset
    """
    This function use to load the dataset from tfrecords file, the dataset is read using TFRecordDataset 
    and then the dataset is map using tf_records function
    """

def data_agument(image, label):
    image = tf.image.random_flip_left_right(image) #flip the image
    image = tf.image.random_flip_up_down(image) #flip the image
    image = tf.image.random_brightness(image, max_delta=0.5) #change the brightness
    image = tf.image.random_contrast(image, lower=0.2, upper=0.5) #change the contrast
    image = tf.image.random_saturation(image, lower=0.2, upper=0.5) #change the saturation
    image = tf.image.random_hue(image, max_delta=0.5) #change the hue
    return image, label
    """
    This function use to agument the image, the image is flip, change brightness, contrast, saturation and hue
    """

def get_training_data():
    dataset = load_dataset(TRAINING_FILENAME) #load the dataset
    dataset = dataset.map(data_agument, num_parallel_calls=AUTO) #map the dataset to data_agument function to agument the image 
    dataset = dataset.repeat() #repeat the dataset 
    dataset = dataset.shuffle(2048) #shuffle the dataset 
    dataset = dataset.batch(BATCH_SIZE) #batch the dataset batch mean the number(amount) of data that will be train in one iteration
    dataset = dataset.prefetch(AUTO) #prefetch the dataset, prefetch is used to load the data before the data is needed 
    return dataset

def get_val_data(ordered=False):
    dataset = load_dataset(VALIDATION_FILENAME, ordered=ordered) #load the dataset
    dataset = dataset.batch(BATCH_SIZE) #batch the dataset batch mean the number(amount) of data that will be train in one iteration
    dataset = dataset.cache() #cache the dataset, cache is used to load the data before the data is needed 
    dataset = dataset.prefetch(AUTO) #prefetch the dataset, prefetch is used to load the data before the data is needed 
    return dataset
