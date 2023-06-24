#reading the data
import tensorflow as tf
def load_and_decode(filename, reshape_dim):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, reshape_dim)


file_name = "E:\\computer vision\\practical-cv\\dataset\\train"
#vizulize the data
import matplotlib.pyplot as plt
def show_img(filename):
    img = load_and_decode(filename, [256, 256])
    plt.imshow(img)
    plt.axis('off')
    plt.show()


print(show_img(file_name))

tulips = tf.io.gfile.glob(
 "gs://cloud-ml-data/img/flower_photos/tulips/*.jpg")
for i in range(5):
    show_img(tulips[i])