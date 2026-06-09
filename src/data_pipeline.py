import tensorflow as tf

IMG_SIZE = 224
BATCH_SIZE = 16 

def parse_tfrecord(example_proto):
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/class/label': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    
    image = tf.image.decode_jpeg(parsed_features['image/encoded'], channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = image / 255.0 
    
    label = parsed_features['image/class/label']
    return image, label

def load_dataset(filepaths, is_training=True):
    dataset = tf.data.Dataset.from_tensor_slices(filepaths)
    
    if is_training:
        dataset = dataset.shuffle(len(filepaths))

    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,

        deterministic=not is_training 
    )
    

    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    

    if is_training:
        dataset = dataset.shuffle(10000)
        

    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True).repeat().prefetch(tf.data.AUTOTUNE)
    return dataset