import tensorflow as tf
import os, glob

tfrec_dir = "/root/MusicGenerationML/data/maestro_tfrecords"
tf_files = glob.glob(os.path.join(tfrec_dir, "*.tfrecord-*"))

print(f"Found {len(tf_files)} TFRecord files")
print("Inspecting first file:", tf_files[0])

# Try to parse one example and see what features are available
dataset = tf.data.TFRecordDataset([tf_files[0]])

for raw_record in dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    
    print("\nAvailable features:")
    for key, feature in example.features.feature.items():
        if feature.HasField('bytes_list'):
            print(f"  {key}: bytes_list (length: {len(feature.bytes_list.value)})")
        elif feature.HasField('float_list'):
            print(f"  {key}: float_list (length: {len(feature.float_list.value)})")
        elif feature.HasField('int64_list'):
            print(f"  {key}: int64_list (length: {len(feature.int64_list.value)})")
    break
