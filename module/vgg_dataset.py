
import glob
import os
from collections import defaultdict
import random
import tensorflow as tf


classes = ["ChauBui","Erik","HoaMinzy","KhoaPub","LamVlog","LanAnh","NguyenVietHoang","PhuongLy","SonTung","TranMinhHieu"]

def get_image_paths_by_class(root_dir, classes = ["ChauBui","Erik","HoaMinzy","KhoaPub","LamVlog","LanAnh","NguyenVietHoang","PhuongLy","SonTung","TranMinhHieu"]):
    class_to_images = defaultdict(list)
    for person in classes:
        image_path = glob.glob(r"{}/{}/*.jpg".format(root_dir,person))
        class_to_images[person].extend(image_path)
    return class_to_images

def generate_triplets(class_to_images, num_triplets=300):
    triplets = []
    labels = list(class_to_images.keys())

    for _ in range(num_triplets):
        anchor_class = random.choice(labels)
        negative_class = random.choice([l for l in labels if l != anchor_class])

        anchor_imgs = class_to_images[anchor_class]
        negative_imgs = class_to_images[negative_class]

        if len(anchor_imgs) < 2 or len(negative_imgs) < 1:
            continue

        anchor = random.choice(anchor_imgs)
        positive = random.choice([img for img in anchor_imgs if img != anchor])
        negative = random.choice(negative_imgs)

        triplets.append((anchor, positive, negative))

    return triplets

def preprocess_image(file_path, image_size=(224, 224)):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image

def load_triplet(anchor_path, positive_path, negative_path):
    return (
        preprocess_image(anchor_path),
        preprocess_image(positive_path),
        preprocess_image(negative_path)
    )

def make_triplet_dataset(triplets, image_size=(224, 224), batch_size=32):
    anchor_paths, positive_paths, negative_paths = zip(*triplets)

    dataset = tf.data.Dataset.from_tensor_slices((list(anchor_paths), list(positive_paths), list(negative_paths)))
    dataset = dataset.map(lambda a, p, n: load_triplet(a, p, n), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(512).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset

if __name__ == '__main__':
    root_dir = r"D:\Python plus\AI_For_CV\dataset\face_dataset\train"
    path = get_image_paths_by_class(root_dir,classes)
    triple_data = generate_triplets(path)
    dataset = make_triplet_dataset(triple_data)
    for anchor, positive, negative in dataset.take(1):
        print(anchor.shape, positive.shape, negative.shape)
