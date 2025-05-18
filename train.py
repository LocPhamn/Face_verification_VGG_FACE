import os.path

from script.face_recognition.model.vgg_face import FacialRecognitionModel
from script.face_recognition.module import vgg_dataset
import tensorflow_addons as tfa
import tensorflow as tf

triplet_loss_fn = tfa.losses.TripletSemiHardLoss()

optimizer = tf.keras.optimizers.Adam()

@tf.function
def train_step(anchor, positive, negative, model):
    with tf.GradientTape() as tape:
        # Lấy embedding vector cho từng ảnh
        anchor_embed = model(anchor, training=True)
        positive_embed = model(positive, training=True)
        negative_embed = model(negative, training=True)

        # Ghép lại thành batch để dùng với TripletLoss
        embeddings = tf.concat([anchor_embed, positive_embed, negative_embed], axis=0)

        # Tạo nhãn ảo: mỗi nhóm 3 ảnh là một class
        # Giả định batch size là B => tổng số ảnh = 3B => 0,1,...,B-1 lặp 3 lần
        batch_size = tf.shape(anchor_embed)[0]
        labels = tf.range(batch_size)
        labels = tf.concat([labels, labels, labels], axis=0)

        # Tính loss
        loss = triplet_loss_fn(labels, embeddings)

    # Backprop
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def run():
    epochs = 3

    train_dir = r"D:\Python plus\AI_For_CV\dataset\face_dataset\train"
    val_dir = r"D:\Python plus\AI_For_CV\dataset\face_dataset\val"
    checkpoint_dir = r"D:\Python plus\AI_For_CV\dataset\face_dataset\checkpoint\v1"

    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    train_img_paths = vgg_dataset.get_image_paths_by_class(train_dir)
    train_triple = vgg_dataset.generate_triplets(train_img_paths)
    train_dataset = vgg_dataset.make_triplet_dataset(train_triple)

    val_img_paths = vgg_dataset.get_image_paths_by_class(val_dir)
    val_triple = vgg_dataset.generate_triplets(val_img_paths)
    val_dataset = vgg_dataset.make_triplet_dataset(val_triple)

    face_model = FacialRecognitionModel()
    model = face_model.load_model()

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        for anchor, positive, negative in train_dataset:
            loss = train_step(anchor, positive, negative, model)
            total_loss += loss
            num_batches += 1

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / num_batches:.4f}")

if __name__ == '__main__':
    run()

