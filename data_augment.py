from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import os
import scipy

# # Define the directory where your original images are stored and where augmented images will be saved.
original_data_dir = 'keyframe'
augmented_data_dir = './augmented_keyframes'

# Create an instance of the ImageDataGenerator with desired augmentation parameters.
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')


augmented_images_per_image = 5


# List all image files in the original dataset directory.
for gestures in os.listdir('keyframe'):
    print(gestures)


    for images in os.listdir(os.path.join(f'keyframe/{gestures}')):
        print(images)
        img = image.load_img(os.path.join('keyframe',gestures,images))
        x = image.img_to_array(img)
        x = x.reshape((1,) + x.shape)

        i = 0
        for batch in datagen.flow(x, batch_size=1):
            augmented_image = image.array_to_img(batch[0])
            if not os.path.exists(os.path.join(augmented_data_dir, gestures)):
                os.mkdir(os.path.join(augmented_data_dir, gestures ))

            # augmented_image.save(os.path.join(augmented_data_dir, original_image_file.split('.')[0] + f'_augmented_{i}.jpg'))
            augmented_image.save(os.path.join(augmented_data_dir, gestures , images + f'_augmented_{i}.jpg'))

            i += 1
            if i >= augmented_images_per_image:
                break

