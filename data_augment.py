from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import os
import numpy as np
import cv2

# # Define the directory where your original images are stored and where augmented images will be saved.
original_data_dir = 'keyframe'
augmented_data_dir = './augmented_keyframes_updated'

# Create an instance of the ImageDataGenerator with desired augmentation parameters.
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')


"""
# Create an ImageDataGenerator for data augmentation and preprocessing
datagen = ImageDataGenerator(
    rescale=1.0/255,  # Rescale pixel values to [0, 1]
    rotation_range=20,  # Rotate images by up to 20 degrees
    width_shift_range=0.2,  # Shift width by up to 20% of the image width
    height_shift_range=0.2,  # Shift height by up to 20% of the image height
    shear_range=0.2,  # Shear transformations
    zoom_range=0.2,  # Zoom in/out by up to 20%
    horizontal_flip=True,  # Randomly flip images horizontally
    fill_mode='nearest'  # Fill in any missing pixels with the nearest value
)
"""

augmented_images_per_image = 5

def image_augmentation():

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
                    os.makedirs(os.path.join(augmented_data_dir, gestures ))

                # augmented_image.save(os.path.join(augmented_data_dir, original_image_file.split('.')[0] + f'_augmented_{i}.jpg'))
                augmented_image.save(os.path.join(augmented_data_dir, gestures , images.split('.')[0] + f'augmented_{i}.jpg'))

                i += 1
                if i >= augmented_images_per_image:
                    break


def image_augmetation_with_segmentation():

# ToDo : need a common variable for assigning path

    for gestures in os.listdir('../keyframe'):
        print(gestures)


        for images in os.listdir(os.path.join(f'../keyframe/{gestures}')):
            img = cv2.imread(os.path.join(f'../keyframe/{gestures}/{images}'))
            # Apply segmentation (e.g., grabCut) to separate the foreground from the background.
            mask = np.zeros(img.shape[:2], np.uint8)
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            rect = (50, 50, img.shape[1] - 50, img.shape[0] - 50)
            cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            segmented_img = img * mask2[:, :, np.newaxis]

            # Expand the dimensions to make it rank 4 (batch_size=1).
            segmented_img = np.expand_dims(segmented_img, axis=0)
            
            i = 0
            for batch in datagen.flow(segmented_img, batch_size=1):
                augmented_image = image.array_to_img(batch[0])
                if not os.path.exists(os.path.join(augmented_data_dir, gestures)):
                    os.makedirs(os.path.join(augmented_data_dir, gestures ))

                # augmented_image.save(os.path.join(augmented_data_dir, original_image_file.split('.')[0] + f'_augmented_{i}.jpg'))
                augmented_image.save(os.path.join(augmented_data_dir, gestures , images.split('.')[0] + f'augmented_{i}.jpg'))

                i += 1
                if i >= augmented_images_per_image:
                    break

if __name__ == '__main__':
    
    # To Do call any of the function
    pass