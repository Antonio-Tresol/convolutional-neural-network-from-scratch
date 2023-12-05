import cv2
import os


class ImageProcessor:
    def __init__(self, width=512, height=512):
        self._width = width
        self._height = height

    def resize_pictures(self, folder_location):
        print("Loading and resizing pictures")

        self.transform_pictures(folder_location, self.resize_picture,
                                ("resized_data_size_" + str(self._width) + "x" + str(self._height)))

    def generate_pictures_for_training(self, folder_location):
        print("Generating pictures for training")

        print("Rotating pictures by 90 degrees")
        self.transform_pictures(folder_location, self.rotate_pictures_90, "rotated_90")

        print("Rotating pictures by 180 degrees")
        self.transform_pictures(folder_location, self.rotate_pictures_180, "rotated_180")

        print("Rotating pictures by 270 degrees")
        self.transform_pictures(folder_location, self.rotate_pictures_270, "rotated_270")

        print("Mirroring pictures on the x axis")
        self.transform_pictures(folder_location, self.mirror_image_x, "mirrored_x")

        print("Mirroring pictures on the y axis")
        self.transform_pictures(folder_location, self.mirror_image_y, "mirrored_y")

    def rotate_pictures_90(self, file, root):
        return self.apply_rotation(file, root, cv2.ROTATE_90_CLOCKWISE)

    def rotate_pictures_180(self, file, root):
        return self.apply_rotation(file, root, cv2.ROTATE_180)

    def rotate_pictures_270(self, file, root):
        return self.apply_rotation(file, root, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def mirror_image_x(self, file, root):
        return self.apply_flip(file, root, 0)  # Flip vertically

    def mirror_image_y(self, file, root):
        return self.apply_flip(file, root, 1)  # Flip horizontally

    def resize_picture(self, file, root):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(root, file)

            img = cv2.imread(file_path)

            if img is not None:
                # Resize the image
                new_image = cv2.resize(img, (self._width, self._height), interpolation=cv2.INTER_LANCZOS4)

                return new_image
            else:
                print(f"Failed to load image: {file_path}")

    @staticmethod
    def transform_pictures(folder_location, function, change_string_description):
        new_folder_location = folder_location + f"_{change_string_description}"

        if not os.path.exists(new_folder_location):
            os.makedirs(new_folder_location)

        for root, dirs, files in os.walk(folder_location):
            # Create a relative path from the original folder to the current folder being processed
            relative_path = os.path.relpath(root, folder_location)
            new_subfolder = os.path.join(new_folder_location, relative_path)

            # Create the corresponding subfolder in the new location
            if not os.path.exists(new_subfolder):
                os.makedirs(new_subfolder)

            for file in files:
                new_image = function(file, root)
                new_image_folder = os.path.join(new_subfolder, file)
                cv2.imwrite(new_image_folder, new_image)

    @staticmethod
    def apply_rotation(file, root, rotation_code):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(root, file)
            img = cv2.imread(file_path)
            if img is not None:
                return cv2.rotate(img, rotation_code)
            else:
                print(f"Failed to load image: {file_path}")
                return None

    @staticmethod
    def apply_flip(file, root, flip_code):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(root, file)
            img = cv2.imread(file_path)
            if img is not None:
                return cv2.flip(img, flip_code)
            else:
                print(f"Failed to load image: {file_path}")
                return None
