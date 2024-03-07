from PIL import Image
import os
import cv2
import numpy as np
import requests
import io
import math


import random


def ensure_3_channels(image):
    if len(image.shape) == 2:  # Grayscale
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def maintain_dimensions(original_image, new_image):

    def zoom_to_fit(image, new_width, new_height):
        """
        Zooms an image to fit new dimensions, maintaining aspect ratio.

        Args:
        image (np.array): The original image.
        new_width (int): The new width.
        new_height (int): The new height.

        Returns:
        np.array: The resized (zoomed in) image.
        """
        image = ensure_3_channels(image)
        # Original dimensions
        original_height, original_width = image.shape[:2]

        # Calculating aspect ratios
        original_aspect = original_width / original_height
        new_aspect = new_width / new_height

        # Determine the scaling factors for width and height
        if original_aspect > new_aspect:
            # Original is wider relative to new dimensions
            scale_factor = new_height / original_height
        else:
            # Original is taller relative to new dimensions
            scale_factor = new_width / original_width

        # Calculate new dimensions based on the scale factor
        scaled_width = int(original_width * scale_factor)
        scaled_height = int(original_height * scale_factor)

        # Resize (zoom in) the image
        scaled_image = cv2.resize(
            image, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)

        # Crop the scaled image to the new dimensions
        start_x = max(0, (scaled_width - new_width) // 2)
        start_y = max(0, (scaled_height - new_height) // 2)
        cropped_image = scaled_image[start_y:start_y +
                                     new_height, start_x:start_x + new_width]

        return cropped_image

    return zoom_to_fit(new_image, *original_image.shape[:2])


def apply_random_blur_cv(image, factors=(1, 20)):
    image = ensure_3_channels(image)

    # Ensure odd number for kernel size
    blur_radius = random.randint(*factors) * 2 + 1
    blurred_image = cv2.GaussianBlur(image, (blur_radius, blur_radius), 0)
    return blurred_image


def apply_random_transparency_cv(image, factors=(0.01, 1.00)):
    image = ensure_3_channels(image)

    transparency = random.uniform(*factors)  # Random transparency level
    if len(image.shape) == 2:  # Grayscale image, add a dimension
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif image.shape[2] == 3:  # Convert BGR to BGRA
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    alpha_channel = image[:, :, 3] * transparency
    image[:, :, 3] = alpha_channel.astype(np.uint8)
    return image


def apply_color_grading_cv(image, factors=(0.2, 2.0)):
    image = ensure_3_channels(image)
    # Assuming image is in BGR format
    for channel in range(3):
        factor = random.uniform(*factors)
        image[:, :, channel] = np.clip(image[:, :, channel] * factor, 0, 255)
    return image


def apply_color_tint_cv(image):
    image = ensure_3_channels(image)

    tint_color = np.array([random.randint(0, 255)
                          for _ in range(3)])  # BGR tint
    opacity = random.randint(10, 90) / 100  # Tint opacity

    original_alpha = None
    if image.shape[2] == 3:  # Convert BGR to BGRA
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    else:
        # Save original alpha channel if present
        original_alpha = image[:, :, 3]

    image[:, :, :3] = np.clip(
        (1 - opacity) * image[:, :, :3] + opacity * tint_color, 0, 255)

    if original_alpha is not None:  # If the original alpha channel was saved
        image[:, :, 3] = original_alpha  # Restore the original alpha channel

    return image


def apply_random_rotation_cv(image):
    def rotate_image(image, angle):
        """
        Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
        (in degrees). The returned image will be large enough to hold the entire
        new image, with a black background
        """

        # Get the image size
        # No that's not an error - NumPy stores image matricies backwards
        image_size = (image.shape[1], image.shape[0])
        image_center = tuple(np.array(image_size) / 2)

        # Convert the OpenCV 3x2 rotation matrix to 3x3
        rot_mat = np.vstack(
            [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
        )

        rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

        # Shorthand for below calcs
        image_w2 = image_size[0] * 0.5
        image_h2 = image_size[1] * 0.5

        # Obtain the rotated coordinates of the image corners
        rotated_coords = [
            (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
            (np.array([image_w2,  image_h2]) * rot_mat_notranslate).A[0],
            (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
            (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
        ]

        # Find the size of the new image
        x_coords = [pt[0] for pt in rotated_coords]
        x_pos = [x for x in x_coords if x > 0]
        x_neg = [x for x in x_coords if x < 0]

        y_coords = [pt[1] for pt in rotated_coords]
        y_pos = [y for y in y_coords if y > 0]
        y_neg = [y for y in y_coords if y < 0]

        right_bound = max(x_pos)
        left_bound = min(x_neg)
        top_bound = max(y_pos)
        bot_bound = min(y_neg)

        new_w = int(abs(right_bound - left_bound))
        new_h = int(abs(top_bound - bot_bound))

        # We require a translation matrix to keep the image centred
        trans_mat = np.matrix([
            [1, 0, int(new_w * 0.5 - image_w2)],
            [0, 1, int(new_h * 0.5 - image_h2)],
            [0, 0, 1]
        ])

        # Compute the tranform for the combined rotation and translation
        affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

        # Apply the transform
        result = cv2.warpAffine(
            image,
            affine_mat,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR
        )

        return result

    def largest_rotated_rect(w, h, angle):
        """
        Given a rectangle of size wxh that has been rotated by 'angle' (in
        radians), computes the width and height of the largest possible
        axis-aligned rectangle within the rotated rectangle.

        Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

        Converted to Python by Aaron Snoswell
        """

        quadrant = int(math.floor(angle / (math.pi / 2))) & 3
        sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
        alpha = (sign_alpha % math.pi + math.pi) % math.pi

        bb_w = w * math.cos(alpha) + h * math.sin(alpha)
        bb_h = w * math.sin(alpha) + h * math.cos(alpha)

        gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

        delta = math.pi - alpha - gamma

        length = h if (w < h) else w

        d = length * math.cos(alpha)
        a = d * math.sin(alpha) / math.sin(delta)

        y = a * math.cos(gamma)
        x = y * math.tan(gamma)

        return (
            bb_w - 2 * x,
            bb_h - 2 * y
        )

    def crop_around_center(image, width, height):
        """
        Given a NumPy / OpenCV 2 image, crops it to the given width and height,
        around it's centre point
        """

        image_size = (image.shape[1], image.shape[0])
        image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

        if (width > image_size[0]):
            width = image_size[0]

        if (height > image_size[1]):
            height = image_size[1]

        x1 = int(image_center[0] - width * 0.5)
        x2 = int(image_center[0] + width * 0.5)
        y1 = int(image_center[1] - height * 0.5)
        y2 = int(image_center[1] + height * 0.5)

        return image[y1:y2, x1:x2]

    image_height, image_width = image.shape[0:2]
    random_angle = random.randint(0, 360)

    image_rotated = rotate_image(image, random_angle)
    image_rotated_and_cropped = crop_around_center(
        image_rotated,
        *largest_rotated_rect(
            image_width,
            image_height,
            math.radians(random_angle)
        )
    )
    return maintain_dimensions(image, image_rotated_and_cropped)


def apply_random_scaling_cv(image, factors=(1, 5)):
    image = ensure_3_channels(image)
    scale_factor = random.uniform(*factors)

    # Scale the image
    scaled_image = cv2.resize(
        image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    scaled_height, scaled_width = scaled_image.shape[:2]

    # Original image size
    original_height, original_width = image.shape[:2]

    # Ensure that the crop area does not exceed the scaled image dimensions
    max_x = max(scaled_width - original_width, 0)
    max_y = max(scaled_height - original_height, 0)

    # Randomly choose the top-left corner of the cropping area
    start_x = random.randint(0, max_x)
    start_y = random.randint(0, max_y)

    # Crop the scaled image to simulate random zoom and shift
    cropped_image = scaled_image[start_y:start_y +
                                 original_height, start_x:start_x + original_width]
    return maintain_dimensions(image, cropped_image)


def apply_mirror_effect_cv(image):
    image = ensure_3_channels(image)
    # Always consider both directions
    flip_directions = [random.choice([-1, 0, 1]) for _ in range(2)]
    for flip_direction in flip_directions:
        image = cv2.flip(image, flip_direction)
    return image


def apply_gaussian_blur(image, factors=(1, 10)):
    image = ensure_3_channels(image)
    blur_radius = random.randint(*factors)
    # Ensure the blur radius is odd
    blur_radius = blur_radius if blur_radius % 2 != 0 else blur_radius + 1
    return cv2.GaussianBlur(image, (blur_radius, blur_radius), 0)


def apply_motion_blur(image,  factors=(1, 10)):
    image = ensure_3_channels(image)
    blur_extent = random.randint(*factors)  # Extent of motion blur
    kernel_motion_blur = np.zeros((blur_extent, blur_extent))
    kernel_motion_blur[int((blur_extent - 1) / 2), :] = np.ones(blur_extent)
    kernel_motion_blur = kernel_motion_blur / blur_extent
    return cv2.filter2D(image, -1, kernel_motion_blur)


def apply_box_blur(image,  factors=(1, 10)):
    image = ensure_3_channels(image)
    blur_size = random.randint(*factors)  # Size of the blur
    return cv2.blur(image, (blur_size, blur_size))


def adjust_brightness(image, factors=(0.1, 2.0)):
    image = ensure_3_channels(image)
    # Check if image is grayscale (single channel)
    if len(image.shape) == 2 or image.shape[2] == 1:
        # For grayscale images, adjust brightness directly
        brightness_factor = random.uniform(
            *factors)  # Random brightness factor
        return np.clip(image * brightness_factor, 0, 255).astype(np.uint8)
    else:
        # For BGR color images, adjust brightness in HSV space
        brightness_factor = random.uniform(
            *factors)  # Random brightness factor
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_image[:, :, 2] = cv2.multiply(
            hsv_image[:, :, 2], brightness_factor)
        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


def adjust_contrast(image,  factors=(0.6, 1.4)):
    image = ensure_3_channels(image)
    contrast_factor = random.uniform(*factors)  # Random contrast factor
    f = 131 * (contrast_factor - 0.5)
    alpha_c = f / 127 + 1
    gamma_c = 127 * (1 - alpha_c)
    return cv2.addWeighted(image, alpha_c, image, 0, gamma_c)


def adjust_saturation(image, factors=(0.1, 2.0)):
    image = ensure_3_channels(image)
    # Check if image is grayscale (single channel)
    if len(image.shape) < 3 or image.shape[2] == 1:
        # Option 1: Return the original image if it's grayscale
        # return image

        # Option 2: Convert grayscale to BGR before adjusting saturation
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    saturation_factor = random.uniform(*factors)  # Random saturation factor
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 1] = cv2.multiply(hsv_image[:, :, 1], saturation_factor)
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


# def convert_to_grayscale(image):
# TODO
#     image = ensure_3_channels(image)
#     grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # return to 3 channels
#     return cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)

def apply_global_transparency(image, factors=(0.1, 1)):
    image = ensure_3_channels(image)
    transparency_factor = random.uniform(*factors)  # Random transparency level
    if len(image.shape) == 2:  # Convert grayscale to BGRA
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif image.shape[2] == 3:  # Convert BGR to BGRA
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    image[:, :, 3] = int(255 * transparency_factor)  # Set transparency
    return image


# Update the effects list
def apply_random_effects_cv(image, num_effects=-1):
    image = ensure_3_channels(image)
    effects = [
        apply_gaussian_blur,
        apply_motion_blur,
        apply_box_blur,
        # # Blur effects
        # adjust_brightness,
        # adjust_contrast,
        adjust_saturation,
        # convert_to_grayscale,
        # # # Color adjustments
        # apply_global_transparency,  # Transparency # Apply this effect separately
        apply_random_rotation_cv,
        apply_mirror_effect_cv,
        apply_random_scaling_cv  # Rotational and scaling effects
    ]

    # If num_effects is -1, choose a random number of effects
    if num_effects == -1:
        num_effects = random.randint(1, len(effects))

    selected_effects = random.sample(effects, min(num_effects, len(effects)))

    for effect in selected_effects:
        image = effect(image)
    return image


def test():
    # Function to download and save the original image
    def download_image(url, filename):
        response = requests.get(url)
        image = Image.open(io.BytesIO(response.content))
        image.save(filename)
        # Convert PIL image to OpenCV format
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Download and save the original image
    url = 'https://picsum.photos/1024/768'  # Example resolution
    original_image = download_image(url, 'original_image.jpg')

    # Define all individual effects
    individual_effects = {
        'gaussian_blur': apply_gaussian_blur,
        'motion_blur': apply_motion_blur,
        'box_blur': apply_box_blur,
        # 'brightness_adjustment': adjust_brightness,
        'contrast_adjustment': adjust_contrast,
        'saturation_adjustment': adjust_saturation,
        # 'grayscale_conversion': convert_to_grayscale,
        'global_transparency': apply_global_transparency,
        'random_blur': apply_random_blur_cv,
        'random_transparency': apply_random_transparency_cv,
        'color_grading': apply_color_grading_cv,
        'color_tint': apply_color_tint_cv,
        'random_rotation': apply_random_rotation_cv,
        'random_scaling': apply_random_scaling_cv,
        'mirror_effect': apply_mirror_effect_cv
    }

    # Apply each individual effect and save the result
    for effect_name, effect_func in individual_effects.items():
        result = effect_func(original_image.copy())
        cv2.imwrite(f'{effect_name}_image.jpg', result)

    # Apply all effects together and save the result
    all_effects_image = apply_random_effects_cv(
        original_image.copy(), len(individual_effects))
    cv2.imwrite('all_effects_image.jpg', all_effects_image)

    print("Images processed and saved successfully.")


def compress_images(directory, output_dir, quality=85, resize_factor=0.5):
    """
    Compress all images in the given directory and save them in the output directory.
    :param directory: Path to the directory containing images.
    :param output_dir: Path to the directory where compressed images will be saved.
    :param quality: Quality of the saved image, between 0 and 100.
    :param resize_factor: Factor to resize the image (0.5 reduces size to 50%).
    """
    # Create a directory for compressed images
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = [f for f in os.listdir(directory) if f.lower().endswith(
        ('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    for filename in images:
        filepath = os.path.join(directory, filename)
        with Image.open(filepath) as img:
            # Resize image
            new_size = tuple(int(dim * resize_factor) for dim in img.size)
            img = img.resize(new_size, Image.ANTIALIAS)

            # Save image with compression
            output_filepath = os.path.join(output_dir, filename)
            img.save(output_filepath, quality=quality, optimize=True)


if __name__ == "__main__":
    # Replace with your input directory path
    input_directory = "both_compressed_blocking_images"
    # Replace with your desired output directory path
    output_directory = "DOUBLE_both_compressed_blocking_images"
    compress_images(input_directory, output_directory,
                    quality=50, resize_factor=0.5)
