"""Visualization.py

include functions used to visualize the skeleton on the images

This file can also be imported as a module and contains the following
functions:

    * get_spreadsheet_cols - returns the column headers of the file
    * main - the main function of the script
"""

# Standard library imports
from typing import Tuple
import math

# Third-party library imports
import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

# Local library imports
import Person

#####################################################################################

colors = [(255, 255, 255), (0, 0, 255)]

detection_threshold = 0.1

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

# map edges to a RGB color
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): (147, 20, 255),
    (0, 2): (255, 255, 0),
    (1, 3): (147, 20, 255),
    (2, 4): (255, 255, 0),
    (0, 5): (147, 20, 255),
    (0, 6): (255, 255, 0),
    (5, 7): (147, 20, 255),
    (7, 9): (147, 20, 255),
    (6, 8): (255, 255, 0),
    (8, 10): (255, 255, 0),
    (5, 6): (0, 255, 255),
    (5, 11): (147, 20, 255),
    (6, 12): (255, 255, 0),
    (11, 12): (0, 255, 255),
    (11, 13): (147, 20, 255),
    (13, 15): (147, 20, 255),
    (12, 14): (255, 255, 0),
    (14, 16): (255, 255, 0)
}

# A list of distictive colors
COLOR_LIST = [
    (47, 79, 79),
    (139, 69, 19),
    (0, 128, 0),
    (0, 0, 139),
    (255, 0, 0),
    (255, 215, 0),
    (0, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (30, 144, 255),
    (255, 228, 181),
    (255, 105, 180),
]

def plot_3D_landmarks(images_dict, mpPose, mpDraw):
    # Run MediaPipe Pose and plot 3d pose world landmarks.
    with mpPose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=2) as pose:
        for name, image in images_dict.items():
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print the real-world 3D coordinates of nose in meters with the origin at
            # the center between hips.
            # Draw pose landmarks.
            print(f'Pose landmarks of {name}:')
            print('Nose world landmark:'),
            print(results.pose_world_landmarks.landmark[mpPose.PoseLandmark.NOSE])
            print(f'Pose landmarks of {name}:')
            print('Left + right hip landmark:')
            print(results.pose_world_landmarks.landmark[mpPose.PoseLandmark.LEFT_HIP])
            print(results.pose_world_landmarks.landmark[mpPose.PoseLandmark.RIGHT_HIP])
            print('Left + right knee landmark:'),
            print(results.pose_world_landmarks.landmark[mpPose.PoseLandmark.LEFT_KNEE])
            print(results.pose_world_landmarks.landmark[mpPose.PoseLandmark.RIGHT_KNEE])
            print('Left + right ankle landmark:'),
            print(results.pose_world_landmarks.landmark[mpPose.PoseLandmark.LEFT_ANKLE])
            print(results.pose_world_landmarks.landmark[mpPose.PoseLandmark.RIGHT_ANKLE])
            print('Left + right heel landmark:'),
            print(results.pose_world_landmarks.landmark[mpPose.PoseLandmark.LEFT_HEEL])
            print(results.pose_world_landmarks.landmark[mpPose.PoseLandmark.RIGHT_HEEL])
            print('Left + right foot index landmark:'),
            print(results.pose_world_landmarks.landmark[mpPose.PoseLandmark.LEFT_FOOT_INDEX])
            print(results.pose_world_landmarks.landmark[mpPose.PoseLandmark.RIGHT_FOOT_INDEX])

            # Plot pose world landmarks.
            mpDraw.plot_landmarks(results.pose_world_landmarks, mpPose.POSE_CONNECTIONS)

def visualize(
        image: np.ndarray,
        list_persons: list[Person],
        keypoint_color: Tuple[int, ...] = None,
        keypoint_threshold: float = 0.05,
        instance_threshold: float = 0.1) -> np.ndarray:
    """Draws landmarks and edges on the input image and return it.

  Args:
    image: The input RGB image.
    list_persons: The list of all "Person" entities to be visualize.
    keypoint_color: the colors in which the landmarks should be plotted.
    keypoint_threshold: minimum confidence score for a keypoint to be drawn.
    instance_threshold: minimum confidence score for a person to be drawn.

  Returns:
    Image with keypoints and edges.
  """

    for person in list_persons:
        if person.score < instance_threshold:
            continue

        keypoints = person.keypoints
        #     bounding_box = person.bounding_box
        bounding_box = None

        # Assign a color to visualize keypoints.
        if keypoint_color is None:
            if person.id is None:
                # If there's no person id, which means no tracker is enabled, use
                # a default color.
                person_color = (0, 255, 0)
            else:
                # If there's a person id, use different color for each person.
                person_color = COLOR_LIST[person.id % len(COLOR_LIST)]
        else:
            person_color = keypoint_color

        # Draw all the landmarks
        for i in range(len(keypoints)):
            if keypoints[i].score >= keypoint_threshold:
                cv2.circle(image, keypoints[i].coordinate, 2, person_color, 4)

        # Draw all the edges
        for edge_pair, edge_color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
            if (keypoints[edge_pair[0]].score > keypoint_threshold and
                    keypoints[edge_pair[1]].score > keypoint_threshold):
                cv2.line(image, keypoints[edge_pair[0]].coordinate,
                         keypoints[edge_pair[1]].coordinate, edge_color, 2)

        # Draw bounding_box with multipose
        if bounding_box is not None:
            start_point = bounding_box.start_point
            end_point = bounding_box.end_point
            cv2.rectangle(image, start_point, end_point, person_color, 2)
            # Draw id text when tracker is enabled for MoveNet MultiPose model.
            # (id = None when using single pose model or when tracker is None)
            if person.id:
                id_text = 'id = ' + str(person.id)
                cv2.putText(image, id_text, start_point, cv2.FONT_HERSHEY_PLAIN, 1,
                            (0, 0, 255), 1)

    return image


def keep_aspect_ratio_resizer(
        image: np.ndarray, target_size: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Resizes the image.

  The function resizes the image such that its longer side matches the required
  target_size while keeping the image aspect ratio. Note that the resizes image
  is padded such that both height and width are a multiple of 32, which is
  required by the model. See
  https://tfhub.dev/google/tfjs-model/movenet/multipose/lightning/1 for more
  detail.

  Args:
    image: The input RGB image as a numpy array of shape [height, width, 3].
    target_size: Desired size that the image should be resize to.

  Returns:
    image: The resized image.
    (target_height, target_width): The actual image size after resize.

  """
    height, width, _ = image.shape
    if height > width:
        scale = float(target_size / height)
        target_height = target_size
        scaled_width = math.ceil(width * scale)
        image = cv2.resize(image, (scaled_width, target_height))
        target_width = int(math.ceil(scaled_width / 32) * 32)
    else:
        scale = float(target_size / width)
        target_width = target_size
        scaled_height = math.ceil(height * scale)
        image = cv2.resize(image, (target_width, scaled_height))
        target_height = int(math.ceil(scaled_height / 32) * 32)

    padding_top, padding_left = 0, 0
    padding_bottom = target_height - image.shape[0]
    padding_right = target_width - image.shape[1]
    # add padding to image
    image = cv2.copyMakeBorder(image, padding_top, padding_bottom, padding_left,
                               padding_right, cv2.BORDER_CONSTANT)
    return image, (target_height, target_width)


def draw_prediction_on_image(image, person, crop_region=None, close_figure=True,
                             keep_input_size=False):
    # Draw the detection result on top of the image.
    image_np = visualize(image, [person])

    # Plot the image with detection results.
    height, width, channel = image.shape
    aspect_ratio = float(width) / height
    fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
    im = ax.imshow(image_np)

    if close_figure:
        plt.close(fig)

    if not keep_input_size:
        image_np = keep_aspect_ratio_resizer(image_np, (512, 512))

    return image_np


def get_keypoint_landmarks(person):
    pose_landmarks = np.array(
        [[keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score]
         for keypoint in person.keypoints],
        dtype=np.float32)
    return pose_landmarks


def img_to_tensor(im):
    # im = tf.convert_to_tensor(im, dtype=tf.uint8)
    im = tf.convert_to_tensor(im, dtype=tf.float32)
    return im


def draw_connection(frame, keypoints, edges, color, confidence_threshold=0.1, thickness=1):
    for edge, _ in edges.items():
        p1, p2 = edge
        y1, x1, c1 = keypoints[p1]
        y2, x2, c2 = keypoints[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(y1), int(x1)), (int(y2), int(x2)), color, thickness)


def draw_keypoints(frame, keypoints, color, confidence_threshold=0.1, thickness=3, fill_inside=1):
    for kp in keypoints:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(ky), int(kx)), thickness, color, fill_inside)
