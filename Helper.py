"""Helper.py

This script allows usage of the the Movenet model with various functions

This file can also be imported as a module and contains the following
functions:

    * get_spreadsheet_cols - returns the column headers of the file
    * main - the main function of the script
"""

# Standard library imports
import math

# Third-party library imports
import pandas as pd
import tensorflow as tf
import keras
import numpy as np
import cv2
import mediapipe as mp

# Local library imports
from Person import BodyPart
from Visualization import img_to_tensor, colors, EDGES, draw_connection, draw_keypoints

# Declare some mediapipe variables
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
points = mpPose.PoseLandmark
mpHolistic = mp.solutions.holistic
#####################################################################################


def load_pose_landmarks(csv_path):
  """Loads a CSV created by MoveNetPreprocessor.

  Returns:
    X: Detected landmark coordinates and scores of shape (N, 17 * 3)
    y: Ground truth labels of shape (N, label_count)
    classes: The list of all class names found in the dataset
    dataframe: The CSV loaded as a Pandas dataframe features (X) and ground
      truth labels (y) to use later to train a pose classification model.
  """

  # Load the CSV file
  dataframe = pd.read_csv(csv_path)
  df_to_process = dataframe.copy()

  # Drop the file_name columns as you don't need it during training.
  df_to_process.drop(columns=['file_name'], inplace=True)

  # Extract the list of class names
  classes = df_to_process.pop('class_name').unique()

  # Extract the labels
  y = df_to_process.pop('class_no')

  # Convert the input features and labels into the correct format for training.
  X = df_to_process.astype('float64')
  y = keras.utils.to_categorical(y)

  return X, y, classes, dataframe


def get_center_point(landmarks, left_bodypart, right_bodypart):
    """Calculates the center point of the two given landmarks."""
    left = tf.gather(landmarks, left_bodypart.value, axis=1)
    right = tf.gather(landmarks, right_bodypart.value, axis=1)
    center = left * 0.5 + right * 0.5
    return center


def get_pose_landmarks():
    """Get pose landmarks

    Args:
      path:

    Returns: An array of desired pose landmarks
    """
    p = mpHolistic.PoseLandmark
    pose_landmarks = [
        p.LEFT_SHOULDER, p.RIGHT_SHOULDER,
        p.LEFT_ELBOW, p.RIGHT_ELBOW,
        p.LEFT_WRIST, p.RIGHT_WRIST,
        p.LEFT_HIP, p.RIGHT_HIP,
        p.LEFT_KNEE, p.RIGHT_KNEE,
        p.LEFT_ANKLE, p.RIGHT_ANKLE,
        p.LEFT_FOOT_INDEX, p.RIGHT_FOOT_INDEX,
        p.LEFT_THUMB, p.RIGHT_THUMB,
        p.LEFT_INDEX, p.RIGHT_INDEX,
        p.LEFT_PINKY, p.RIGHT_PINKY,
    ]

    return pose_landmarks
def get_pose_size(landmarks, torso_size_multiplier=2.5):
    """Calculates pose size.
    It is the maximum of two values:
    * Torso size multiplied by `torso_size_multiplier`
    * Maximum distance from pose center to any pose landmark
    """
    # Hips center
    hips_center = get_center_point(landmarks, BodyPart.LEFT_HIP,
                                   BodyPart.RIGHT_HIP)

    # Shoulders center
    shoulders_center = get_center_point(landmarks, BodyPart.LEFT_SHOULDER,
                                        BodyPart.RIGHT_SHOULDER)

    # Torso size as the minimum body size
    torso_size = tf.linalg.norm(shoulders_center - hips_center)
    # Pose center
    pose_center_new = get_center_point(landmarks, BodyPart.LEFT_HIP,
                                       BodyPart.RIGHT_HIP)
    pose_center_new = tf.expand_dims(pose_center_new, axis=1)
    # Broadcast the pose center to the same size as the landmark vector to
    # perform substraction
    pose_center_new = tf.broadcast_to(pose_center_new,
                                      [tf.size(landmarks) // (17 * 2), 17, 2])

    # Dist to pose center
    d = tf.gather(landmarks - pose_center_new, 0, axis=0,
                  name="dist_to_pose_center")
    # Max dist to pose center
    max_dist = tf.reduce_max(tf.linalg.norm(d, axis=0))

    # Normalize scale
    pose_size = tf.maximum(torso_size * torso_size_multiplier, max_dist)
    return pose_size


def normalize_pose_landmarks(landmarks):
    """Normalizes the landmarks translation by moving the pose center to (0,0) and
    scaling it to a constant pose size.
  """
    # Move landmarks so that the pose center becomes (0,0)
    pose_center = get_center_point(landmarks, BodyPart.LEFT_HIP,
                                   BodyPart.RIGHT_HIP)

    pose_center = tf.expand_dims(pose_center, axis=1)
    # Broadcast the pose center to the same size as the landmark vector to perform
    # substraction
    pose_center = tf.broadcast_to(pose_center,
                                  [tf.size(landmarks) // (17 * 2), 17, 2])
    landmarks = landmarks - pose_center

    # Scale the landmarks to a constant pose size
    pose_size = get_pose_size(landmarks)
    landmarks /= pose_size
    return landmarks


def landmarks_to_embedding(landmarks_and_scores):
    """Converts the input landmarks into a pose embedding."""
    # Reshape the flat input into a matrix with shape=(17, 3)
    reshaped_inputs = keras.layers.Reshape((17, 3))(landmarks_and_scores)

    # Normalize landmarks 2D
    landmarks = normalize_pose_landmarks(reshaped_inputs[:, :, :2])
    # Flatten the normalized landmark coordinates into a vector
    embedding = keras.layers.Flatten()(landmarks)
    return embedding


def preprocess_data(X_test):
    embedding = landmarks_to_embedding(tf.reshape(tf.convert_to_tensor(X_test), (1, 51)))
    processed_X_test = (tf.reshape(embedding, (34)))
    processed_X_test = tf.convert_to_tensor(processed_X_test)
    processed_X_test = tf.expand_dims(processed_X_test, axis=0)
    return processed_X_test

def process_model(frame, yoga_pose_img, class_id, model):
    yoga_x, yoga_y, _ = yoga_pose_img.shape
    # Detect pose keypoints:
    input_img = img_to_tensor(frame)
    keypoints_with_scores = detect(input_img)

    pose_landmarks = [[int(keypoint.coordinate.x), int(keypoint.coordinate.y), keypoint.score] for keypoint in
                      keypoints_with_scores.keypoints]
    input_landmarks = np.array(pose_landmarks, dtype=np.float32).flatten()
    model_input = preprocess_data(input_landmarks)
    result = model.predict(model_input)

    overlap_imgs = overlap_a_b(frame[50:yoga_y + 50, 0:yoga_x], yoga_pose_img)
    frame[50:yoga_y + 50, 0:yoga_x] = overlap_imgs

    if (np.argmax(result)) == class_id:

        draw_keypoints(frame, pose_landmarks, colors[0])
        draw_connection(frame, pose_landmarks, EDGES, colors[0])

    else:
        draw_keypoints(frame, pose_landmarks, colors[1])
        draw_connection(frame, pose_landmarks, EDGES, colors[1])

    cv2.putText(frame,
                ('Pose: {} with a {:.2f} %. '.format(pose_classes[np.argmax(result)], 100 * np.max(result))),
                (5, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2)
def overlap_a_b(a, b):
    alpha = 0.5
    beta = 0.5
    gamma = 0
    out_img = cv2.addWeighted(a, alpha, b, beta, gamma)

    return out_img


# loading final csv file
def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    # df.drop(['filename'],axis=0, inplace=True)
    classes = df.pop('class_name').unique()
    y = df.pop('class_no')

    X = df.astype('float64')
    y = keras.utils.to_categorical(y)

    return X, y, classes


def get_center_point(landmarks, left_bodypart, right_bodypart):
    """Calculates the center point of the two given landmarks."""
    left = tf.gather(landmarks, left_bodypart.value, axis=1)
    right = tf.gather(landmarks, right_bodypart.value, axis=1)
    center = left * 0.5 + right * 0.5
    return center


def get_pose_size(landmarks, torso_size_multiplier=2.5):
    """Calculates pose size.
    It is the maximum of two values:
    * Torso size multiplied by `torso_size_multiplier`
    * Maximum distance from pose center to any pose landmark
    """
    # Hips center
    hips_center = get_center_point(landmarks, BodyPart.LEFT_HIP,
                                   BodyPart.RIGHT_HIP)

    # Shoulders center
    shoulders_center = get_center_point(landmarks, BodyPart.LEFT_SHOULDER,
                                        BodyPart.RIGHT_SHOULDER)

    # Torso size as the minimum body size
    torso_size = tf.linalg.norm(shoulders_center - hips_center)
    # Pose center
    pose_center_new = get_center_point(landmarks, BodyPart.LEFT_HIP,
                                       BodyPart.RIGHT_HIP)
    pose_center_new = tf.expand_dims(pose_center_new, axis=1)
    # Broadcast the pose center to the same size as the landmark vector to
    # perform substraction
    pose_center_new = tf.broadcast_to(pose_center_new,
                                      [tf.size(landmarks) // (17 * 2), 17, 2])

    # Dist to pose center
    d = tf.gather(landmarks - pose_center_new, 0, axis=0,
                  name="dist_to_pose_center")
    # Max dist to pose center
    max_dist = tf.reduce_max(tf.linalg.norm(d, axis=0))

    # Normalize scale
    pose_size = tf.maximum(torso_size * torso_size_multiplier, max_dist)
    return pose_size


def normalize_pose_landmarks(landmarks):
    """Normalizes the landmarks translation by moving the pose center to (0,0) and
    scaling it to a constant pose size.
  """
    # Move landmarks so that the pose center becomes (0,0)
    pose_center = get_center_point(landmarks, BodyPart.LEFT_HIP,
                                   BodyPart.RIGHT_HIP)

    pose_center = tf.expand_dims(pose_center, axis=1)
    # Broadcast the pose center to the same size as the landmark vector to perform
    # substraction
    pose_center = tf.broadcast_to(pose_center,
                                  [tf.size(landmarks) // (17 * 2), 17, 2])
    landmarks = landmarks - pose_center

    # Scale the landmarks to a constant pose size
    pose_size = get_pose_size(landmarks)
    landmarks /= pose_size
    return landmarks


def landmarks_to_embedding(landmarks_and_scores):
    """Converts the input landmarks into a pose embedding."""
    # Reshape the flat input into a matrix with shape=(17, 3)
    reshaped_inputs = keras.layers.Reshape((17, 3))(landmarks_and_scores)

    # Normalize landmarks 2D
    landmarks = normalize_pose_landmarks(reshaped_inputs[:, :, :2])
    # Flatten the normalized landmark coordinates into a vector
    embedding = keras.layers.Flatten()(landmarks)
    return embedding


def preprocess_data(X_train):
    """prerporcess the training datasett"""
    processed_X_train = []
    for i in range(X_train.shape[0]):
        embedding = landmarks_to_embedding(tf.reshape(tf.convert_to_tensor(X_train.iloc[i]), (1, 51)))
        processed_X_train.append(tf.reshape(embedding, (34)))
    return tf.convert_to_tensor(processed_X_train)


"""def image_resize(image):
#     screen_res = 1760, 990
    screen_res = 1280, 720
    scale_width = screen_res[0] / image.shape[1]
    scale_height = screen_res[1] / image.shape[0]
    scale = min(scale_width, scale_height)
    #resized window width and height
    window_width = int(image.shape[1] * scale)
    window_height = int(image.shape[0] * scale)
    resized_img=cv2.resize(image, (window_width,window_height))
    return resized_img"""

def image_resize(image):
    """ Resize the image

    Args:
      path: desired image

    Returns: The resized image
    """
    # Set the ideal dimensions
    ideal_dim = 640, 480
    # Get the image width and height from the image shape
    (image_height, image_width) = image.shape[:2]
    # Get the ratio of the image compared to the ideal dimensions
    ratio_width = ideal_dim[0] / image_width
    ratio_height = ideal_dim[1] / image_height
    # Get the ratio based on whether the ratio of the width or height is smaller
    scale = min(ratio_width, ratio_height)
    # Resize window width and height
    window_width = int(image_width * scale)
    window_height = int(image_height * scale)
    resized_img=cv2.resize(image, (window_width, window_height))
    return resized_img

def find_mid(array1,array2):
    """Find middle of 2 points in the 3D Plane

    Args:
      path: two arrays that each define the x, y, z coordinates of two specific landmarks

    Returns: the middle of the two arrays
    """
    # First array
    array1 = np.array(array1)
    # Second array
    array2 = np.array(array2)
    return (array1 + array2)/2


def distance(array1,array2):
    """Find distance of x y z points in the 3D Plane

    Args:
      path: two arrays that each define the x, y, z coordinates of two specific landmarks

    Returns: the distance of the two points
    """
    # First array
    array1 = np.array(array1)
    # Second array
    array2 = np.array(array2)
    return ( ((array1[0]-array2[0])**2) + ((array1[1]-array2[1])**2) + ((array1[2]-array2[2])**2) ) **0.5

def magnitude(a):
    """Find magnitude of the array

    Args:
      path: an array

    Returns: Square of each coordinate to the power of 0.5 to find the magnitude
    """
    a = np.array(a)
    return ((a[0]**2)+(a[1]**2)+(a[2]**2))**0.5

def get_line(array1,array2):
    """Find the line between the two arrays

    Args:
      path: two arrays that represent the coordinates of the points

    Returns: The difference between the two arrays
    """
    array1 = np.array(array1) # First
    array2 = np.array(array2) # Second
    return np.subtract(array1, array2)


def get_joints(landmarks):
    """ Get midpoint and points of joints of head(mid point of eyes), neck, left shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist, left hip, mid hip, right hip,
    left knee, right knee, left ankle, right ankle

    Args:
      path: desired landmarks

    Returns: A dictionary of joints
    """

    # Commented out because we are not using the eyes
    # joints = head(mid point of eyes), neck, left shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist, left hip, mid hip, right hip, left knee, right knee, left ankle, right ankle
    joints = {}

    # Numbers represent the index in the joints dictionary
    # 0. Mid left and right eye - represents the Head
    joints['MID_LEFTEYE_RIGHTEYE'] = find_mid(
        [landmarks[mpPose.PoseLandmark.LEFT_EYE.value].x, landmarks[mpPose.PoseLandmark.LEFT_EYE.value].y,
         landmarks[mpPose.PoseLandmark.LEFT_EYE.value].z],
        [landmarks[mpPose.PoseLandmark.RIGHT_EYE.value].x, landmarks[mpPose.PoseLandmark.RIGHT_EYE.value].y,
         landmarks[mpPose.PoseLandmark.RIGHT_EYE.value].z])

    # 1. Mid left and right shoulder - represents the Neck
    joints['MID_LEFTSHOULDER_RIGHTSHOULDER'] = find_mid(
        [landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].y,
         landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].z],
        [landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].y,
         landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].z])

    # 2. Left shoulder
    joints['LEFTSHOULDER'] = np.array(
        [landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].y,
         landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].z])
    # 3. Right shoulder
    joints['RIGHTSHOULDER'] = np.array(
        [landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].y,
         landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].z])

    # 4. Left elbow
    joints['LEFTELBOW'] = np.array(
        [landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].y,
         landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].z])
    # 5. Right elbow
    joints['RIGHTELBOW'] = np.array(
        [landmarks[mpPose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mpPose.PoseLandmark.RIGHT_ELBOW.value].y,
         landmarks[mpPose.PoseLandmark.RIGHT_ELBOW.value].z])

    # 6. Left wrist
    joints['LEFTWRIST'] = np.array(
        [landmarks[mpPose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mpPose.PoseLandmark.LEFT_WRIST.value].y,
         landmarks[mpPose.PoseLandmark.LEFT_WRIST.value].z])
    # 7. Right wrist
    joints['RIGHTWRIST'] = np.array(
        [landmarks[mpPose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mpPose.PoseLandmark.RIGHT_WRIST.value].y,
         landmarks[mpPose.PoseLandmark.RIGHT_WRIST.value].z])

    # 8. Left hip
    joints['LEFTHIP'] = np.array(
        [landmarks[mpPose.PoseLandmark.LEFT_HIP.value].x, landmarks[mpPose.PoseLandmark.LEFT_HIP.value].y,
         landmarks[mpPose.PoseLandmark.LEFT_HIP.value].z])
    # 9. Right hip
    joints['RIGHTHIP'] = np.array(
        [landmarks[mpPose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mpPose.PoseLandmark.RIGHT_HIP.value].y,
         landmarks[mpPose.PoseLandmark.RIGHT_HIP.value].z])

    # 10. Mid left and right shoulder
    joints['MID_LEFTHIP_RIGHTHIP'] = find_mid(
        [landmarks[mpPose.PoseLandmark.LEFT_HIP.value].x, landmarks[mpPose.PoseLandmark.LEFT_HIP.value].y,
         landmarks[mpPose.PoseLandmark.LEFT_HIP.value].z],
        [landmarks[mpPose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mpPose.PoseLandmark.RIGHT_HIP.value].y,
         landmarks[mpPose.PoseLandmark.RIGHT_HIP.value].z])

    # 11. Left knee
    joints['LEFTKNEE'] = np.array(
        [landmarks[mpPose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mpPose.PoseLandmark.LEFT_KNEE.value].y,
         landmarks[mpPose.PoseLandmark.LEFT_KNEE.value].z])
    # 12. Right knee
    joints['RIGHTKNEE'] = np.array(
        [landmarks[mpPose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mpPose.PoseLandmark.RIGHT_KNEE.value].y,
         landmarks[mpPose.PoseLandmark.RIGHT_KNEE.value].z])

    # 13. Left ankle
    joints['LEFTANKLE'] = np.array(
        [landmarks[mpPose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mpPose.PoseLandmark.LEFT_ANKLE.value].y,
         landmarks[mpPose.PoseLandmark.LEFT_ANKLE.value].z])
    # 14. Right ankle
    joints['RIGHTANKLE'] = np.array(
        [landmarks[mpPose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mpPose.PoseLandmark.RIGHT_ANKLE.value].y,
         landmarks[mpPose.PoseLandmark.RIGHT_ANKLE.value].z])

    # 15. Left foot index
    joints['LEFTFOOTINDEX'] = np.array(
        [landmarks[mpPose.PoseLandmark.LEFT_FOOT_INDEX.value].x, landmarks[mpPose.PoseLandmark.LEFT_FOOT_INDEX.value].y,
         landmarks[mpPose.PoseLandmark.LEFT_FOOT_INDEX.value].z])
    # 16. Right foot index
    joints['RIGHTFOOTINDEX'] = np.array([landmarks[mpPose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                                         landmarks[mpPose.PoseLandmark.RIGHT_FOOT_INDEX.value].y,
                                         landmarks[mpPose.PoseLandmark.RIGHT_FOOT_INDEX.value].z])

    return joints


def get_body_lines(joints):
    """ Get body lines based on the different points and some mid points

    Args:
      path: joint coordinates used to then define the lines

    Returns: A dictionary of body lines
    """
    body_lines = {}
    # Head to neck
    body_lines['HEAD_TO_NECK'] = get_line(joints['MID_LEFTEYE_RIGHTEYE'], joints['MID_LEFTSHOULDER_RIGHTSHOULDER'])

    # Neck to hip
    body_lines['NECK_TO_HIP'] = get_line(joints['MID_LEFTSHOULDER_RIGHTSHOULDER'], joints['MID_LEFTHIP_RIGHTHIP'])

    # Side hips to mid hip
    body_lines['LEFTHIP_TO_MIDHIP'] = get_line(joints['LEFTHIP'], joints['MID_LEFTHIP_RIGHTHIP'])
    body_lines['RIGHTHIP_TO_MIDHIP'] = get_line(joints['RIGHTHIP'], joints['MID_LEFTHIP_RIGHTHIP'])

    # Side hips to side knees
    body_lines['LEFTHIP_TO_LEFTKNEE'] = get_line(joints['LEFTHIP'], joints['LEFTKNEE'])
    body_lines['RIGHTHIP_TO_RIGHTKNEE'] = get_line(joints['RIGHTHIP'], joints['RIGHTKNEE'])

    # Side knees to side ankles
    body_lines['LEFTKNEE_TO_LEFTANKLE'] = get_line(joints['LEFTKNEE'], joints['LEFTANKLE'])
    body_lines['RIGHTKNEE_TO_RIGHTANKLE'] = get_line(joints['RIGHTKNEE'], joints['RIGHTANKLE'])

    # Side shoulder to neck
    body_lines['LEFTSHOULDER_TO_NECK'] = get_line(joints['LEFTSHOULDER'], joints['MID_LEFTSHOULDER_RIGHTSHOULDER'])
    body_lines['RIGHTSHOULDER_TO_NECK'] = get_line(joints['RIGHTSHOULDER'], joints['MID_LEFTSHOULDER_RIGHTSHOULDER'])

    # Side shoulder to side elbow
    body_lines['LEFTSHOULDER_TO_LEFTELBOW'] = get_line(joints['LEFTSHOULDER'], joints['LEFTELBOW'])
    body_lines['RIGHTSHOULDER_TO_RIGHTELBOW'] = get_line(joints['RIGHTSHOULDER'], joints['RIGHTELBOW'])

    # Side shoulder to side elbow
    body_lines['LEFTELBOW_TO_LEFTWRIST'] = get_line(joints['LEFTELBOW'], joints['LEFTWRIST'])
    body_lines['RIGHTELBOW_TO_RIGHTWRIST'] = get_line(joints['RIGHTELBOW'], joints['RIGHTWRIST'])

    return body_lines


def get_body_angles(body_lines):
    """ Get body angles based on the body lines

    Args:
      path: body lines obtained from the drawing lines between different points

    Returns: A dictionary of body angles
    """
    body_angles = {}

    for i in range(0, len(list(body_lines))):
        for j in range(i + 1, len(list(body_lines))):
            numerator = np.dot(list(body_lines.values())[i], list(body_lines.values())[j])
            denominator = magnitude(list(body_lines.values())[i]) * magnitude(list(body_lines.values())[j])
            # print the body angle with respect to the the cosine value of the numerator and denominator
            body_angles[str(list(body_lines)[i]) + str('_WITH_') + str(list(body_lines)[j])] = math.acos(
                numerator / denominator)

    return body_angles


def calculate_features(filename):
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    output = []
    # Checking if the image is empty or not
    if image is None:
        result = "Image is empty!!"
        return
    image = image_resize(image)

    ## Setup mediapipe instance
    with mpPose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        results = pose.process(image)
        if not results.pose_landmarks:
            return
        try:
            landmarks = results.pose_landmarks.landmark
            output = []
            joints = get_joints(landmarks)
            #             output=output+joints
            output = output + get_body_angles(get_body_lines(joints))
            return output
        except Exception as e:
            print(e)


def calculate_features_frame(frame):
    img = image_resize(frame)

    ## Clarissa - Setup media instance - This was pose before, should it be holistic?
    with mpHolistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]
        results = holistic.process(imgRGB)

        # if no landmarks are detected, then just return
        if not results.pose_landmarks:
            return

        try:
            # current pose
            pose_landmarks = results.pose_landmarks.landmark
            # pose_landmarks = get_pose_landmarks().landmark
            output = {}
            joints = get_joints(pose_landmarks)
            output = get_body_angles(get_body_lines(joints))
            return output
        except Exception as e:
            print(e)