# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

## Import packages
# Standard library imports
from datetime import datetime
import enum
import math
import pickle
from typing import Dict, List, NamedTuple, Tuple
import os

# Third-party library imports
# Import cv2 to activate video / camera input
import cv2

# Import libraries for data analysis
import numpy as np
import pandas as pd

# Import matplotlib libraries
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

# Import Mediapipe
import mediapipe as mp

# Some modules to display an animation using imageio.
import imageio
from IPython.display import HTML, display

# Import tensorflow modules
import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import load_model
import tensorflow as tf
# import tensorflow_hub as hub
# from tensorflow_docs.vis import embed
# import keras
# from sklearn.model_selection import train_test_split

# Local library imports
import Person
from Movenet import Movenet
from Helper import load_pose_landmarks, landmarks_to_embedding, image_resize, calculate_features_frame, get_joints, get_body_angles, get_body_lines
from Visualization import get_keypoint_landmarks, draw_prediction_on_image, img_to_tensor, colors, EDGES, draw_keypoints, draw_connection
#####################################################################################
# Declare some initial variables
# Declare some mediapipe variables
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
points = mpPose.PoseLandmark

# Extract the yoga pose classifier from the previous Colab notebook
classifier_filename = 'models/Movenet_model_pretrained_Yoga82_model.sav'
classifier = pickle.load(open(classifier_filename, 'rb'))

# Extract the movenet model
# movenet_model_path = 'models/movenet_thunder'
no2_movenet_model_path = 'models/lite-model_movenet_singlepose_lightning_tflite_int8_4'
movenet = Movenet(no2_movenet_model_path)

# define directory for train data
csvs_out_train_path = 'train_test/train_complete_data.csv'

# define path directories
target_results = 'results/'

#####################################################################################
def detect(input_tensor, inference_count=3):
    #  Detect pose using the full input image
    movenet.detect(input_tensor.numpy(), reset_crop_region=True)
    # Repeatedly using previous detection result to identify the region of
    # interest and only croping that region to improve detection accuracy
    for _ in range(inference_count - 1):
        person = movenet.detect(input_tensor.numpy(), reset_crop_region=False)

        return person

def give_feedback(confidence, prev_confidence, student_body_angles, ideal_features):
    if (confidence >= 0.75 and confidence > prev_confidence):

        max_confidence = confidence
        # similar=kNN.kneighbors(X=np.array(test).reshape(1, -1), return_distance=True)[1][0][0]
        # ideal_features = idealFeatures_df.iloc['asana']
        curr_features = student_body_angles

        # find the differences in the body angles
        matrix = []
        matrix.append(ideal_features['HEAD_TO_NECK_WITH_NECK_TO_HIP'][1] - curr_features['HEAD_TO_NECK_WITH_NECK_TO_HIP'])
        matrix.append(ideal_features['LEFTSHOULDER_TO_NECK_WITH_LEFTSHOULDER_TO_LEFTELBOW'][1] - curr_features[
            'LEFTSHOULDER_TO_NECK_WITH_LEFTSHOULDER_TO_LEFTELBOW'])
        matrix.append(ideal_features['RIGHTSHOULDER_TO_NECK_WITH_RIGHTSHOULDER_TO_RIGHTELBOW'][1] - curr_features[
            'RIGHTSHOULDER_TO_NECK_WITH_RIGHTSHOULDER_TO_RIGHTELBOW'])
        matrix.append(ideal_features['LEFTSHOULDER_TO_NECK_WITH_LEFTELBOW_TO_LEFTWRIST'][1] - curr_features[
            'LEFTSHOULDER_TO_NECK_WITH_LEFTELBOW_TO_LEFTWRIST'])
        matrix.append(ideal_features['RIGHTSHOULDER_TO_NECK_WITH_RIGHTELBOW_TO_RIGHTWRIST'][1] - curr_features[
            'RIGHTSHOULDER_TO_NECK_WITH_RIGHTELBOW_TO_RIGHTWRIST'])
        matrix.append(ideal_features['NECK_TO_HIP_WITH_LEFTHIP_TO_MIDHIP'][1] - curr_features[
            'NECK_TO_HIP_WITH_LEFTHIP_TO_MIDHIP'])
        matrix.append(ideal_features['NECK_TO_HIP_WITH_RIGHTHIP_TO_MIDHIP'][1] - curr_features[
            'NECK_TO_HIP_WITH_RIGHTHIP_TO_MIDHIP'])
        matrix.append(ideal_features['LEFTSHOULDER_TO_NECK_WITH_LEFTELBOW_TO_LEFTWRIST'][1] - curr_features[
            'LEFTSHOULDER_TO_NECK_WITH_LEFTELBOW_TO_LEFTWRIST'])
        matrix.append(ideal_features['RIGHTSHOULDER_TO_NECK_WITH_RIGHTELBOW_TO_RIGHTWRIST'][1] - curr_features[
            'RIGHTSHOULDER_TO_NECK_WITH_RIGHTELBOW_TO_RIGHTWRIST'])
        matrix.append(ideal_features['LEFTHIP_TO_LEFTKNEE_WITH_LEFTKNEE_TO_LEFTANKLE'][1] - curr_features[
            'LEFTHIP_TO_LEFTKNEE_WITH_LEFTKNEE_TO_LEFTANKLE'])
        matrix.append(ideal_features['RIGHTHIP_TO_RIGHTKNEE_WITH_RIGHTKNEE_TO_RIGHTANKLE'][1] - curr_features[
            'RIGHTHIP_TO_RIGHTKNEE_WITH_RIGHTKNEE_TO_RIGHTANKLE'])
        #                     matrix=matrix/math.pi
        advice = []
        for diff in matrix:
            diff = diff * 180 / math.pi
            #                         advice.append(str(diff))
            if (abs(diff) <= 13.9):
                advice.append("Correct")
            elif (diff < 0):
                advice.append('Open more')
            else:
                advice.append('Bend more')
        print(advice)
    #                         else:
    #                             advice.append(str(diff))


def test_with_images():
    # Load the image
    image_path = 'data/images/DanielWrong.jpg'
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image)

    image_height, image_width, channel = image.shape

    # detect the image
    person = detect(image)

    # get the key point landmarks
    pose_landmarks = get_keypoint_landmarks(person)

    # pose_landmarks bjg u
    lm_pose = landmarks_to_embedding(tf.reshape(tf.convert_to_tensor(pose_landmarks), (1, 51)))
    # print(lm_pose.output_shape)
    predict = classifier.predict(lm_pose)

    '''Draw on image'''
    font = cv2.FONT_HERSHEY_DUPLEX
    org = (10, 40)
    fontScale = 1
    color = (19, 255, 30)
    thickness = 1

    image = np.array(image)

    # accuracy
    acc = round(np.max(predict[0], axis=0) * 100, 2)

    cv2.putText(image, class_names[np.argmax(predict)] + " | " + str(acc) + "%", org, font,
                fontScale, color, thickness, cv2.LINE_AA)

    image = draw_prediction_on_image(image, person, crop_region=None, close_figure=False, keep_input_size=True)

    curr_datetime = datetime.now().strftime('%Hh%Mm%Ss %d_%m_%Y ')
    r = str(acc) + "% " + curr_datetime
    image_pred_path = target_results + '/draw_skeleton %s.png' % r
    image_result_path = target_results + '/result %s.png' % r
    # print(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_pred_path, image)

    # Set up some initial variables
    confidence = 0
    max_confidence = 0
    prev_confidence = 0

    # Extract landmarks
    pose_landmarks = get_keypoint_landmarks(person)
    # body_angles_landmarks = person.pose_landmarks.landmark
    lm_pose = landmarks_to_embedding(tf.reshape(tf.convert_to_tensor(pose_landmarks), (1, 51)))
    ### For getting the student_body_angles
    student_body_angles = calculate_features_frame(image)
    student_body_angles_list = list(student_body_angles.values())
    print("this is the student body angle")
    print(student_body_angles)

    # Initialize the file of target poses
    ideal_features = pd.read_csv('data/target_pose/datasetFinalAngles.csv')
    ###
    give_feedback(confidence, prev_confidence, student_body_angles, ideal_features)

    '''--------------------------------------- SHOW IMAGE -------------------------------------------'''
    # Read First Image
    img1 = cv2.imread(image_path)
    # Read Second Image
    img2 = cv2.imread(image_pred_path)
    # concatenate image Horizontally
    # Hori = np.concatenate((img1, img2), axis=1)
    cv2.imwrite(image_result_path, image)  # this is for in non-collab setup

    # # concatenate image Vertically
    # Verti = np.concatenate((img1, img2), axis=0)
    print("Output shape:", predict.shape)
    print("This picture is:", class_names[np.argmax(predict[0])])
    print("Accuracy:", np.max(predict[0], axis=0))
    # print(np.argmax(predict))
    # print(np.array(predict[0]))
    cv2.imshow('CLASSIFICATION OF YOGA POSE', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Load the train data
    X, y, class_names, _ = load_pose_landmarks(csvs_out_train_path)

    # Initialize the file of target poses
    ideal_features = pd.read_csv('data/target_pose/datasetFinalAngles2023-07-31.csv')

    # Start the live recording or pre-recorded video of the student avatar
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("data/videos/LennardDiagnosticUttanasana_Flipped.mp4")

    # If you have trouble opening it then give an error
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    # Set up some initial variables
    confidence = 0
    max_confidence = 0
    prev_confidence = 0
    results2 = 0

    # Set target_pose_image as none
    target_pose_image = None

    # Set some variables for the visualization
    font = cv2.FONT_HERSHEY_DUPLEX
    org = (10, 40)
    fontScale = 1
    color = (19, 255, 30)
    thickness = 1

    # Setup mediapipe instance
    with mpPose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # while its open
        while cap.isOpened():
            ret, frame = cap.read()

            # ret is a boolean variable that returns true if the frame is available, so if ret is true
            if ret:
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = image_resize(image)
                image.flags.writeable = False
                img1 = img_to_tensor(image)

                # Make detection
                person = detect(img1)

                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                try:

                    # Extract landmarks
                    pose_landmarks = get_keypoint_landmarks(person)
                    # body_angles_landmarks = person.pose_landmarks.landmark
                    lm_pose = landmarks_to_embedding(tf.reshape(tf.convert_to_tensor(pose_landmarks), (1, 51)))
                    ### For getting the student_body_angles
                    student_body_angles = calculate_features_frame(image)
                    student_body_angles_list = list(student_body_angles.values())
                    print("this is the student body angle")
                    print(student_body_angles)
                    ###

                    # predict the asana of what the person is doing
                    student_asana = classifier.predict(lm_pose)

                    #print("this is it")
                    #print(student_asana)

                    # print the confidence of how confident you are about the identified asana of the student

                    # confidence = sorted(classifier.predict_proba(np.array(student_body_angles).reshape(1, -1))[0])[-1]
                    confidence = sorted(classifier.predict(np.array(student_body_angles_list).reshape(1, -1))[0])[-1]
                    print("this is the confidence")
                    print(confidence)

                    # put the text of the asana on the image
                    # cv2.putText((img1, 'Asana = ' + str(student_asana[0]), 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                    #             2, cv2.LINE_AA)

                    # Asana should match the one of the similar image
                    #cv2.putText(img1, 'Confidence = ' + str(((confidence * 1000000) // 100) / 100) + '%', (20, 80),
                    #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                    # accuracy
                    # acc = round(np.max(student_asana[0], axis=0) * 100, 2)

                    # cv2.putText(image, class_names[np.argmax(student_asana)] + " | " + str(acc) + "%", org, font,
                                # fontScale, color, thickness, cv2.LINE_AA)

                    # image = draw_prediction_on_image(image, person, crop_region=None, close_figure=False,
                                                     # keep_input_size=True)

                    print(class_names[np.argmax(student_asana)])

                    # if confidence is greater than 0.75 and the confidence is greater than the previous one
                    if (confidence >= 0.75 and confidence > prev_confidence):

                        max_confidence = confidence
                        # similar=kNN.kneighbors(X=np.array(test).reshape(1, -1), return_distance=True)[1][0][0]
                        # ideal_features = idealFeatures_df.iloc['asana']
                        curr_features = student_body_angles

                        # find the differences in the body angles
                        matrix = []
                        matrix.append(ideal_features['HEAD_TO_NECK_WITH_NECK_TO_HIP'][1] - curr_features['HEAD_TO_NECK_WITH_NECK_TO_HIP'])
                        matrix.append(ideal_features['LEFTSHOULDER_TO_NECK_WITH_LEFTSHOULDER_TO_LEFTELBOW'][1] - curr_features['LEFTSHOULDER_TO_NECK_WITH_LEFTSHOULDER_TO_LEFTELBOW'])
                        matrix.append(ideal_features['RIGHTSHOULDER_TO_NECK_WITH_RIGHTSHOULDER_TO_RIGHTELBOW'][1] - curr_features['RIGHTSHOULDER_TO_NECK_WITH_RIGHTSHOULDER_TO_RIGHTELBOW'])
                        matrix.append(ideal_features['LEFTSHOULDER_TO_NECK_WITH_LEFTELBOW_TO_LEFTWRIST'][1] - curr_features['LEFTSHOULDER_TO_NECK_WITH_LEFTELBOW_TO_LEFTWRIST'])
                        matrix.append(ideal_features['RIGHTSHOULDER_TO_NECK_WITH_RIGHTELBOW_TO_RIGHTWRIST'][1] - curr_features['RIGHTSHOULDER_TO_NECK_WITH_RIGHTELBOW_TO_RIGHTWRIST'])
                        matrix.append(ideal_features['NECK_TO_HIP_WITH_LEFTHIP_TO_MIDHIP'][1] - curr_features['NECK_TO_HIP_WITH_LEFTHIP_TO_MIDHIP'])
                        matrix.append(ideal_features['NECK_TO_HIP_WITH_RIGHTHIP_TO_MIDHIP'][1] - curr_features['NECK_TO_HIP_WITH_RIGHTHIP_TO_MIDHIP'])
                        matrix.append(ideal_features['LEFTSHOULDER_TO_NECK_WITH_LEFTELBOW_TO_LEFTWRIST'][1] - curr_features['LEFTSHOULDER_TO_NECK_WITH_LEFTELBOW_TO_LEFTWRIST'])
                        matrix.append(ideal_features['RIGHTSHOULDER_TO_NECK_WITH_RIGHTELBOW_TO_RIGHTWRIST'][1] - curr_features['RIGHTSHOULDER_TO_NECK_WITH_RIGHTELBOW_TO_RIGHTWRIST'])
                        matrix.append(ideal_features['LEFTHIP_TO_LEFTKNEE_WITH_LEFTKNEE_TO_LEFTANKLE'][1] - curr_features['LEFTHIP_TO_LEFTKNEE_WITH_LEFTKNEE_TO_LEFTANKLE'])
                        matrix.append(ideal_features['RIGHTHIP_TO_RIGHTKNEE_WITH_RIGHTKNEE_TO_RIGHTANKLE'][1] - curr_features['RIGHTHIP_TO_RIGHTKNEE_WITH_RIGHTKNEE_TO_RIGHTANKLE'])
                        #                     matrix=matrix/math.pi
                        advice = []
                        for diff in matrix:
                            diff = diff * 180 / math.pi
                            #                         advice.append(str(diff))
                            if (abs(diff) <= 13.9):
                                advice.append("Correct")
                            elif (diff < 0):
                                advice.append('Open more')
                            else:
                                advice.append('Bend more')
                        print(advice)
                    #                         else:
                    #                             advice.append(str(diff))

                    #                     print(features)

                    # asana=asana_names[similar]
                    # file_name=file_names[similar]
                    # similar_image=cv2.imread(os.path.join('Images', asana, file_name), cv2.IMREAD_COLOR)
                    # similar_image=image_resize_small(similar_image)

                    if confidence >= 0.75 and len(np.unique(advice)) == 1 and advice[0] == 'Correct':
                        cv2.putText(image, 'Correct pose',
                                    (470, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA
                                    )
                    elif max_confidence > 0:
                        #                     cv2.putText(image, 'Neck = '+advice[0],
                        #                            (510, 50),
                        #                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 1, cv2.LINE_AA
                        #                                 )
                        cv2.putText(image, 'left_shoulder= ' + advice[1],
                                    (800, 190),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA
                                    )
                        cv2.putText(image, 'right_shoulder = ' + advice[2],
                                    (100, 190),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA
                                    )
                        cv2.putText(image, 'left_elbow = ' + advice[3],
                                    (820, 300),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA
                                    )
                        cv2.putText(image, 'right_elbow = ' + advice[4],
                                    (90, 300),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA
                                    )
                        cv2.putText(image, 'left_hip = ' + advice[5],
                                    (800, 400),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA
                                    )
                        cv2.putText(image, 'right_hip = ' + advice[6],
                                    (100, 400),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA
                                    )
                        cv2.putText(image, 'left_knee = ' + advice[7],
                                    (800, 500),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA
                                    )
                        cv2.putText(image, 'right_knee = ' + advice[8],
                                    (100, 500),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA
                                    )
                        prev_confidence = confidence
                        #### Feedback givign part is done
                except Exception as e:
                    pass

                # Render detections
                # mpDraw.draw_landmarks(image, pose.pose_landmarks, mpPose.POSE_CONNECTIONS)
                image = draw_prediction_on_image(image, person, crop_region=None, close_figure=False,keep_input_size=True)
                cv2.imshow('CLASSIFICATION OF YOGA POSE', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            else:
                break


                # if student_asana != teacher_asanas:  # skip the if statement and go to the next one and might be easier to put
                #   continue

                # accuracy
                acc = round(np.max(student_asana[0], axis=0) * 100, 2)

                cv2.putText(image, class_names[np.argmax(student_asana[0])] + " | " + str(acc) + "%", org, font,
                            fontScale, color, thickness, cv2.LINE_AA)

                curr_datetime = datetime.now().strftime('%Hh%Mm%Ss %d_%m_%Y ')
                r = str(acc) + "% " + curr_datetime
                image_pred_path = target_results + '/draw_skeleton %s.png' % r
                image_result_path = target_results + '/result %s.png' % r
                image = draw_prediction_on_image(image, person, crop_region=None, close_figure=False,
                                                 keep_input_size=True)

                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(image_pred_path, image)
                cv2.imwrite(image_result_path, image)  # this is for in non-collab setup



                '''--------------------------------------- SHOW IMAGE -------------------------------------------'''
                # Read First Image
                # img1 = cv2.imread(image_path)
                # Read Second Image
                # img2 = cv2.imread(image_pred_path)
                # concatenate image Horizontally
                # Hori = np.concatenate((img1, img2), axis=1)

                # # concatenate image Vertically
                # Verti = np.concatenate((img1, img2), axis=0)
                print("Output shape:", student_asana.shape)
                print("This picture is:", class_names[np.argmax(student_asana[0])])
                print("Accuracy:", np.max(student_asana[0], axis=0))
                # print(np.argmax(predict))
                # print(np.array(predict[0]))


        cap.release()
        cv2.destroyAllWindows()

    # if the whole body is not detected after 2 seconds, then give a message
    # make sure all limbs are detected and green

    # if the whole body is detected do the following
    #           print(test)

    # for teacher_asana in predefined_asanas ---  until the end of the length of the list of asanas
    # if target_pose has not been identified, show the non.jpg
    # if target_pose_image is None:
    #    target_pose_image=cv2.imread(r'none.jpg', cv2.IMREAD_COLOR)
    #    target_pose_image=image_resize_small(similar_image)

# asana should match the one of the teacher avatar as defined by list in predifined_asana
# while the student_asana and teacher_asana doesn't match, then keep looking
# want to have target pose in unity and predict user's pose in python and send user's pose to unity that then compares it to the teacher pose in unity
# keep looking
#    target_pose_image=cv2.imread(r'none.jpg', cv2.IMREAD_COLOR)
#    target_pose_image=image_resize_small(similar_image)
# keep printing some pose instructions
# if some time has now passed, then have a message asking if user is still interested in practicing

# print "amazing, you are now in the pose"