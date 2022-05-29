from cgi import test
import math
import cv2
import numpy as np
import time
import mediapipe as mp
import matplotlib.pyplot as plt
import os
import io
from google.cloud import automl
from google.cloud import videointelligence_v1p3beta1 as videointelligence
from google.cloud import storage
from pyngrok import ngrok
import multiprocessing
from re import M
import sched, time
from rsa import verify
import urllib3
import ssl
import requests
from flask import Flask



# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose

# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils 


def detectPose(image, pose, display=True):
    '''
    This function performs pose detection on an image.
    Args:
        image: The input image with a prominent person whose pose landmarks needs to be detected.
        pose: The pose setup function required to perform the pose detection.
        display: A boolean value that is if set to true the function displays the original input image, the resultant image, 
                 and the pose landmarks in 3D plot and returns nothing.
    Returns:
        output_image: The input image with the detected pose landmarks drawn.
        landmarks: A list of detected landmarks converted into their original scale.
    '''
    
    # Create a copy of the input image.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the Pose Detection.
    results = pose.process(imageRGB)
    
    # Retrieve the height and width of the input image.
    height, width, _ = image.shape
    
    # Initialize a list to store the detected landmarks.
    landmarks = []
    
    # Check if any landmarks are detected.
    if results.pose_landmarks:
    
        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        
        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:
            
            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))
    
    # Check if the original input image and the resultant image are specified to be displayed.
    if display:
    
        # Display the original input image and the resultant image.
        plt.figure(figsize=[22,22])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
        # Also Plot the Pose landmarks in 3D.
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        
    # Otherwise
    else:
        
        # Return the output image and the found landmarks.
        return output_image, landmarks



def calculateAngle(landmark1, landmark2, landmark3):
    '''
    This function calculates angle between three different landmarks.
    Args:
        landmark1: The first landmark containing the x,y and z coordinates.
        landmark2: The second landmark containing the x,y and z coordinates.
        landmark3: The third landmark containing the x,y and z coordinates.
    Returns:
        angle: The calculated angle between the three landmarks.

    '''

    # Get the required landmarks coordinates.
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    # Check if the angle is less than zero.
    if angle < 0:

        # Add 360 to the found angle.
        angle += 360
    
    # Return the calculated angle.
    return angle

def findX(landmark1,landmark2):
    # Get the required landmarks coordinates.
    x1 , _ , _ = landmark1
    x2 , _ , _ = landmark2
    
    locatey = x1 - x2
    
    return locatey


def classifyPose(landmarks, output_image, display=False):
    '''
    This function classifies yoga poses depending upon the angles of various body joints.
    Args:
        landmarks: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: A image of the person with the detected pose landmarks drawn.
        display: A boolean value that is if set to true the function displays the resultant image with the pose label 
        written on it and returns nothing.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.

    '''
    
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'

    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)
    
    # Calculate the required angles.
    #----------------------------------------------------------------------------------------------------------------
    
    # Get the angle between the left shoulder, elbow and wrist points. 
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
    # Get the angle between the right shoulder, elbow and wrist points. 
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])   
    
    # Get the angle between the left elbow, shoulder and hip points. 
    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    # Get the angle between the right hip, shoulder and elbow points. 
    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    # Get the angle between the left hip, knee and ankle points. 
    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    # Get the angle between the right hip, knee and ankle points 
    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    #get angle for the left hip
    left_hip_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])
    
    #get angle for the left hip
    right_hip_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])   


    rheeltonose = findX(landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value],
                        landmarks[mp_pose.PoseLandmark.NOSE.value])
    
    lheeltonose = findX(landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value],
                        landmarks[mp_pose.PoseLandmark.NOSE.value])
    
   
#Check if it is the falling.
    #----------------------------------------------------------------------------------------------------------------

# Check if shoulders are above 100 degrees.
    if left_shoulder_angle > 80 and left_shoulder_angle < 190 or right_shoulder_angle > 80 and right_shoulder_angle < 190:

        # Check if the other leg is bended at the required angle.
        #or just angle <90 if just want flailing of arms
        if left_knee_angle < 120 or right_knee_angle < 120:
            
            # Check if one of the elbow is bended.
                 if left_elbow_angle > 90 or right_elbow_angle > 90 :
            
                    # Specify the label of the pose that is Warrior II pose.
                    label = 'Falling'
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if the pose is classified successfully
    if label != 'Unknown Pose':

        # Update the color (to green) with which the label will be written on the image.
        color = (0, 255, 0)  
    
    # Write the label on the output image. 
    cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    # Check if the resultant image is specified to be displayed.
    if display:
    
        # Display the resultant image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
    else:
        # Return the output image and the classified label.
        return output_image, label


#detect lying
def googleAIPrediction2(video_number):


                os.chdir("D:/Desktop/ffmpeg-5.0.1-essentials_build/ffmpeg-5.0.1-essentials_build/bin/")
                os.system("ffmpeg -i video" + str(video_number) + ".avi -strict -2 video" + str(video_number) + ".mp4")
                time.sleep(1)
                os.system("cmd /c ffmpeg -i video" + str(video_number) + ".mp4 -c copy -map 0 -movflags faststart output" + str(video_number) + ".mp4")
                time.sleep(1)
                os.chdir("D:/Desktop/tensorflow python")
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "GoogleCloud_Key.json"
                time.sleep(1)
                path = "D:/Desktop/ffmpeg-5.0.1-essentials_build/ffmpeg-5.0.1-essentials_build/bin/output" + str(video_number) + ".mp4"
                # project_id = 'project_id'
                # model_id = 'automl_action_recognition_model_id'

                client = videointelligence.StreamingVideoIntelligenceServiceClient()

                #get lying down model
                model_path = automl.AutoMlClient.model_path(
                        "becarefall01", "us-central1", "8074870568991588352"
                )
                print("Getting Predictions for lying down for video" + str(video_number) + ".mp4" )
                automl_config = videointelligence.StreamingAutomlActionRecognitionConfig(
                    model_name=model_path
                )
                video_config = videointelligence.StreamingVideoConfig(
                    feature=videointelligence.StreamingFeature.STREAMING_AUTOML_ACTION_RECOGNITION,
                    automl_action_recognition_config=automl_config,
                )
                # config_request should be the first in the stream of requests.
                config_request = videointelligence.StreamingAnnotateVideoRequest(
                    video_config=video_config
                )
                # Set the chunk size to 5MB (recommended less than 10MB).
                chunk_size = 5 * 1024 * 1024

                def stream_generator():
                    yield config_request
                    # Load file content.
                    # Note: Input videos must have supported video codecs. See
                    # https://cloud.google.com/video-intelligence/docs/streaming/streaming#supported_video_codecs
                    # for more details.
                    with io.open(path, "rb") as video_file:
                        while True:
                            data = video_file.read(chunk_size)
                            if not data:
                                break
                            yield videointelligence.StreamingAnnotateVideoRequest(
                                input_content=data
                            )

                request = stream_generator()

                # streaming_annotate_video returns a generator.
                # The default timeout is about 300 seconds.
                # To process longer videos it should be set to
                # larger than the length (in seconds) of the video.
                responses = client.streaming_annotate_video(request, timeout=900)

                    #initialize variable to store highest confidence of video
                highest_confidence = 0.0
                # Each response corresponds to about 1 second of video.
                for response in responses:
                    # Check for errors.
                    if response.error.message:
                        print(response.error.message)
                        break

                for label in response.annotation_results.label_annotations:
                    for frame in label.frames:
                        confidence = float(frame.confidence)
                        print(confidence)
                        #check if confidence is higher and set as highest confidence
                        if confidence > highest_confidence:
                            highest_confidence = confidence

                            # print(
                            #     "At {:3d}s segment, {:5.1%} {}".format(
                            #         frame.time_offset.seconds,
                            #         frame.confidence,
                            #         label.entity.entity_id,
                            #     )
                            # )
                        print("Highest confidence from video: " + str("{:5.1%}").format(highest_confidence))

                        highest_confidence = highest_confidence * 100

                        if highest_confidence >= 80:
                            print("3 Minutes is up we have contacted the ambulance")
                            requests.get("https://451c-202-166-40-140.ap.ngrok.io/spam_sms",verify=False)



def uploadBlobAndPublic(bucket_name, source_file_name, destination_blob_name):

    #upload file to google cloud storage
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )
    time.sleep(1)

    #make video public in google cloud storage
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.make_public()
    print(
        f"Blob {blob.name} is publicly accessible at {blob.public_url}"
    )

    return blob.public_url


#detect fall
def googleAIPrediction(test_video_no,current_video_no):

    #infinite loop
    infinite = 0
    while infinite < 10:

        video_no = int(test_video_no.value)
        time.sleep(5)
        
        #if test_video_no (global variable is not 0)
        if video_no != 0:

            #go to ffmpeg directory
            os.chdir("D:/Desktop/ffmpeg-5.0.1-essentials_build/ffmpeg-5.0.1-essentials_build/bin/")
            #convert .avi file to .mp4 file
            os.system("ffmpeg -i video" + str(video_no) + ".avi -strict -2 video" + str(video_no) + ".mp4")
            time.sleep(1)
            #move to MOOV Atom to the beginning of mp4 file to be able to use for testing
            os.system("cmd /c ffmpeg -i video" + str(video_no) + ".mp4 -c copy -map 0 -movflags faststart output" + str(video_no) + ".mp4")
            os.chdir("D:/Desktop/tensorflow python")
            #access to GCP
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "GoogleCloud_Key.json"
            time.sleep(1)
            #set local path of video to test
            path = "D:/Desktop/ffmpeg-5.0.1-essentials_build/ffmpeg-5.0.1-essentials_build/bin/output" + str(video_no) + ".mp4"

            # project_id = 'project_id'
            # model_id = 'automl_action_recognition_model_id'

            client = videointelligence.StreamingVideoIntelligenceServiceClient()
            #get falling down model
            model_path = automl.AutoMlClient.model_path(
                    "becarefall01", "us-central1", "6796974174725210112"
            )
            print("Getting Predictions for falling down for video" + str(video_no) + ".mp4" )
            automl_config = videointelligence.StreamingAutomlActionRecognitionConfig(
                model_name=model_path
            )
            video_config = videointelligence.StreamingVideoConfig(
                feature=videointelligence.StreamingFeature.STREAMING_AUTOML_ACTION_RECOGNITION,
                automl_action_recognition_config=automl_config,
            )
            # config_request should be the first in the stream of requests.
            config_request = videointelligence.StreamingAnnotateVideoRequest(
                video_config=video_config
            )
            # Set the chunk size to 5MB (recommended less than 10MB).
            chunk_size = 5 * 1024 * 1024

            def stream_generator():
                yield config_request
                # Load file content.
                # Note: Input videos must have supported video codecs. See
                # https://cloud.google.com/video-intelligence/docs/streaming/streaming#supported_video_codecs
                # for more details.
                with io.open(path, "rb") as video_file:
                    while True:
                        data = video_file.read(chunk_size)
                        if not data:
                            break
                        yield videointelligence.StreamingAnnotateVideoRequest(
                            input_content=data
                        )

            request = stream_generator()

            # streaming_annotate_video returns a generator.
            # The default timeout is about 300 seconds.
            # To process longer videos it should be set to
            # larger than the length (in seconds) of the video.
            responses = client.streaming_annotate_video(request, timeout=900)

            #initialize variable to store highest confidence of video
            highest_confidence = 0.0
            # Each response corresponds to about 1 second of video.
            for response in responses:
                # Check for errors.
                if response.error.message:
                    print(response.error.message)
                    break

                for label in response.annotation_results.label_annotations:
                    for frame in label.frames:
                        confidence = float(frame.confidence)
                        print(confidence)
                        #check if confidence is higher and set as highest confidence
                        if confidence > highest_confidence:
                            highest_confidence = confidence

                        # print(
                        #     "At {:3d}s segment, {:5.1%} {}".format(
                        #         frame.time_offset.seconds,
                        #         frame.confidence,
                        #         label.entity.entity_id,
                        #     )
                        # )

            print("Highest confidence from video: " + str("{:5.1%}").format(highest_confidence))

            highest_confidence = highest_confidence * 100

            if highest_confidence >= 80:
                
                # result_url = uploadBlobAndPublic("android_python_api_bucket","D:/Desktop/ffmpeg-5.0.1-essentials_build/ffmpeg-5.0.1-essentials_build/bin/" + "video" + str(video_no) + ".mp4","video" + str(video_no) + ".mp4")
                # print(result_url)
                # result_url = result_url.replace("/","%2B")
                # print(result_url)
                #send request to API
                requests.get("https://451c-202-166-40-140.ap.ngrok.io/spam",verify=False)
                # requests.get("https://96de-115-66-189-103.ngrok.io/geturl/" + result_url,verify=False)
                
                print("Timer has started. 3 minute before the system decides to inform the ambulance")
                minutes = 3
                minutes = int(minutes)
                start_time = time.time()
                s = sched.scheduler(time.time, time.sleep)

                def do_something(): 
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    # print("current time (in seconds) is: " + str(elapsed_time))
                    print("Test lying video no: " + str(current_video_no.value))
                    googleAIPrediction2(int(current_video_no.value))

                # minute_counter = 1
                # while minute_counter <= minutes:
                #     timestamp = minute_counter * 30 
                #     s.enter(timestamp, 1, do_something)
                #     print(minute_counter)
                #     minute_counter = minute_counter + 1
                timestamp = minutes * 60
                s.enter(timestamp,1,do_something)
                s.run()

            test_video_no.value = 0

        else:
            None

#Flask Server
app = Flask(__name__)

result = "nil"
confirm_sms = "nil"
final_url = "nil"

@app.route("/")
def hello():
  return "Hello World"

@app.route("/spam")
def spam():
  global result
  result = "fall"
  return result

@app.route("/result")
def result():
  global result
  final_result = result
  result = "nil"
  return final_result

@app.route("/spam_sms")
def spam_sms():
  global confirm_sms
  confirm_sms = "confirm"
  return confirm_sms

@app.route("/sms_result")
def sms_result():
  global confirm_sms
  final_sms = confirm_sms
  confirm_sms = "nil"
  return final_sms

@app.route("/geturl/<url>")
def get_url(url):
    global final_url
    url = url.replace("+","\/")
    final_url = url
    print(final_url)
    return final_url 

@app.route("/sendurl")
def send_url():
    global final_url
    return final_url


def startFlaskServer():
    app.run(host='0.0.0.0')


if __name__ == '__main__':
    #a integer value shared among main (OpeCV video recording in __main__) and child process (googleAIPrediction)
    test_video_no  = multiprocessing.Value("d",1)
    print(test_video_no.value)

    current_video_no = multiprocessing.Value("d",1)

    #starting up child process (googleAIPrediction)
    p1 = multiprocessing.Process(target=googleAIPrediction,args=(test_video_no,current_video_no))
    p1.start()

    #starting up child process (flask server for API)
    p2 = multiprocessing.Process(target=startFlaskServer)
    p2.start()
    
    # Setup Pose function for video.
    pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

    # Initialize the VideoCapture object to read from the webcam.
    cap = cv2.VideoCapture(0)

    #set timer start and end duration for each recording
    capture_duration = 10
    start_time = time.time()

    #video writer settings
    width = int(cap.get(3))
    height = int(cap.get(4))
    fcc = cv2.VideoWriter_fourcc(*'XVID')

    #initial settings before video is recording
    recording = False
    videono = 0
    pose = None
    
    # Iterate until the webcam is accessed successfully.
    while(1):
        # Read a frame.
        ok, frame = cap.read()

        #set time on top left of frame
        hms = time.strftime('%H:%M:%S', time.localtime())
        cv2.putText(frame, str(hms), (0, 15), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
        
        # # Get the width and height of the frame
        # frame_height, frame_width, _ =  frame.shape
        
        # # Resize the frame while keeping the aspect ratio.
        # frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
        
        # Perform Pose landmark detection.
        frame, landmarks = detectPose(frame, pose_video, display=False)
        
        # Check if the landmarks are detected.
        if landmarks:
            
            # Perform the Pose Classification.
            frame, pose = classifyPose(landmarks, frame, display=False)
        
        # Display the frame.
        cv2.imshow('Pose Classification', frame)

        
        #to detect the video number of mp4 file to send to google cloud for prediction
        if pose == "Falling":
            test_video_no.value = videono
            print("Fall detected at video no: " + str(test_video_no.value))


        #start recording when recording is false and time is less than capture duration
        if ( int(time.time() - start_time) < capture_duration ) and recording is False:
                videono += 1
                path = 'D:/Desktop/ffmpeg-5.0.1-essentials_build/ffmpeg-5.0.1-essentials_build/bin/video' + str(videono) + '.avi'
                print(path+' recording')
                writer = cv2.VideoWriter(path, fcc, 30.0, (width, height))
                recording = True

        #write frame if recording
        if recording:
            writer.write(frame)

        #stop recording when time is up and start a new recording
        if not ( int(time.time() - start_time) < capture_duration ):
            print('recording finished')
            recording = False
            writer.release()
            current_video_no.value = videono
            print("Most recent video no: (that is recorded finished) " + str(current_video_no.value))
            start_time = time.time()
        
        # Wait until a key is pressed.
        # Retreive the ASCII code of the key pressed
        k = cv2.waitKey(1) & 0xFF
        
        # Check if 'ESC' is pressed.
        if(k == 27):
            
            # Break the loop.
            break

    # Release the VideoCapture object and close the windows.
    cap.release()
    cv2.destroyAllWindows()