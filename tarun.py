# import cv2
# import dlib
# import numpy as np
# from keras.models import load_model
# from keras.preprocessing.image import img_to_array

# # Load the pre-trained models
# age_model = load_model("age_model.h5")
# emotion_model = load_model("emotion_model.h5")

# # Define the age and emotion labels
# age_labels = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
# emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# # Initialize the face detector and landmark predictor
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# # Capture the video stream using the default camera
# cap = cv2.VideoCapture(0)

# # Loop over the frames from the video stream
# while True:
#     # Read a frame from the video stream
#     ret, frame = cap.read()
#     if not ret:
#         break
        
#     # Convert the frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     # Detect the faces in the frame
#     faces = detector(gray, 0)
    
#     # Loop over the faces
#     for face in faces:
#         # Get the facial landmarks for the face region
#         landmarks = predictor(gray, face)
        
#         # Convert the landmarks to a numpy array
#         landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
        
#         # Extract the face region as a 64x64 grayscale image
#         x, y, w, h = face.left(), face.top(), face.width(), face.height()
#         face_roi = gray[y:y+h, x:x+w]
#         face_roi = cv2.resize(face_roi, (64, 64))
        
#         # Preprocess the face region for age and emotion detection
#         face_roi = face_roi.astype("float") / 255.0
#         face_roi = img_to_array(face_roi)
#         face_roi = np.expand_dims(face_roi, axis=0)
        
#         # Predict the age and emotion
#         age_preds = age_model.predict(face_roi)[0]
#         age_label = age_labels[np.argmax(age_preds)]
        
#         emotion_preds = emotion_model.predict(face_roi)[0]
#         emotion_label = emotion_labels[np.argmax(emotion_preds)]
        
#         # Draw the landmarks and labels on the frame
#         for i, (x, y) in enumerate(landmarks):
#             cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
        
#         label = "{} {}".format(age_label, emotion_label)
#         cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
#         # Draw a rectangle around the face
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
#     # Display the frame
#     cv2.imshow("Face Detection, Age Detection, and Emotion Detection", frame)
    
#     # Wait for a key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the resources and close the windows
# cap.release()
# cv2.destroyAllWindows()










# import numpy as np
# import cv2
# import tensorflow as tf

# # Load the cascade classifier for face detection
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# # Load the emotion detection model
# emotion_model = tf.keras.models.load_model('emotion_model.hdf5')

# # Define the emotion labels
# emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# # Define the function for emotion detection
# def detect_emotion(img):
#     # Preprocess the image for emotion detection
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray_img = cv2.resize(gray_img, (64, 64))
#     gray_img = gray_img.astype("float") / 255.0
#     gray_img = np.expand_dims(gray_img, 0)
#     gray_img = np.expand_dims(gray_img, -1)

#     # Predict the emotion
#     emotion = emotion_model.predict(gray_img)[0]
#     emotion = emotion_labels[np.argmax(emotion)]
#     print(emotion)

#     a = emotion

# # Load the video capture object
# cap = cv2.VideoCapture(0)

# while True:
#     # Read a frame from the video capture object
#     ret, frame = cap.read()

#     # Detect the faces in the frame
#     faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     # Draw a rectangle around the faces and detect emotions
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
#         roi_color = frame[y:y+h, x:x+w]

#         # Detect the emotion
#         emotion = detect_emotion(roi_color)

#         # Add the emotion label to the frame
#         cv2.putText(frame, emotion, (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#     # Show the frame
#     cv2.imshow('Face and Emotion Detection', frame)

#     # Exit the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture object and close all windows
# cap.release()
# cv2.destroyAllWindows()










# import numpy as np
# import cv2
# import tensorflow as tf

# # Load the cascade classifier for face detection
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# # Load the emotion detection model
# emotion_model = tf.keras.models.load_model('emotion_model.hdf5')

# # Load the age detection model
# age_net = cv2.dnn.readNetFromCaffe('age_deploy.prototxt', 'age_net.caffemodel')

# # Define the emotion labels
# emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# # Define the function for emotion detection
# def detect_emotion(img):
#     # Preprocess the image for emotion detection
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray_img = cv2.resize(gray_img, (64, 64))
#     gray_img = gray_img.astype("float") / 255.0
#     gray_img = np.expand_dims(gray_img, 0)
#     gray_img = np.expand_dims(gray_img, -1)

#     # Predict the emotion
#     emotion = emotion_model.predict(gray_img)[0]
#     emotion = emotion_labels[np.argmax(emotion)]
#     return emotion

# # Load the video capture object
# cap = cv2.VideoCapture(0)

# while True:
#     # Read a frame from the video capture object
#     ret, frame = cap.read()

#     # Detect the faces in the frame
#     faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     # Draw a rectangle around the faces and detect emotions and age
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
#         roi_color = frame[y:y+h, x:x+w]

#         # Detect the emotion
#         emotion = detect_emotion(roi_color)

#         # Add the emotion and age labels to the frame
#         blob = cv2.dnn.blobFromImage(roi_color, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
#         age_net.setInput(blob)
#         age_preds = age_net.forward()
#         age = age_preds[0][0] * 100
#         cv2.putText(frame, "Emotion: " + emotion, (x, y-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#         cv2.putText(frame, "Age: " + str(int(age)), (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#     # Show the frame
#     cv2.imshow('Face and Emotion Detection', frame)

#     # Exit the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture object and close all windows
# cap.release()
# cv2.destroyAllWindows()



import numpy as np
import cv2
import tensorflow as tf
import webbrowser


# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the emotion detection model
emotion_model = tf.keras.models.load_model('emotion_model.hdf5')

# Load the age detection model
age_net = cv2.dnn.readNetFromCaffe('age_deploy.prototxt', 'age_net.caffemodel')

# Define the emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Define the emotion to query mapping
# query_mapping = {
#     'happy': 'happy songs playlist for {} year person',
#     'sad': 'sad songs for {} year person',
#     'angry': 'angry songs for {} year person',
#     'surprise': 'surprise songs for {} year person',
#     'disgust': 'disgust songs for {} year person',
#     'fear': 'fear songs for {} year person',
#     'neutral': 'neutral songs for {} year person'
# }

# Define the age labels
ageList = ['(0-6)', '(6-12)', '(12-18)', '(18-30)', '(30-45)', '(45-60)', '(60-100)']

# Define the function for emotion detection
def detect_emotion(img):
    # Preprocess the image for emotion detection
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.resize(gray_img, (64, 64))
    gray_img = gray_img.astype("float32") / 255.0
    gray_img = np.expand_dims(gray_img, 0)
    gray_img = np.expand_dims(gray_img, -1)

    # Predict the emotion
    emotion = emotion_model.predict(gray_img)[0]
    emotion = emotion_labels[np.argmax(emotion)]
    return emotion

# Load the video capture object
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture object
    ret, frame = cap.read()

    # Detect the faces in the frame
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw a rectangle around the faces and detect emotions and age
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        roi_color = frame[y:y+h, x:x+w]

        # Detect the emotion
        emotion = detect_emotion(roi_color)

        # Add the emotion and age labels to the frame
        blob = cv2.dnn.blobFromImage(roi_color, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        age_net.setInput(blob)
        age_preds = age_net.forward()
        # age = age_preds[0][0]*100
        age = ageList[age_preds[0].argmax()]
        
        cv2.putText(frame, "Emotion: " + emotion, (x, y-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Age: " + age, (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Face and Emotion Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
    # Get the current age and emotion
        age_list = []
        emotion_list = []
        for (x, y, w, h) in faces:
            roi_color = frame[y:y+h, x:x+w]
            blob = cv2.dnn.blobFromImage(roi_color, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
            age_net.setInput(blob)
            age_preds = age_net.forward()
            # age = age_preds[0][0]*100
            age = ageList[age_preds[0].argmax()]
            age_range = age.split('-')
            min_age = int(age_range[0][1:])
            max_age = int(age_range[1][:-1])
            age = (min_age + max_age) // 2

            age_list.append(int(age))
            emotion = detect_emotion(roi_color)
            emotion_list.append(emotion)
        
        # Print and store the results
        print("Age:", age_list)
        print("Emotion:", emotion_list)
        if emotion_list[0] == 'happy':
            query = 'happy mood songs playlist for ' + str(age_list[0]) + ' year person'
        elif emotion_list[0] == 'sad':
            query = 'sad mood songs playlist for ' + str(age_list[0]) + ' year person'
        elif emotion_list[0] == 'angry':
            query = 'angry mood songs playlist for ' + str(age_list[0]) + ' year person'
        elif emotion_list[0] == 'surprise':
            query = 'surprise mood songs playlist for ' + str(age_list[0]) + ' year person'
        elif emotion_list[0] == 'disgust':
            query = 'disgust mood songs playlist for ' + str(age_list[0]) + ' year person'
        elif emotion_list[0] == 'fear':
            query = 'fear mood songs playlist for ' + str(age_list[0]) + ' year person'
        else:
            query = 'neutral mood songs playlist for ' + str(age_list[0]) + ' year person'
        break

    # if emotion in query_mapping:
    #     query = query_mapping[emotion].format(age)
    #     url = f"https://www.youtube.com/results?search_query={query}"
    #     webbrowser.open_new_tab(url)

webbrowser.open('https://www.youtube.com/results?search_query=' + query)

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()




# import numpy as np
# import cv2
# import tensorflow as tf
# import webbrowser

# # Load the cascade classifier for face detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Load the emotion detection model
# emotion_model = tf.keras.models.load_model('emotion_model.hdf5')

# # Load the age detection model
# age_net = cv2.dnn.readNetFromCaffe('age_deploy.prototxt', 'age_net.caffemodel')

# # Define the emotion labels
# emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# # Define the function for emotion detection
# def detect_emotion(img):
#     # Preprocess the image for emotion detection
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray_img = cv2.resize(gray_img, (64, 64))
#     gray_img = gray_img.astype("float") / 255.0
#     gray_img = np.expand_dims(gray_img, 0)
#     gray_img = np.expand_dims(gray_img, -1)

#     # Predict the emotion
#     emotion = emotion_model.predict(gray_img)[0]
#     emotion = emotion_labels[np.argmax(emotion)]
#     return emotion

# # Load the video capture object
# cap = cv2.VideoCapture(0)

# while True:
#     # Read a frame from the video capture object
#     ret, frame = cap.read()

#     # Detect the faces in the frame
#     faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     # Draw a rectangle around the faces and detect emotions and age
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
#         roi_color = frame[y:y+h, x:x+w]

#         # Detect the emotion
#         emotion = detect_emotion(roi_color)

#         # Add the emotion and age labels to the frame
#         blob = cv2.dnn.blobFromImage(roi_color, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
#         age_net.setInput(blob)
#         age_preds = age_net.forward()
#         age = age_preds[0][0] * 100
#         cv2.putText(frame, "Emotion: " + emotion, (x, y-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#         cv2.putText(frame, "Age: " + str(int(age)), (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#         # Search for a song on YouTube based on the emotion detected
#         if emotion == 'happy':
#             query = 'happy songs'
#         elif emotion == 'sad':
#             query = 'sad songs'
#         elif emotion == 'angry':
#             query = 'angry songs'
#         elif emotion == 'surprise':
#             query = 'surprise songs'
#         elif emotion == 'disgust':
#             query = 'disgust songs'
#         elif emotion == 'fear':
#             query = 'fear songs'
#         else:
#             query = 'neutral songs'
#     # Open the YouTube search results page in the default web browser
#     webbrowser.open('https://www.youtube.com/results?search_query=' + query)

#     # Display the resulting frame
#     cv2.imshow('frame', frame)

#     # Wait for the 'q' key to be pressed to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # # Release the video capture object and close all windows
# cap.release()
# cv2.destroyAllWindows()



