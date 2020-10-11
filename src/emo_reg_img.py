import cv2
from keras.models import load_model
import numpy

# load models path
face_models_path = '../trained_model/face_detection_models/haarcascade_frontalface_default.xml'
emotions_models_path = '../trained_model/emotion_models/emotion_recod_5_acc-0.351485.model'

# set emotion labels
emotion_labels = ['angry', 'fear', 'happy', 'sad', 'surprise', 'neutral']
# set face classifier
face_detection = cv2.CascadeClassifier(face_models_path)
# set emotion classifier
emotion_model = load_model(emotions_models_path)
# get network input shape
emotion_model_input_size = emotion_model.input_shape[1:3]
# read sample image
frame = cv2.imread("5.jpeg")
# frame = cv2.resize(frame,(480,640))
# convert to gray
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# detect faces
faces = face_detection.detectMultiScale(gray, 1.3, 5)
for x, y, w, h in faces:
    gray_face = gray[y:y+h, x:x+w]
    # apply same preprocessing
    gray_face = cv2.resize(gray_face, emotion_model_input_size)
    preprocessed_img = gray_face.astype('float32')
    preprocessed_img /= 255
    expanded_dimen_img = numpy.expand_dims(preprocessed_img, 0)
    expanded_dimen_img = numpy.expand_dims(expanded_dimen_img, -1)
    # predict the classes
    emotion_probabilities = emotion_model.predict(expanded_dimen_img)
    # select MAX predicted CLASS value
    emotion_max_prob = numpy.max(emotion_probabilities)
    emotion_label = numpy.argmax(emotion_probabilities)
    # make bounding box
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
    cv2.putText(frame, emotion_labels[emotion_label],
                (x, y), cv2.FONT_HERSHEY_COMPLEX, 4, (0, 255, 0), 10)

cv2.imshow('emotion_recognition', frame)
cv2.waitKey(30000)


cv2.destroyAllWindows()
