from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2
import os


# //////////////OBJECT DETECTION------------------------------------

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor", "keyboard", "mouse", "paper", "wood"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("Loading Object Detection model...")
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")


# //////////////OBJECT DETECTION--------------------------------------


# //////////////FACE RECOGNITION--------------------------------------

subjects = ["", "Ida", "Mo", "Ariana", "Samuel"]

# function to detect face using OpenCV
def detect_face(img):
    # convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # load OpenCV face detector, I am using LBP which is fast
    # there is also a more accurate but slow Haar classifier
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None

    # under the assumption that there will be only one face,
    # extract the face area
    (x, y, w, h) = faces[0]

    # return only the face part of the image
    return gray[y:y + w, x:x + h], faces[0]

def prepare_training_data(data_folder_path):
    # ------STEP-1--------
    # get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)

    # list to hold all subject faces
    faces = []
    # list to hold labels for all subjects
    labels = []

    # let's go through each directory and read images within it
    for dir_name in dirs:

        # our subject directories start with letter 's' so
        # ignore any non-relevant directories if any
        if not dir_name.startswith("s"):
            continue

        # ------STEP-2--------
        # extract label number of subject from dir_name
        # format of dir name = slabel
        # , so removing letter 's' from dir_name will give us label
        label = int(dir_name.replace("s", ""))

        # build path of directory containing images for current subject subject
        # sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name

        # get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)

        # ------STEP-3--------
        # go through each image name, read image,
        # detect face and add face to list of faces
        for image_name in subject_images_names:

            # ignore system files like .DS_Store
            if image_name.startswith("."):
                continue

            # build image path
            # sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name

            # read image
            image = cv2.imread(image_path)

            # display an image window to show the image
            # cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            # cv2.waitKey(100)

            # detect face
            face, rect = detect_face(image)

            # ------STEP-4--------
            # for the purpose of this tutorial
            # we will ignore faces that are not detected
            if face is not None:
                # add face to list of faces
                faces.append(face)
                # add label for this face
                labels.append(label)

    return faces, labels


# print("Preparing data...")
faces, labels = prepare_training_data("FaceData")

# print total faces and labels
# print("Total faces: ", len(faces))
# print("Total labels: ", len(labels))

print("Creating Face Recognition model...")

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# train our face recognizer of our training faces
face_recognizer.train(faces, np.array(labels))

print("Done!")

def recognize(face):
    # make a copy of the image as we don't want to chang original image
    img = face.copy()

    # predict the image using our face recognizer
    label, confidence = face_recognizer.predict(img)
    # print(label)

    # get name of respective label returned by face recognizer
    label_text = subjects[label]
    # print(label_text)

    return label_text

# //////////////FACE RECOGNITION--------------------------------------

# ////LOADING VIDEO SOURCE
print("Loading Video...")
# vs = VideoStream(src='http://192.168.137.138:8080/video').start()
vs = VideoStream(src=0).start()
# vs = VideoStream(src='C:/Users/Mo/Videos/CCTV/CCTV - C&A Shoplifter in eye-catching bright shirt.mp4').start()

time.sleep(2.0)
fps = FPS().start()

print("Success!")
print("Video Analysis in Progress...")
def recognition():
    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        # --faces/////////////////////////////////////////////////////////////
        image = frame
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        facecascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        # facecascade = cv2.CascadeClassifier("lbpcascade_frontalface.xml")
        faces = facecascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30)
        )

        # print("[INFO] Found {0} Faces.".format(len(faces)))
        for (x, y, w, h) in faces:
            face = recognize(gray[y:y + w, x:x + h])
            roi_color = image[y:y + h, x:x + w]
            if face in subjects:
                text = face
                # cv2.imwrite('images/' + text + '_' + str(w) + str(h) + '.jpg', roi_color)
            else:
                text = "Unknown"
                # cv2.imwrite('images/' + 'Unknown_' + str(w) + str(h) + '.jpg', roi_color)

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

            print("Found {}".format(text))

        # --faces//////////////////////////////////////////////////////////////

        frame = imutils.resize(frame, width=400)

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > 0.6:
                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # draw the prediction on the frame
                label = "{}: {:.2f}%".format(CLASSES[idx],
                                             confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # update the FPS counter
        fps.update()

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()


recognition()
