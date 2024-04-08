import cv2
import numpy as np
from collections import Counter
from PIL import Image
import os
from time import sleep
# Path for face image database
path = 'dataset'
# recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml");
# function to get the images and label data
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize the image to a uniform shape (e.g., 100x100)
        resized_image = cv2.resize(gray_image, (100, 100))

        # Flatten the image into a 1D array
        flattened_pixels = resized_image.flatten()

        # Convert the flattened pixels to a NumPy array
        img_numpy = np.array(flattened_pixels)
        # PIL_img = Image.open(imagePath).convert('L') # grayscale
        # Flatten the 2D array of pixel intensities into a 1D array
        # flattened_pixels = PIL_img.flatten()

        # Append the flattened pixel intensities to the list
        # faceSamples.append(flattened_pixels)
        # img_numpy = np.array(PIL_img,'uint8')
        img_numpy = img_numpy/255.0
        print(img_numpy.shape)
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        # faces = detector.detectMultiScale(img_numpy)
        # print(faces.shape)
        faceSamples.append(img_numpy)
        ids.append(id)
        # for (x,y,w,h) in faces:
        #     faceSamples.append(img_numpy[y:y+h,x:x+w])
        #     ids.append(id)
    return faceSamples,ids
print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
print(ids)
# print(faces)
from sklearn.model_selection import train_test_split

X = faces
print(faces[0])
y = ids

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
# X_train_flattened = [image.flatten() for image in X_train]
X_train = np.array(X_train)
X_test = np.array(X_test)
# Flatten images in X_test
# X_test_flattened = [image.flatten() for image in X_test]
# print(X_test_flattened[0])
from sklearn.svm import SVC
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.pipeline import make_pipeline

# For dimensionality reduction
pca = RandomizedPCA(n_components=44, whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced', C=0.1, gamma = 100)
model = make_pipeline(pca, svc)
# print("Shape of X_train_flattened:", np.array(X_train_flattened).shape)
# print("Shape of X_test_flattened:", np.array(X_test_flattened).shape)

model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

predictions = model.predict(X_test)

accuracy = accuracy_score(predictions, y_test)
print(f"Accuracy: {accuracy}")

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Predict labels for the test set
predictions = model.predict(X_test)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)

# Calculate precision, recall, and F1-score
precision = precision_score(y_test, predictions, average='macro')
recall = recall_score(y_test, predictions, average='macro')
f1 = f1_score(y_test, predictions, average='macro')

# Print confusion matrix, precision, recall, and F1-score
print("Confusion Matrix:")
print(conf_matrix)
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

# recognizer.train(faces, np.array(ids))
# # Save the model into trainer/trainer.yml
# recognizer.write('trainer/trainer.yml')
# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))


def recognize():

    cascadePath = "Cascades/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    font = cv2.FONT_HERSHEY_SIMPLEX
    names = ['None', 'Shreeya', 'Shakshi', 'Jinhal']
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)
    count = 10
    arr = [];
    while count:
        sleep(1)
        ret, img = cam.read()
        # sleep(1000)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_region = gray[y:y+h, x:x+w]
            resized_face = cv2.resize(face_region, (100, 100))
            normalized_face = resized_face / 255.0
            id = model.predict(normalized_face.flatten().reshape(1, -1))[0]
            name = names[id]
            # print(name)
            arr.append(name)

            # Calculate confidence score
            confidence = model.decision_function(normalized_face.flatten().reshape(1, -1))
            #
            # # Update name if confidence is below threshold
            # if confidence < -0.5:  # Extracting the first element of the array
            #     name = "Unknown"
            # #
            # cv2.putText(img, name, (x, y - 10), font, 0.8, (255, 255, 255), 2)
            cv2.putText(img, name, (x, y + h + 20), font, 0.5, (255, 255, 255), 1)

        cv2.imshow('camera', img)
        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break
        count -= 1;
    counter = Counter(arr)

    # Find the element with the maximum occurrence
    if arr:
       max_occurrence = max(counter, key=counter.get)
    else:
       max_occurence = "unknown"

    print(max_occurrence)

    # Cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()

recognize()
