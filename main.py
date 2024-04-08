import cv2
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
# Path for face image database
path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml");
# function to get the images and label data
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') # grayscale
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            # print(img_numpy[y:y+h,x:x+w].shape)
            ids.append(id)
    return faceSamples,ids
print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
X_train, X_test, y_train, y_test = train_test_split(faces, ids, test_size=0.5, random_state=0)
recognizer.train(X_train, np.array(y_train))

# Save the model into trainer/trainer.yml
recognizer.write('trainer/trainer.yml')
# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

# Assuming you already have 'recognizer' trained and 'X_test' prepared
pred = cv2.face.LBPHFaceRecognizer_create()
pred.read('trainer/trainer.yml')
predicted = []
for test_image in X_test:
    # Convert the image to grayscale (assuming it's not already in grayscale)
    # image = Image.open(test_image).convert('L')
    image_np = np.array(test_image, 'uint8')

    # Resize the face image to match the size used for training
    predictions = pred.predict(image_np)
    predicted.append(predictions[0])

    # Predict the label for the face image

    # Print the predicted label and confidence level
    # print(f"Predicted ID: {predicted_id}, Confidence: {confidence}")

    # Assuming you want to show the image with predicted label and confidence
    # cv2.putText(test_image, f'ID: {predicted_id}, Confidence: {confidence}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
    #             (255, 255, 255), 2)
    # cv2.imshow('Test Image', test_image)
    # cv2.waitKey(0)  # Press any key to move to the next test image
    # cv2.destroyAllWindows()
# Convert predicted list to numpy array for easier comparison
X_train, X_test, y_train, y_test = train_test_split(faces, ids, test_size=0.5, random_state=42)

# Train the LBPH recognizer on the training set
recognizer.train(X_train, np.array(y_train))

# Save the trained model
recognizer.write('trainer/trainer.yml')

# Use the trained model to predict labels for the testing set
predictions = []
for test_img in X_test:
    label, confidence = recognizer.predict(test_img)
    predictions.append(label)

# Calculate accuracy
accuracy = np.mean(np.array(predictions) == np.array(y_test)) * 100
print(f"\n [INFO] Accuracy: {accuracy:.2f}%")
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Predict labels for the test set
predicted = [pred.predict(image) for image in X_test]

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

