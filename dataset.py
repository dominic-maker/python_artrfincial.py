import cv2


trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread('images/best.jpg')

# must convert to grayscale
grascaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# face detect
face_coordinates = trained_face_data.detectMultiScale(grascaled_img)

# Draw the rectangle around the faces
(x, y, w, h) = face_coordinates[0]
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 10)

cv2.imshow('face detector', img)
cv2.waitKey()
