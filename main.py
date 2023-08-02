import cv2

#creating classifiers

face_classifier = cv2.CascadClassifier('haarcascade_frontface_default.xml')
eye_classifier = cv2.CascadeClassifier('haarcascade_eye.xml')

#read image, convert to grayscale, run classifier

image = cv2.imread('example.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
face =  face_classifier.detectMultiScale(image_gray, 1.1,4)

#create the rectangles

for(a,b,c,d) in face:
    cv2.rectangle(image,(a,b),(a+b,c+d),(255,255,0),2)
    roi_gray = image_gray[b:b+d, a:a+c]
    roi_color = image[b:b+d, a:a+c]
    eyes = eye_classifier = eye.classifier.detectMultiScale(roi_gray)
    for(ea,eb,ec,ed)in eyes:
        cv2.rectangle(roi_color, (ea,eb),(ea+ec,eb+ed), (0,127,255),2)

#show final image

cv2.imshow('img',image)
cv2.waitKey()