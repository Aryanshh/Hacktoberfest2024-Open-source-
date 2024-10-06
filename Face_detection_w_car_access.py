pip install opencv-python face-recognition # Dependencies 

import face_recognition
import cv2
import numpy as np

# Load a sample picture and learn how to recognize it.
authorized_image = face_recognition.load_image_file("authorized_person.jpg")
authorized_face_encoding = face_recognition.face_encodings(authorized_image)[0]

# Initialize some variables
face_locations = []
face_encodings = []

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Resize frame to speed up face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (OpenCV uses BGR) to RGB (face_recognition uses RGB)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # Check if any face matches the authorized face
    access_granted = False
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        match = face_recognition.compare_faces([authorized_face_encoding], face_encoding)

        if match[0]:
            access_granted = True
            break

    # Display the results
    for (top, right, bottom, left) in face_locations:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Label the result as authorized or not
        label = "Access Granted" if access_granted else "Access Denied"
        color = (0, 255, 0) if access_granted else (0, 0, 255)

        # Draw the label
        cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    # Display the resulting frame
    cv2.imshow('Car Access System', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
