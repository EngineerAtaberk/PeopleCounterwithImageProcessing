import numpy as np
import cv2

peopleout, peoplein = 0, 0
line = 250
path = 'C:/Users/atabe/OneDrive/Masaüstü/'

cap = cv2.VideoCapture(path + "peopleCounter.mp4")

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

contours_previous = []
contours_now = []


sayac = 0
while cap.isOpened():

    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)

    try:

        thresh = cv2.dilate(fgmask, None, iterations=2)
        thresh = cv2.erode(fgmask, None, iterations=2)
        _,cnts,hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours_now = []

        for c in cnts:

            if cv2.contourArea(c) < 1000:
                continue

            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            contours_now.append([x, y])

        if (len(contours_previous) == 0):
            contours_previous = contours_now
            continue

        # compare contours in new frame and the previous one
        closest_contour_list = []

        for i in range(len(contours_now)):

            minimum = 1000000

            for k in range(len(contours_previous)):

                diff_x = contours_now[i][0] - contours_previous[k][0]
                diff_y = contours_now[i][1] - contours_previous[k][1]

                distance = diff_x * diff_x + diff_y * diff_y

                if (distance < minimum):
                    minimum = distance
                    closest_contour = k

            closest_contour_list.append(closest_contour)

        for i in range(len(contours_now)):

            y_previous = contours_previous[closest_contour_list[i]][1]

            if (contours_now[i][1] < line and y_previous > line):
                peopleout = peopleout + 1

            if (contours_now[i][1] > line and y_previous < line):
                peoplein = peoplein + 1

        contours_previous = contours_now


        cv2.line(frame, (0, line), (frame.shape[1], line), (0, 255, 255), 2)

        cv2.putText(frame, "down: " + str(peopleout), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "up: " + str(peoplein), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Frame', frame)
        cv2.imshow('Background Substraction',fgmask)

    except Exception as e:

        print(e)
        break


    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()  # release video file
cv2.destroyAllWindows()  # close all openCV windows