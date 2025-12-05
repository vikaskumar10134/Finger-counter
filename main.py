import cv2
import mediapipe as mp
import time
import os
import hand_track_module as htm

wCam , hCam = 720 , 480

cap = cv2.VideoCapture(0)

# set the width and
cap.set(3 , wCam)
cap.set(4 , hCam)

folderPath = r'D:\Data Science\Real time library\Mediapipe\06-Finger counter projects\FingerImages'
myList = os.listdir(folderPath)
print(myList)

OverlayList = []

for imPath in myList:

    image = cv2.imread(f'{folderPath}\{imPath}')
    #print(f'{folderPath}\{imPath}')

    OverlayList.append(image)

# print(len(OverlayList))
previous_time = 0


# make object for detection the hand
detection = htm.HandDector()

tipIds = [4 , 8 , 12 , 16 , 20]

while True:

    succes , frame = cap.read(0)

    if not succes:
        break

    # call method to detect the hand
    frame = detection.findHands(frame)

    # call method to detect the position of hand
    landmark_list = detection.findPosition(frame , draw = False)



    if len(landmark_list) != 0:

        boolean = []
        fingers = []

        # check for the thumb
        if landmark_list[tipIds[0]][1] > landmark_list[tipIds[0] - 1][1]:
            fingers.append(1)

        else:
            fingers.append(0)



        # check for the other remains finger
        for id in range(1 ,5):

            if landmark_list[tipIds[id]][2] < landmark_list[tipIds[id] - 2][2]:
                 
                fingers.append(1)

            else:
                fingers.append(0)      

        # print(fingers)            

        totalFingers = fingers.count(1)


        # for check img
        OverlayList[totalFingers - 1] = cv2.resize(OverlayList[totalFingers - 1] , (200 , 200))
        # print('shape1' , frame.shape)
        # print('shape2' , OverlayList[0].shape)


        height , width , channel = OverlayList[totalFingers - 1].shape

        frame[0:height , 0:width , :] = OverlayList[totalFingers - 1]

        cv2.rectangle(frame , (20 , 225) , (170 , 425) , (0 , 255 , 0) , cv2.FILLED)
        cv2.putText(frame , str(totalFingers) , (45 , 375 ) , cv2.FONT_HERSHEY_PLAIN , 10 , (255 , 0 , 255) , 25)


    current_time = time.time()

    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    # show on frame 
    cv2.putText(frame , f'FPS : {int(fps)}' , (400 , 70) , cv2.FONT_HERSHEY_PLAIN , 2 , (255 , 255 ,0) , 2)


    # to show the frame
    cv2.imshow('frame ' , frame)

    # user press q it break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):

        print('Quiting....')
        break

cap.release()
cv2.destroyAllWindows()