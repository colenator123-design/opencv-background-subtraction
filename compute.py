import cv2
import numpy as np
cap=cv2.VideoCapture(0)
kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg=cv2.bgsegm.createBackgroundSubtractorMOG()
# Fill image with gray color(set each pixel to gray)
image = np.zeros((480, 640, 3), np.uint8)
image[:] = (128, 128, 128)

cv2.imshow('Result', image)

while(True):
    ret,frame=cap.read()
    
    if not ret:
        break
    mask = np.zeros(frame.shape[:2], dtype="uint8")
    cv2.rectangle(mask, (100, 100), (400, 200), 255, -1)
    
    frame = cv2.flip(frame, 1)
    
    
    fgmask=fgbg.apply(frame)
    
    btn_mask = cv2.bitwise_and(fgmask, fgmask, mask=mask)
    if(cv2.countNonZero(btn_mask) > 0):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY )
    
    cv2.rectangle(frame, (100, 100), (400, 200), (0, 0, 255), 3, cv2.LINE_AA)
    cv2.imshow('frame',frame)
    cv2.imshow('btn', btn_mask)
    
    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyAllWindows()