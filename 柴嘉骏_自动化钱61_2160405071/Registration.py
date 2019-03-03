# coding: utf-8
import cv2
import numpy as np

imgA = cv2.imread('/home/hero/Documents/DIP-Homework/Homework1-2.22/Requirement/ImageA.jpg')
imgB = cv2.imread('/home/hero/Documents/DIP-Homework/Homework1-2.22/Requirement/ImageB.jpg')
height, width = imgA.shape[0], imgA.shape[1]
P = np.empty(shape=[3, 0])
Q = np.empty(shape=[3, 0])

def on_EVENT_LBUTTONDOWN_A(event, x, y, flags, param):
    global P
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = np.array((x, y, 1)).reshape([3, 1])
        P = np.concatenate((P, xy), axis=1)
        cv2.circle(imgA, (x, y), 1, (255, 0, 0), thickness=-1)
        cv2.putText(imgA, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("imageA", imgA)
		
def on_EVENT_LBUTTONDOWN_B(event, x, y, flags, param):
    global Q
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = np.array((x, y, 1)).reshape([3, 1])
        Q = np.concatenate((Q, xy), axis=1)
        cv2.circle(imgB, (x, y), 1, (255, 0, 0), thickness=-1)
        cv2.putText(imgB, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("imageB", imgB)
cv2.namedWindow("imageA", 0)
cv2.resizeWindow("imageA", 640, 480)
cv2.setMouseCallback("imageA", on_EVENT_LBUTTONDOWN_A)
cv2.imshow("imageA", imgA)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.namedWindow("imageB", 0)
cv2.resizeWindow("imageB", 640, 480)
cv2.setMouseCallback("imageB", on_EVENT_LBUTTONDOWN_B)
cv2.imshow("imageB", imgB)
cv2.waitKey(0)
cv2.destroyAllWindows()

H1 = Q.dot(P.transpose())
H2 = np.linalg.inv(P.dot(P.transpose()))
H = H1.dot(H2)[:2]

np.set_printoptions(suppress=True)
print("P:", P)
print("Q:", Q)
print("H:", H)

registrated_img = cv2.warpAffine(imgA, H, (height, width))
cv2.imwrite('/home/hero/Documents/DIP-Homework/Homework1-2.22/Requirement/Registration.bmp', registrated_img)
