#-*-coding:utf-8-*-
import cv2
import dlib
import numpy as np
CASC_PATH = './data/haarcascade_files/haarcascade_frontalface_default.xml'
PREDICTOR_PATH = "./data/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)
detector = dlib.get_frontal_face_detector()
SCALE_FACTOR = 1
class Image_process():
    def format_image(self, image):
        if len(image.shape) > 2 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = cascade_classifier.detectMultiScale(
        image,
        scaleFactor = 1.3,
        minNeighbors = 5
        )
        if not len(faces) > 0:
            return None, None
        max_are_face = faces[0]

        for face in faces:
            if face[2] * face[3] > max_are_face[2] * max_are_face[3]:
                  max_are_face = face

        # face to image
        face_coor =  max_are_face
        image = image[face_coor[1]:(face_coor[1] + face_coor[2]), face_coor[0]:(face_coor[0] + face_coor[3])]
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = cv2.resize(image, (64, 64))
        return  image, face_coor

    def get_landmarks(self, im):
        rects = detector(im, 1)

        if len(rects) > 1:
            raise False
        if len(rects) == 0:
            raise False

        return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])
    def read_im_and_landmarks(self, fname):
        '''im = cv2.resize(fname, (fname.shape[1] * SCALE_FACTOR,
                                fname.shape[0] * SCALE_FACTOR))'''
        s = self.get_landmarks(fname)
        print(type(s))
        for i in range(np.shape(s)[0]):
            cv2.circle(fname, (int(s[i][0, 0]), int(s[i][0, 1])), 1,(0, 255, 0), 1)
        return fname, s
if __name__ == '__main__':
    ImP=Image_process();
    #path = raw_input("Please input a image path\n")
    #path = path
    path = "./test.png"
    Image = cv2.imread(path)
    cv2.imshow("Ori_image",Image)
    Detected_Im, Detected_Coordinate = ImP.format_image(Image)
    cv2.imshow("Detected_image", Detected_Im)
    LandMaek_Img, LandMark = ImP.read_im_and_landmarks(Detected_Im)
    cv2.imshow("LandMaek_Img", LandMaek_Img)
    cv2.waitKey()
