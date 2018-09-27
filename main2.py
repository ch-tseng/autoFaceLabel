import cv2
from imutils.face_utils import rect_to_bb
import dlib
import imutils
import os, time

#-----------------------------------
datasetPath = "faceYolo/"
imgPath = "images/"
labelPath = "labels/"
imgType = "jpg"  # jpg, png

xml_file = "xml_file.txt"
object_xml_file = "xml_object.txt"

cam_resolution = (1280,1024)
labelName = "face"
minFaceSize = (30, 30)
faceDetectType = "dlib"   #cascade or dlib
#for Cascade type
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cascade_scale = 1.2
cascade_neighbors = 10

dlib_detectorRatio = 1
folderCharacter = "/"  # \\ is for windows

#------------------------------------

if not os.path.exists(datasetPath):
    os.makedirs(datasetPath)

if not os.path.exists(datasetPath + imgPath):
    os.makedirs(datasetPath + imgPath)

if not os.path.exists(datasetPath + labelPath):
    os.makedirs(datasetPath + labelPath)

def writeObjects(label, bbox):
    with open(object_xml_file) as file:
        file_content = file.read()

    file_updated = file_content.replace("{NAME}", label)
    file_updated = file_updated.replace("{XMIN}", str(bbox[0]))
    file_updated = file_updated.replace("{YMIN}", str(bbox[1]))
    file_updated = file_updated.replace("{XMAX}", str(bbox[0] + bbox[2]))
    file_updated = file_updated.replace("{YMAX}", str(bbox[1] + bbox[3]))

    return file_updated

def getFaces_dlib(img):
    detector = dlib.get_frontal_face_detector()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector( gray , dlib_detectorRatio)

    bboxes = []
    for rect in rects:
        (x, y, w, h) = rect_to_bb(rect)
        if(w>minFaceSize[0] and h>minFaceSize[1]):
            bboxes.append((x, y, w, h))

    return bboxes

def getFaces_cascade(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor= cascade_scale,
        minNeighbors=cascade_neighbors,
        minSize=minFaceSize,
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    bboxes = []
    for (x,y,w,h) in faces:
        if(w>minFaceSize[0] and h>minFaceSize[1]):
            bboxes.append((x, y, w, h))

    return bboxes

def generateXML(img, filename, fullpath, bboxes):
    xmlObject = ""
    for bbox in bboxes:
        xmlObject = xmlObject + writeObjects(labelName, bbox)

    print("SHAPE:", img.shape)
    with open(xml_file) as file:
        xmlfile = file.read()

    (h, w, ch) = img.shape
    xmlfile = xmlfile.replace( "{WIDTH}", str(w) )
    xmlfile = xmlfile.replace( "{HEIGHT}", str(h) )
    xmlfile = xmlfile.replace( "{FILENAME}", filename )
    xmlfile = xmlfile.replace( "{PATH}", fullpath + filename )
    xmlfile = xmlfile.replace( "{OBJECTS}", xmlObject )

    return xmlfile

def putText(image, text, x, y, color=(255,255,255), thickness=1, size=1.2):
    if x is not None and y is not None:
        cv2.putText( image, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)
    return image


start_time = time.time()
VIDEO_IN = cv2.VideoCapture(1)
VIDEO_IN.set(cv2.CAP_PROP_FRAME_WIDTH, cam_resolution[0])
VIDEO_IN.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_resolution[1])

i = 0
while True:
    hasFrame, frame = VIDEO_IN.read()
    if not hasFrame:
            print("Done processing !!!")
            print("--- %s seconds ---" % (time.time() - start_time))
            break

    if(faceDetectType == "dlib"):
        faces = getFaces_dlib(frame)

    else:
        faces = getFaces_cascade(frame)

    frameCopy = frame.copy()
    frameCopy = putText(frameCopy, str(len(faces))+" faces", 10,30, color=(0,255,0), thickness=2, size=0.8)

    if(len(faces)>0):
        for (x,y,w,h) in faces:
            cv2.rectangle( frameCopy,(x,y),(x+w,y+h),(0,255,0),2)

        filename = str(time.time()) + "." + str(i)
        #save images to dataset
        cv2.imwrite(datasetPath + imgPath + filename + "." + imgType, frame)

        xmlfilename = filename + ".xml"
        xmlContent = generateXML(frame, xmlfilename, datasetPath + labelPath + xmlfilename, faces)
        file = open(datasetPath + labelPath + xmlfilename, "w")
        file.write(xmlContent)
        file.close

        frameCopy = putText(frameCopy, "saved to "+xmlfilename, 10,80, color=(0,255,0), thickness=2, size=0.8)
        i += 1

    cv2.imshow("Frame", imutils.resize(frameCopy, width=640))
    cv2.waitKey(1)
