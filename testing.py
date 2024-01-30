import cv2 as cv 
import numpy as np
import tensorflow as tf 

mod=tf.keras.models.load_model("rJ_31.h5")


tol=cv.VideoCapture(0)


WIDTH=tol.get(cv.CAP_PROP_FRAME_WIDTH)
HEIGHT=tol.get(cv.CAP_PROP_FRAME_HEIGHT)

def last(image, mod):
    img = cv.resize(image, (28, 28))
    img = img / 255
    img = img.reshape(1, 28, 28, 1) 
    predict = mod.predict(img)
    return np.argmax(predict)


while(True):
    r,y=tol.read()

    bb=(60,60)

    bbox = [(int(WIDTH // 2 - bb[0] // 2), int(HEIGHT // 2 - bb[1] // 2)),
            (int(WIDTH // 2 + bb[0] // 2), int(HEIGHT // 2 + bb[1] // 2))]
    
    
    dr=y[bbox[0][1]:bbox[1][1],bbox[0][0]:bbox[1][0]]


    dr=cv.cvtColor(dr,cv.COLOR_BGR2GRAY)
    
    dr=cv.resize(dr,(200,200))


    
    cv.rectangle(y,(bbox[0][0],bbox[0][1]),(bbox[1][0],bbox[1][1]),(255,255,255),thickness=4)

    cv.imshow("input",dr)
    cv.imshow("full",y)
   
    print(last(dr,mod))


    
   
    
    if cv.waitKey(1)& 0xFF==27:
        break
cv.destroyAllWindows()    