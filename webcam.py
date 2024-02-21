import cv2
from closeness import driver


# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    message = driver(frame)
    h,w = frame.shape[:2]
    # Window name in which image is displayed
    window_name = 'Closeness Check'

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    org = (int(h/8),int(w/2))

    # fontScale
    fontScale = 1

    # Blue color in BGR
    color = (0, 0, 0)

    # Line thickness of 2 px
    thickness = 1

    # Using cv2.putText() method
    image = cv2.putText(frame, message, org, font,fontScale, color, thickness, cv2.LINE_AA)
    # Display the resulting frame
    cv2.imshow('frame', image)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()