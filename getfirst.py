import cv2
import urllib.request

# Set the path of the video file
video_path = '/home/andrew/open-inference/data/cropped_mini.mp4'

# Open the video file and read the first frame
video_capture = cv2.VideoCapture(video_path)
success, first_frame = video_capture.read()

# Check if the frame was successfully read
if not success:
    print('Error: Could not read first frame of video')
    exit()

# Save the first frame as an image file
cv2.imwrite('first_frame.jpg', first_frame)

# Display the first frame
cv2.imshow('First Frame', first_frame)

# Wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()