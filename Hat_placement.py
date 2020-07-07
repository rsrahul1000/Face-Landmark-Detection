import cv2
import numpy as np
import dlib
from PIL import Image, ImageDraw, ImageFilter

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture(0)
#hat = Image.open('graduation_hat.png') #cv2.imread('graduation_hat.png')
hat = cv2.imread('graduation_hat_image.jpg')
hat = cv2.resize(hat, (100, 100), interpolation=cv2.INTER_AREA)
print('hat shape: '+ str(hat.shape))
# upper and lower range of HSV
lower = np.array([6,10,68])
upper = np.array([30,36,122])

# create kernel for image dilation
kernel = np.ones((3,3),np.uint8)

# Set initial value of weights
#alpha = 0.4
while True:
	_, frame = cap.read()
	#print('frame shape: ' + str(frame.shape))
	gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
	faces = detector(gray)
	#print(frame.shape) #((240, 195, 3), (1080, 1920, 3)) 200:300, 300:400
	#break
	for face in faces:
		x1 = face.left()
		y1 = face.top()
		x2 = face.right()
		y2 = face.bottom()
		x_avg = []
		y_avg = []
		
		landmarks = predictor(image = gray, box=face)
		for n in range(27, 36):
			x = landmarks.part(n).x
			y = landmarks.part(n).y
			x_avg.append(x)
			y_avg.append(y)
			
		x = int(sum(x_avg)/len(x_avg))
		y = int(sum(y_avg)/len(y_avg))

		# extract the area where we will place the logo
		# the dimensions of this area should match with those of the logo
		mini_frame = frame[200:300, 300:400, :]

		# create HSV image
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		
		# create binary mask 
		mask = cv2.inRange(hsv, lower, upper)

		# perform dilation on the mask to reduce noise
		dil = cv2.dilate(mask,kernel,iterations = 5)
		
		# extract the area where we will place the logo
		# create 3 channels 
		mini_dil = np.zeros_like(mini_frame)
		print(mini_dil.shape)
		mini_dil[:, :, 0] = dil[200:300, 300:400]
		mini_dil[:, :, 1] = dil[200:300, 300:400]
		mini_dil[:, :, 2] = dil[200:300, 300:400]

		# copy image of the hat
		hat_copy = hat.copy()
		print(hat_copy.shape)
		# set pixel values to 1 where the pixel values of the mask is 0
		#assert(hat_copy[mini_dil == 0])
		hat_copy[mini_dil == 0] = 1
		
		# set pixel values to 1 where the pixel values of the logo is 0
		#assert(hat_copy[logo == 0])
		hat_copy[hat == 0] = 1
		
		# set pixel values to 1 where the pixel values of the logo is not 1
		mini_frame[hat_copy != 1] = 1
		
		# merge images (array multiplication)
		mini_frame = mini_frame*hat_copy
		
		# insert logo in the frame
		frame[200:300, 300:400, :] = mini_frame
		
		# resize the frame (optional)
		#frame = cv2.resize(frame, (480, 270), interpolation = cv2.INTER_AREA)


	cv2.imshow(winname="Face", mat=frame)
	if cv2.waitKey(delay=1) == 27:
		break

cap.release()
cv2.destroyAllWindows()	
