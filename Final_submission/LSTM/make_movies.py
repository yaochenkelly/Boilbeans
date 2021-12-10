import cv2
import os

folder = 'figures'
image_folder = "./results/" + "/figures"
video_folder = "./results/" + "/video/"
if not os.path.exists(os.path.join(os.getcwd(), video_folder)):
    os.mkdir(os.path.join(os.getcwd(), video_folder))

def getEpoch(s):
    print(s)
    return int(s.split('.')[0].split('ch')[1])

video_name = video_folder + '/example0.avi'

images = [(img, getEpoch(img)) for img in os.listdir(image_folder) if img[0]=='e']
images.sort(key=lambda x: x[1])
images = [img[0] for img in images]
frame = cv2.imread(os.path.join(image_folder, images[0]))
images = images[1:]
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 2, (width, height))

for image in images:
    print(image)
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()

