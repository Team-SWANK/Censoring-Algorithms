import os, sys, io, base64, requests
from skimage.io import imread
import matplotlib.pyplot as plt

# decodes base64 string into RGB image
def stringToRGB(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    return imread(io.BytesIO(imgdata))

os.environ['NO_PROXY'] = '127.0.0.1'
url = 'http://127.0.0.1:5001/api/censor?options=[pixel_sort2]'

img = open(sys.argv[1], 'rb')
img_mask = open(sys.argv[2], 'rb')

files={'image': (sys.argv[1],img,'multipart/form-data'), 'mask': (sys.argv[2],img_mask,'multipart/form-data') }

x = requests.post(url, files=files)
print(x.json()['message'])

img = stringToRGB(x.json()['ImageBytes'])
plt.imshow(img)
plt.show()
