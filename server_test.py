from flask import Flask, request, abort, send_file, jsonify
from algorithms import guassian_blur
from skimage.io import imread
import io
import numpy as np
from base64 import encodebytes
from PIL import Image


import matplotlib.pyplot as plt

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
	return '.' in filename and \
		filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_response_image(image_path):
	pil_img = Image.fromarray(np.uint8(image_path)) # reads the PIL image
	byte_arr = io.BytesIO()
	pil_img.save(byte_arr, format='JPEG') # convert the PIL image to byte array
	encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
	return encoded_img

@app.route('/api/censor', methods=['GET', 'POST'])
def post_photo():
	if(request.method == 'GET'):
		options=request.args.get('options', None)
		return str(options)
	elif(request.method == 'POST'):
		# reads file streams and inputs them in correct array structure
		files = request.files.to_dict()
		img = imread(io.BytesIO(files['image'].read()))[:,:,:3]
		mask_img = imread(io.BytesIO(files['mask'].read()))[:,:,:1].astype(np.float)
		# runs guassian blur on image with mask
		return_img = guassian_blur(img, mask_img, 7)
		# encodes image in base64 before sending
		encoded_img = get_response_image(return_img)
		my_message = 'here is my message'
		response =  { 'Status' : 'Success', 'message': my_message , 'ImageBytes': encoded_img}
		return jsonify(response)

if __name__ == '__main__':
	app.run(debug=True)

