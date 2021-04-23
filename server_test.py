from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from algorithms import guassian_blur, pixelization, pixel_sort, fill_in, pixel_sort2
from skimage.io import imread
import io
import numpy as np
from base64 import encodebytes
from PIL import Image

app = Flask(__name__)
api = Api(app)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
	return '.' in filename and \
		filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_response_image(image_path):
	pil_img = Image.fromarray(np.uint8(image_path)) # reads the PIL image
	byte_arr = io.BytesIO()
	pil_img.save(byte_arr, format='JPEG') # convert the PIL image to byte array
	encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
	return encoded_img

class Censor(Resource):
	def get(self):
		return {'hello': 'world'}

	def post(self):
		options=request.args.get('options', None)
		options = options.strip('][').split(', ')
		# reads file streams and inputs them in correct array structure
		files = request.files.to_dict()
		img = imread(io.BytesIO(files['image'].read()))[:,:,:3]
		mask_img = imread(io.BytesIO(files['mask'].read()))[:,:,:1].astype(np.float)
		# runs guassian blur on image with mask
		if('pixelization' in options):
			img = pixelization(img, mask_img)
		if('gaussian' in options):
			img = guassian_blur(img, mask_img, 10)
		if('pixel_sort' in options):
			img = pixel_sort(img, mask_img)
		if('fill_in' in options):
			img = fill_in(img, mask_img)
		if('pixel_sort2' in options):
			img = pixel_sort2(Image.open(files['image']), Image.open(files['mask']))
		# encodes image in base64 before sending
		encoded_img = get_response_image(img)
		my_message = 'here is my message'
		response =  { 'Status' : 'Success', 'message': my_message , 'ImageBytes': encoded_img}
		return jsonify(response)

api.add_resource(Censor, '/api/censor')

if __name__ == '__main__':
	app.run(debug=True,host="0.0.0.0",port=5001)
