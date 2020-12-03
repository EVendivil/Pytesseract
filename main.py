from flask import Flask, render_template, request, make_response
from flask_restful import Api, Resource
from PIL import Image
from cv2 import imdecode, cvtColor, IMREAD_UNCHANGED, COLOR_BGR2RGB, rectangle, imencode, cvtColor, resize
from pytesseract import Output

from io import StringIO 
import numpy as np
import random
import io
import pytesseract
import base64


app = Flask(__name__)
api = Api(app)

@app.route("/")
def home():
	print("im home")	
	return render_template("home.html")

def show_contours(image):		
	d = pytesseract.image_to_data(image, output_type=Output.DICT)
	n_boxes = len(d['level'])
	boxes = cvtColor(np.float32(image.copy()), COLOR_BGR2RGB)
	image_size = image.size
	current_x = 0
	current_y = 0
	current_w = 0
	current_h = 0
	check = False
	output_bbox = []
	for i in range(n_boxes):
	    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])  
	    boxes = rectangle(boxes, (x, y), (x + w, y + h), (0, 255, 0), 0)
	    if((x == 0 and y == 0) or (image_size[0] == w and image_size[1] == h)):
	        continue
	    if not check:
	        current_x = x
	        current_y = y
	        current_w = w
	        current_h = h
	        check = True
	        boxes = rectangle(boxes, (x, y), (x + w, y + h), (0, 255, 0), 0)
	    if check and ((x >= current_x and x <= (current_x + current_w)) and (y >= current_y and y <= (current_y + current_h))):
	        continue
	    else:
	        boxes = rectangle(boxes, (x, y), (x + w, y + h), (0, 255, 0), 0)
	        check = False
	boxes = rectangle(boxes, (x, y), (x + w, y + h), (0, 255, 0), 0)
	boxes = resize(boxes, (800,800))
	img = Image.fromarray(boxes.astype('uint8'))
	output = io.BytesIO()
	img.save(output, format="PNG")
	base64Image = base64.b64encode(output.getvalue())
	output.close()
	return base64Image

@app.route("/upload_image", methods=['POST'])
def upload_image():
	if request.files:
			image = request.files["image"]
			cv_image = imdecode(np.frombuffer(image.read(), np.uint8), IMREAD_UNCHANGED)
			cv_image = cvtColor(cv_image, COLOR_BGR2RGB)
			cv_image = Image.fromarray(cv_image)

			width, height = cv_image.size
			w_scale = 1000/width
			h_scale = 1000/height

			ocr_df = pytesseract.image_to_data(cv_image, output_type='data.frame')
			ocr_df = ocr_df.dropna() \
               .assign(left_scaled = ocr_df.left*w_scale,
                       width_scaled = ocr_df.width*w_scale,
                       top_scaled = ocr_df.top*h_scale,
                       height_scaled = ocr_df.height*h_scale,
                       right_scaled = lambda x: x.left_scaled + x.width_scaled,
                       bottom_scaled = lambda x: x.top_scaled + x.height_scaled)

			float_cols = ocr_df.select_dtypes('float').columns
			ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
			ocr_df.reset_index(drop=True, inplace=True)
			output = ocr_df[['line_num','word_num', 'text']]
			image = show_contours(cv_image)
			# print("/////////////////////", image)
			# print("CV IMAGE", response)
			# return response
			# pil_image = Image.open(image, mode='r')
			# print("TYPE", type(pil_image))

	return render_template("upload_image.html", happy="True", tables=[output.to_html(classes='data')], titles=output.columns.values, image=image.decode('ascii'))

if __name__ == "__main__":
	# app.run(debug=True)
	app.run(debug=True, host='0.0.0.0', port=80)

