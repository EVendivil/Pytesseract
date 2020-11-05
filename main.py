from flask import Flask, render_template, request, make_response
from flask_restful import Api, Resource
from PIL import Image
from cv2 import imdecode, cvtColor, IMREAD_UNCHANGED, COLOR_BGR2RGB

import numpy as np
import random
import io
import pytesseract


app = Flask(__name__)
api = Api(app)

@app.route("/")
def home():
	print("im home")
	return render_template("home.html")

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
			output = ocr_df[['block_num','line_num','word_num', 'text']]

			# print("CV IMAGE", response)
			# return response
			# pil_image = Image.open(image, mode='r')
			# print("TYPE", type(pil_image))

	return render_template("upload_image.html", happy="True", tables=[output.to_html(classes='data')], titles=output.columns.values)

if __name__ == "__main__":
	# app.run(debug=True)
	app.run(debug=True, host='0.0.0.0', port=80)

