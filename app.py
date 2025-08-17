
import os
import io
import uuid
import sys
import yaml
import traceback

with open('./config.yaml', 'r') as fd:
    opts = yaml.safe_load(fd)

sys.path.insert(0, './white_box_cartoonizer/')

import cv2
from flask import Flask, render_template, make_response, flash, request, jsonify, url_for,send_from_directory
from urllib.parse import unquote
import flask
from PIL import Image
import numpy as np
import skvideo.io
if opts['colab-mode']:
    from flask_ngrok import run_with_ngrok #to run the application on colab using ngrok

##adding
# Add these after your imports
class CustomOilPaintingEffect:
    def __init__(self):
        pass

    def color_quantization(self, image, k=10):
        Z = image.reshape((-1, 3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        quantized = centers[labels.flatten()]
        return quantized.reshape(image.shape)

    def apply_effect(self, image):
        smooth = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        gray = cv2.cvtColor(smooth, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges_inv = cv2.bitwise_not(edges)
        edges_colored = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2RGB)
        quantized = self.color_quantization(smooth, k=20)
        return cv2.bitwise_and(quantized, edges_colored)

class AdvancedPencilSketch:
    def __init__(self, max_process_dim=1600):
        self.max_process_dim = max_process_dim

    def smart_resize(self, image):
        h, w = image.shape[:2]
        if max(h, w) > self.max_process_dim:
            ratio = self.max_process_dim / max(h, w)
            return cv2.resize(image, (int(w*ratio), int(h*ratio)), interpolation=cv2.INTER_LANCZOS4)
        return image

    def infer(self, image):
        processed_img = self.smart_resize(image)
        sharpened = cv2.filter2D(processed_img, -1, np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]))
        gray = cv2.cvtColor(sharpened, cv2.COLOR_RGB2GRAY)
        inv_gray = 255 - gray
        blur = cv2.GaussianBlur(inv_gray, (21,21), 0)
        return cv2.divide(gray, 255-blur, scale=256)

##adding end
from cartoonize import WB_Cartoonize


if not opts['run_local']:
    if 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
        from gcloud_utils import upload_blob, generate_signed_url, delete_blob, download_video
    else:
        raise Exception("GOOGLE_APPLICATION_CREDENTIALS not set in environment variables")
    from video_api import api_request
    # Algorithmia (GPU inference)
    import Algorithmia

app = Flask(__name__, static_folder='static', static_url_path='/static')
if opts['colab-mode']:
    run_with_ngrok(app)   #starts ngrok when the app is run

app.config['CARTOONIZED_FOLDER'] = 'static/cartoonized_images'
app.config['UPLOAD_FOLDER'] = 'static/uploads' 
 # Added3
@app.route('/uploads/<path:filename>')
def serve_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
##adding2
@app.route('/apply_filter', methods=['POST'])
def apply_filter():
    try:
        original_img_filename = request.form.get('original_image_path')
        filter_type = request.form.get('filter_type')
        
        if not original_img_filename or not filter_type:
            return jsonify({'success': False, 'error': 'Missing parameters'})

        full_path = os.path.join(app.config['UPLOAD_FOLDER'], original_img_filename)
        
        if not os.path.exists(full_path):
            return jsonify({'success': False, 'error': 'Image not found'})

        # Process image
        img = cv2.imread(full_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if filter_type == 'oil_painting':
            processor = CustomOilPaintingEffect()
            result = processor.apply_effect(img)
        elif filter_type == 'pencil_sketch':
            processor = AdvancedPencilSketch()
            result = processor.infer(img)
        else:
            return jsonify({'success': False, 'error': 'Invalid filter type'})

        # Save result
        result_filename = f"filtered_{uuid.uuid4().hex}.jpg"
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        cv2.imwrite(result_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

        return jsonify({
            'success': True,
            'filtered_url': url_for('static', filename=f"uploads/{result_filename}")
        })

    except Exception as e:
        print(f"Error in apply_filter: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

##adding2 end
app.config['OPTS'] = opts

## Init Cartoonizer and load its weights 
wb_cartoonizer = WB_Cartoonize(os.path.abspath("white_box_cartoonizer/saved_models/"), opts['gpu'])

def convert_bytes_to_image(img_bytes):
    """Convert bytes to numpy array

    Args:
        img_bytes (bytes): Image bytes read from flask.

    Returns:
        [numpy array]: Image numpy array
    """
    
    pil_image = Image.open(io.BytesIO(img_bytes))
    if pil_image.mode=="RGBA":
        image = Image.new("RGB", pil_image.size, (255,255,255))
        image.paste(pil_image, mask=pil_image.split()[3])
    else:
        image = pil_image.convert('RGB')
    
    image = np.array(image)
    
    return image

@app.route('/')
@app.route('/cartoonize', methods=["POST", "GET"])
def cartoonize():
    opts = app.config['OPTS']
    if flask.request.method == 'POST':
        try:
            if flask.request.files.get('image'):
                img_file = flask.request.files["image"]
                img_bytes = img_file.read()

                # Save original uploaded image
                img_name = str(uuid.uuid4())
                original_img_filename = img_name + ".jpg"
                original_img_path = os.path.join(app.config['UPLOAD_FOLDER'], original_img_filename)
                
                # Ensure upload directory exists
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                
                with open(original_img_path, 'wb') as f:
                    f.write(img_bytes)

                # Convert to numpy array for cartoonization
                image = convert_bytes_to_image(img_bytes)
                cartoon_image = wb_cartoonizer.infer(image)
                
                # Ensure cartoonized directory exists
                os.makedirs(app.config['CARTOONIZED_FOLDER'], exist_ok=True)
                
                cartoonized_img_name = os.path.join(app.config['CARTOONIZED_FOLDER'], img_name + ".jpg")
                cv2.imwrite(cartoonized_img_name, cv2.cvtColor(cartoon_image, cv2.COLOR_RGB2BGR))

                return render_template(
                    "index_cartoonized.html",
                    cartoonized_image=url_for('static', filename=f"cartoonized_images/{img_name}.jpg"),
                    original_image=original_img_filename
                )

        except Exception:
            print(traceback.print_exc())
            flash("Our server hiccuped :/ Please upload another file! :)")
            return render_template("index_cartoonized.html")
    else:
        # Handle GET requests and filtered image display
         filtered_image = request.args.get('filtered_image')
    if filtered_image:
        # Ensure the URL is properly formatted
        filtered_image = filtered_image.replace('/static/uploads/', '/uploads/')
        if not filtered_image.startswith(('http://', 'https://', '/')):
            filtered_image = '/' + filtered_image
    
    return render_template(
        "index_cartoonized.html",
        cartoonized_image=cartoonized_img_name if 'cartoonized_img_name' in locals() else None,
        original_image=original_img_filename if 'original_img_filename' in locals() else None,
        filtered_image=filtered_image
    )
if __name__ == "__main__":
    # Commemnt the below line to run the Appication on Google Colab using ngrok
    if opts['colab-mode']:
        app.run()
    else:
        app.run(debug=True, host='127.0.0.1', port=int(os.environ.get('PORT', 8080)))