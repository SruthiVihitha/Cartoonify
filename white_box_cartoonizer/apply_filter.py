import os
import uuid
import cv2
import numpy as np
from flask import Blueprint, request, jsonify, current_app, url_for

# Create a Blueprint for modularity
apply_filter_bp = Blueprint('apply_filter', __name__)

def oil_painting_effect(image):
    # Custom oil painting effect using bilateral filter and color quantization
    def color_quantization(img, k=10):
        Z = img.reshape((-1, 3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        quantized = centers[labels.flatten()]
        return quantized.reshape(img.shape)
    smooth = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    gray = cv2.cvtColor(smooth, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges_inv = cv2.bitwise_not(edges)
    edges_colored = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2RGB)
    quantized = color_quantization(smooth, k=20)
    oil_painting = cv2.bitwise_and(quantized, edges_colored)
    return oil_painting

def pencil_sketch_effect(image):
    # Pencil sketch effect (grayscale, invert, blur, dodge blend)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    inverted = 255 - gray
    blur = cv2.GaussianBlur(inverted, (21, 21), sigmaX=0, sigmaY=0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    sketch_rgb = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
    return sketch_rgb

@apply_filter_bp.route('/apply_filter', methods=['POST'])
def apply_filter():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image provided'})
    file = request.files['image']
    filter_type = request.form.get('filter_type')
    # Save uploaded file temporarily
    filename = f"{uuid.uuid4()}.jpg"
    upload_folder = current_app.config.get('CARTOONIZED_FOLDER', 'static/cartoonized_images')
    os.makedirs(upload_folder, exist_ok=True)
    filepath = os.path.join(upload_folder, filename)
    file.save(filepath)
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if filter_type == 'oil_painting':
        result = oil_painting_effect(img)
        suffix = 'oil'
    elif filter_type == 'pencil_sketch':
        result = pencil_sketch_effect(img)
        suffix = 'sketch'
    else:
        os.remove(filepath)
        return jsonify({'success': False, 'error': 'Invalid filter type'})

    result_filename = f"{uuid.uuid4()}_{suffix}.jpg"
    result_path = os.path.join(upload_folder, result_filename)
    cv2.imwrite(result_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    os.remove(filepath)

    filtered_url = url_for('static', filename=f"cartoonized_images/{result_filename}")
    # In your apply_filter endpoint, return URLs like this:
return jsonify({
    'success': True,
    'filtered_url': url_for('serve_uploaded_files', filename=result_filename)
})