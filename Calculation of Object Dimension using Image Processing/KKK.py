import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import cv2

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_bounding_boxes(image_path, scale):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    measurements = []
    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        width = w * scale
        height = h * scale
        cv2.putText(image, f'Width: {width:.2f} units', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(image, f'Height: {height:.2f} units', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        measurements.append({'width': width, 'height': height})

    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'bounding_boxes_output.jpg')
    cv2.imwrite(output_path, image)
    return output_path, measurements, len(contours)

def detect_curves(image_path, scale):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    curve_measurements = []
    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
        perimeter = cv2.arcLength(contour, True) * scale
        cv2.putText(image, f'Perimeter: {perimeter:.2f} units', tuple(contour[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        curve_measurements.append(perimeter)
    
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'curves_output.jpg')
    cv2.imwrite(output_path, image)
    return output_path, curve_measurements, len(curve_measurements)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            scale = 0.1  # Adjust this value based on your calibration
            bounding_boxes_output_path, measurements, num_boxes = detect_bounding_boxes(filepath, scale)
            curves_output_path, curve_measurements, num_curves = detect_curves(filepath, scale)
            bounding_boxes_output_filename = os.path.basename(bounding_boxes_output_path)
            curves_output_filename = os.path.basename(curves_output_path)
            return render_template('upload.html', filename=filename, bounding_boxes_output_filename=bounding_boxes_output_filename, curves_output_filename=curves_output_filename, measurements=measurements, curve_measurements=curve_measurements, num_boxes=num_boxes, num_curves=num_curves)
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
