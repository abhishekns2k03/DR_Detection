from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(_name_)

# Load the trained model
model = load_model('our_DR_detection_model.h5')

# Class names
class_names = ['Mild NPDR', 'Moderate NPDR', 'PDR', 'Severe NPDR', 'Image is incorrect']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded file
        img_file = request.files['file']
        img_path = 'uploads/' + img_file.filename
        img_file.save(img_path)

        # Preprocess the image
        img = image.load_img(img_path, target_size=(600, 600))
        img_array = image.img_to_array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]

        return render_template('result.html', prediction=class_names[predicted_class])

if _name_ == '_main_':
    app.run(debug=True)