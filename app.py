from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import os
import uuid
from tensorflow.keras.preprocessing import image as keras_image
from model_definitions import build_inceptionresnetv2_with_cbam, build_custom_cnn_with_deep_attention, build_xception_with_cbam

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Build models
model_task1 = build_inceptionresnetv2_with_cbam(num_classes=10)
model_task2 = build_custom_cnn_with_deep_attention(num_classes=10)
model_task3 = build_xception_with_cbam(num_classes=18)

# Load weights
model_task1.load_weights('models/inceptionresnetv2_best_weights.weights.h5')
model_task2.load_weights('models/custom_cnn_variety_best_weights.weights.h5')
model_task3.load_weights('models/xception_age_best_weights.weights.h5')

def preprocess_image(img_path, target_size=(224, 224)):
    img = keras_image.load_img(img_path, target_size=target_size)
    img_array = keras_image.img_to_array(img)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

def get_top_predictions(predictions, class_labels, top_n=3):
    top_indices = np.argsort(predictions[0])[::-1][:top_n]
    return [(class_labels[i], float(predictions[0][i])) for i in top_indices]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        filename = str(uuid.uuid4())[:8] + os.path.splitext(file.filename)[1]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        Image.open(file.stream).save(filepath)

        processed_img = preprocess_image(filepath)

        # Predict
        pred_task1 = model_task1.predict(processed_img)
        pred_task2 = model_task2.predict(processed_img)
        pred_task3 = model_task3.predict(processed_img)

        # Replace with your actual class labels
        class_labels_task1 = ['bacterial_leaf_blight', 'bacterial_leaf_streak', 'bacterial_panicle_blight', 'blast', 'brown_spot', 'dead_heart', 'downy_mildew', 'hispa', 'normal', 'tungro']
        class_labels_task2 = ['ADT45', 'AndraPonni', 'AtchayaPonni', 'IR20', 'KarnatakaPonni',
                              'Othanel', 'Ponni', 'RR', 'Surya', 'Zonal']
        age_classes = [45, 47, 50, 55, 57, 60, 62, 65, 66, 67, 68, 70, 72, 73, 75, 77, 80, 82]


        top3_task1 = get_top_predictions(pred_task1, class_labels_task1)
        top3_task2 = get_top_predictions(pred_task2, class_labels_task2)
        age = get_top_predictions(pred_task3, age_classes)

        return jsonify({
            'image_url': f'/static/uploads/{filename}',
            'condition_preds': top3_task1,
            'variety_preds': top3_task2,
            'age': age
        })

    return render_template('index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)