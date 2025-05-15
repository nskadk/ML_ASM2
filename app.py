from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import os
import uuid
import cv2
from tensorflow.keras.preprocessing import image as keras_image
from model_definitions import build_inceptionresnetv2_with_cbam, build_custom_cnn_with_deep_attention, build_xception_with_cbam
from final_preprocessing_pipeline import PreprocessingPipeline, create_preprocessing_config
from trainingpipeline import predict_single_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PROCESSED'] = 'static/processed'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED'], exist_ok=True)

CONFIG = {
    'img_size': (224, 224),
    'batch_size': 32,
    'learning_rate': 0.0001,
    'steps_per_epoch': 100,
    'metadata_path': 'meta_train.csv',
    'output_dir': 'output'
}

preprocessing_config = create_preprocessing_config(
    dataset_path='dataset/train_images',
    output_dir='output'
)
pipeline = PreprocessingPipeline(preprocessing_config)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        try:
            # Save uploaded file
            filename = str(uuid.uuid4())[:8] + os.path.splitext(file.filename)[1]
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            Image.open(file.stream).save(filepath)

            
            # Save processed image
            processed_filename = f"processed_{filename}"
            processed_path = os.path.join(app.config['PROCESSED'], processed_filename)

            # Preprocess using pipeline
            processed_img = pipeline.preprocess_single_image(filepath, processed_path )

            if processed_img is None:
                return jsonify({'error': 'Image preprocessing failed'}), 400

            # Save processed image 
            Image.fromarray(processed_img).save(processed_path)

            prediction_results = predict_single_image(
                image_path=processed_path,
                config=CONFIG,
                disease_model_name="inceptionresnetv2",
                variety_model_name="custom_cnn_variety",
                age_model_name="xception_age"
            )

            def convert_numpy_types(obj):
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                if isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                return obj

            response = convert_numpy_types({
                'image_url': f'/static/uploads/{filename}',
                'processed_image': f'/static/processed/{processed_filename}',
                'disease': {
                    'prediction': prediction_results['label'],
                    'top3': prediction_results['top3_disease']
                },
                'variety': {
                    'prediction': prediction_results['variety'],
                    'top3': prediction_results['top3_variety']
                },
                'age': {
                    'prediction': prediction_results['age'],
                    'top3': prediction_results['top3_age']
                }

            })

            return jsonify(response)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

        # finally:
        #     # Cleanup temporary files
        #     if os.path.exists(filepath):
        #         os.remove(filepath)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
