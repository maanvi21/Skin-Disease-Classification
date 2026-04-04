from flask import Flask, request, render_template, jsonify
from predict import predict_image
from PIL import Image
import io
import base64
from io import BytesIO

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max upload

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        result = predict_image(image)

        # Convert heatmap (numpy) to base64 PNG
        heatmap_pil = Image.fromarray(result["heatmap"])
        buffered = BytesIO()
        heatmap_pil.save(buffered, format="PNG")
        heatmap_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        response = {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "message": result["message"],
            "heatmap": f"data:image/png;base64,{heatmap_base64}"
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Skin Disease Classification Web App Started!")
    app.run(debug=True, host='0.0.0.0', port=5000)