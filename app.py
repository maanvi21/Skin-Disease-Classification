from flask import Flask, request, render_template
from predict import predict_image
from PIL import Image

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])  # This is the missing part
def index():
    return render_template('index.html')  # This should match a file in the templates folder
@app.route('/predict', methods=['POST','GET'])
@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file uploaded"

    file = request.files['image']  # Match the name="image" in HTML
    if file.filename == '':
        return "No file selected"

    image = Image.open(file.stream)
    result = predict_image(image)
    return f"The model predicts: {result}"

if __name__ == '__main__':
    app.run(debug=True)
