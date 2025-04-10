from flask import Flask, request, render_template


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])  # This is the missing part
def index():
    return render_template('index.html')  # This should match a file in the templates folder
@app.route('/predict', methods=['POST','GET'])
def predict():
    pass 
if __name__ == '__main__':
    app.run(debug=True)
