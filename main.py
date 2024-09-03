from flask import Flask, request, jsonify
import cv2
import pytesseract
import easyocr
import secrets

app = Flask(__name__)

# Initialize OCR engines
tesseract_path = '/usr/local/bin/tesseract'  # Example path to Tesseract executable
easyocr_reader = easyocr.Reader(['en'])

# Security class for API key management
class Security:
    def __init__(self):
        self.api_keys = {}

    def generate_api_key(self, user_id):
        api_key = secrets.token_hex(16)
        self.api_keys[user_id] = api_key
        return api_key

    def validate_api_key(self, user_id, api_key):
        return self.api_keys.get(user_id) == api_key

security = Security()

# Image preprocessing function
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    enhanced = cv2.equalizeHist(blurred)
    return enhanced

# OCR processing function
def recognize_text(image, method='tesseract'):
    if method == 'tesseract':
        return pytesseract.image_to_string(image)
    elif method == 'easyocr':
        results = easyocr_reader.readtext(image)
        return ' '.join([result[1] for result in results])
    else:
        raise ValueError('Unsupported OCR method.')

# Batch processing function
def process_batch(image_paths):
    results = []
    for path in image_paths:
        image = cv2.imread(path)
        processed_image = preprocess_image(image)
        text = recognize_text(processed_image)
        results.append(text)
    return results

@app.route('/ocr_process', methods=['POST'])
def ocr_process():
    if 'api_key' not in request.headers:
        return jsonify({'error': 'API key required'}), 401
    user_id = request.headers.get('user_id')
    api_key = request.headers.get('api_key')
    if not security.validate_api_key(user_id, api_key):
        return jsonify({'error': 'Invalid API key'}), 403

    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No image provided'}), 400

    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    processed_image = preprocess_image(image)
    text = recognize_text(processed_image)
    return jsonify({'recognized_text': text})

@app.route('/generate_api_key', methods=['POST'])
def generate_api_key():
    user_id = request.json.get('user_id')
    if not user_id:
        return jsonify({'error': 'User ID required'}), 400
    api_key = security.generate_api_key(user_id)
    return jsonify({'api_key': api_key})

@app.route('/batch_process', methods=['POST'])
def batch_process():
    if 'api_key' not in request.headers:
        return jsonify({'error': 'API key required'}), 401
    user_id = request.headers.get('user_id')
    api_key = request.headers.get('api_key')
    if not security.validate_api_key(user_id, api_key):
        return jsonify({'error': 'Invalid API key'}), 403

    image_paths = request.json.get('image_paths')
    if not image_paths:
        return jsonify({'error': 'No image paths provided'}), 400

    results = process_batch(image_paths)
    return jsonify({'batch_results': results})

if __name__ == '__main__':
    app.run(debug=True)