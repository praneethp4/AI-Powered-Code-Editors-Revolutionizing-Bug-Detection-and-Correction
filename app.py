from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

app = Flask(__name__)

# Load the trained model and tokenizer configuration
model = keras.models.load_model('path_to_your_model.h5')
tokenizer_config = np.load('tokenizer_config.npy', allow_pickle=True).item()
# Create a new tokenizer and set its configuration
tokenizer = Tokenizer()
tokenizer.word_index = tokenizer_config['word_index']
tokenizer.num_words = tokenizer_config['num_words']
tokenizer.filters = tokenizer_config['filters']
tokenizer.char_level = tokenizer_config['char_level']

@app.route('/')
def home():
    return render_template('./index.html')

@app.route('/check_code', methods=['POST'])
def check_code():
    try:
        code = request.json['code']
        sequence = tokenizer.texts_to_sequences([code])
        padded_sequence = pad_sequences(sequence, maxlen=model.input_shape[1])
        prediction = model.predict(padded_sequence)[0][0]
        
        return jsonify({'prediction': float(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
