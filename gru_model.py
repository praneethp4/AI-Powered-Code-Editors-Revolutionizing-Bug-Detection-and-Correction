import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample data
code_samples = [
    "for i in range(10):\n    print(i)",
    "while True:\n    print('Hello, World!')",
    "print('Hello, World'",
    "def my_function():\n    print('Function is working!'\n",
    "print(i)"  
]

labels = [1, 0, 1, 0,1]

# Tokenize and pad sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(code_samples)
sequences = tokenizer.texts_to_sequences(code_samples)
max_sequence_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

# Define and compile the GRU model
model = keras.Sequential([
    keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=max_sequence_length),
    keras.layers.GRU(128),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Convert labels to numpy array
labels = np.array(labels)

# Train the model
model.fit(padded_sequences, labels, epochs=10, batch_size=32, validation_split=0.2)

# Save the model
model.save('path_to_your_model.h5')
print("Model saved")

# Save tokenizer configuration
tokenizer_config = tokenizer.get_config()
np.save('tokenizer_config.npy', tokenizer_config)
print("Tokenizer configuration saved")
