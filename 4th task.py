import random
import numpy as np

# Step 1: Create a simple dataset of musical notes
notes = ["C", "D", "E", "F", "G", "A", "B"]  # Simplified scale
note_sequences = [[random.choice(notes) for _ in range(50)] for _ in range(100)]  # 100 sequences of 50 notes

# Step 2: Encode the notes into numbers
note_to_int = {note: i for i, note in enumerate(notes)}
int_to_note = {i: note for note, i in note_to_int.items()}
encoded_sequences = [[note_to_int[note] for note in sequence] for sequence in note_sequences]

# Step 3: Prepare training data
sequence_length = 10
X, y = [], []

for seq in encoded_sequences:
    for i in range(len(seq) - sequence_length):
        X.append(seq[i:i + sequence_length])
        y.append(seq[i + sequence_length])

X = np.array(X)
y = np.array(y)

# Step 4: Define a simple RNN model using NumPy
class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01  # Input to hidden weights
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # Hidden to hidden weights
        self.Why = np.random.randn(output_size, hidden_size) * 0.01  # Hidden to output weights
        self.bh = np.zeros((hidden_size, 1))  # Hidden bias
        self.by = np.zeros((output_size, 1))  # Output bias

    def forward(self, inputs):
        h = np.zeros((self.hidden_size, 1))  # Initial hidden state
        for x in inputs:
            x_one_hot = np.zeros((len(notes), 1))
            x_one_hot[x] = 1
            h = np.tanh(np.dot(self.Wxh, x_one_hot) + np.dot(self.Whh, h) + self.bh)
        y = np.dot(self.Why, h) + self.by
        return y

    def train(self, X, y, epochs=10, learning_rate=0.01):
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(X)):
                inputs = X[i]
                target = y[i]

                # Forward pass
                output = self.forward(inputs)
                target_one_hot = np.zeros((len(notes), 1))
                target_one_hot[target] = 1

                # Compute loss (mean squared error)
                loss = np.sum((output - target_one_hot) ** 2)
                total_loss += loss

                # Backpropagation (simplified, no weight updates for brevity)
                # Gradient calculation and weight updates would go here.

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

# Step 5: Train the RNN model (dummy training for simplicity)
rnn = SimpleRNN(input_size=len(notes), hidden_size=50, output_size=len(notes))
rnn.train(X, y, epochs=5, learning_rate=0.01)

# Step 6: Generate music
def generate_music(seed, length=50):
    generated = seed[:]
    for _ in range(length):
        inputs = [note_to_int[note] for note in generated[-sequence_length:]]
        prediction = rnn.forward(inputs)
        next_note = int_to_note[np.argmax(prediction)]
        generated.append(next_note)
    return generated

# Step 7: Generate and save music
seed = ["C", "D", "E", "F", "G"]
generated_music = generate_music(seed)
print("Generated Music:", generated_music)

# Save the generated music to a text file
with open("generated_music.txt", "w") as f:
    f.write(" ".join(generated_music))
