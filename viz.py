import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QCheckBox,
    QHBoxLayout, QFrame, QSizePolicy
)
from PySide6.QtGui import QPainter, QColor, QPen, QImage, QLinearGradient, QFont
from PySide6.QtCore import Qt, QTimer, QPoint


# ==============================================================
# Simple fully connected network
# ==============================================================
class SimpleMLP:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.params = {}
        for i in range(1, len(layer_sizes)):
            self.params[f"W{i}"] = np.random.randn(layer_sizes[i - 1], layer_sizes[i]) * 0.3
            self.params[f"b{i}"] = np.zeros((1, layer_sizes[i]))

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

    def forward(self, X):
        self.caches = {"A0": X}
        A = X
        for i in range(1, self.num_layers + 1):
            W, b = self.params[f"W{i}"], self.params[f"b{i}"]
            Z = np.dot(A, W) + b
            A = self.relu(Z) if i != self.num_layers else self.softmax(Z)
            self.caches[f"A{i}"] = A
        return A


# ==============================================================
# Drawing Canvas (buffer conversion fixed)
# ==============================================================
class DrawingCanvas(QFrame):
    def __init__(self, size=280):
        super().__init__()
        self.setFixedSize(size, size)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setStyleSheet("background-color: black;")
        self.image = QImage(size, size, QImage.Format_RGB32)
        self.image.fill(Qt.black)
        self.drawing = False
        self.last_point = QPoint()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.position().toPoint() if hasattr(event, "position") else event.pos()

    def mouseMoveEvent(self, event):
        if self.drawing:
            cur = event.position().toPoint() if hasattr(event, "position") else event.pos()
            painter = QPainter(self.image)
            pen = QPen(Qt.white, 22, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.last_point, cur)
            painter.end()
            self.last_point = cur
            self.update()

    def mouseReleaseEvent(self, event):
        self.drawing = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(0, 0, self.image)

    def clear(self):
        self.image.fill(Qt.black)
        self.update()

    def get_flattened_input(self):
        """Convert the drawn image → 28x28 grayscale numpy array (1×784)."""
        small = self.image.scaled(28, 28, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        gray = small.convertToFormat(QImage.Format_Grayscale8)

        # === Universal safe conversion to bytes ===
        bits = gray.bits()
        try:
            # PySide6 < 6.6
            bits.setsize(gray.byteCount())
            arr = np.frombuffer(bits, np.uint8)
        except Exception:
            # PySide6 ≥ 6.6 / PyQt6
            arr = np.frombuffer(bytes(bits), np.uint8)

        arr = arr.reshape((28, 28)).astype(np.float32) / 255.0
        return arr.flatten().reshape(1, -1)


# ==============================================================
# Visualizer GUI
# ==============================================================
class NNVisualizer(QWidget):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp
        self.setWindowTitle("Digit Recognizer & Neural Network Visualizer")
        self.resize(1000, 700)

        main_layout = QVBoxLayout()

        self.label = QLabel("Draw a digit (white on black) and click Recognize.")
        self.label.setStyleSheet("color: white; font-size: 14px;")
        main_layout.addWidget(self.label, alignment=Qt.AlignHCenter)

        # Drawing canvas
        self.canvas = DrawingCanvas(size=320)
        c_layout = QHBoxLayout()
        # c_layout.addStretch()
        c_layout.addWidget(self.canvas)
        c_layout.addStretch()
        main_layout.addLayout(c_layout)

        # Buttons below
        b_layout = QHBoxLayout()
        # b_layout.addStretch()
        self.recognize_btn = QPushButton("Recognize")
        self.recognize_btn.clicked.connect(self.recognize_digit)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.canvas.clear)
        self.animate_check = QCheckBox("Animate activations")
        self.animate_check.setChecked(True)
        # Fix checkbox tick visibility on dark background
        self.animate_check.setStyleSheet("""
            QCheckBox {
                color: white;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QCheckBox::indicator:unchecked {
                border: 1px solid #888;
                background-color: #222;
            }
            QCheckBox::indicator:checked {
                border: 1px solid #00c853;
                background-color: #00c853;
            }
        """)

        for w in [self.recognize_btn, self.clear_btn, self.animate_check]:
            b_layout.addWidget(w)
        b_layout.addStretch()
        main_layout.addLayout(b_layout)

        main_layout.addStretch()
        self.setLayout(main_layout)

        # State
        self.activations = [np.zeros(n) for n in self.mlp.layer_sizes]
        self.phase = 0.0
        self.prediction = None

        self.timer = QTimer()
        self.timer.timeout.connect(self.animate)
        self.timer.start(50)
        self.setStyleSheet("background-color: #111;")

    def recognize_digit(self):
        try:
            x = self.canvas.get_flattened_input()
            out = self.mlp.forward(x)
            self.activations = [
                self.mlp.caches.get(f"A{i}", np.zeros(s)).flatten()
                for i, s in enumerate(self.mlp.layer_sizes)
            ]
            self.prediction = int(np.argmax(out))
            self.label.setText(f"Predicted (untrained): {self.prediction}")
            self.update()
        except Exception as e:
            self.label.setText(f"Error during recognition: {e}")

    def animate(self):
        if self.animate_check.isChecked():
            self.phase += 0.2
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        w, h = self.width(), self.height()
        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0, QColor(25, 25, 35))
        grad.setColorAt(1, QColor(10, 10, 20))
        painter.fillRect(self.rect(), grad)

        # Move network closer to the canvas (left side)
        start_x = self.canvas.width() + 20   # smaller gap between canvas and network
        avail_w = max(200, w - start_x - 20)

        n_layers = len(self.mlp.layer_sizes)
        h_space = avail_w / (n_layers + 1)
        layers = []
        for i, n_neurons in enumerate(self.mlp.layer_sizes):
            n_draw = min(n_neurons, 50)
            y_space = h / (n_draw + 1)
            off = (h - (n_draw * y_space)) / 2
            neurons = []
            r = max(6, int(min(avail_w, h) / (n_draw * 2.5 * n_layers)))
            for j in range(n_draw):
                x = start_x + (i + 1) * h_space
                y = off + (j + 1) * y_space
                neurons.append((x, y, r, j))
            layers.append(neurons)

        # Connections
        for i in range(len(layers) - 1):
            W = self.mlp.params[f"W{i + 1}"]
            for (x1, y1, _, j1) in layers[i]:
                for (x2, y2, _, j2) in layers[i + 1]:
                    if j1 >= W.shape[0] or j2 >= W.shape[1]:
                        continue
                    w_val = W[j1, j2]
                    if abs(w_val) < 0.05:
                        continue
                    c_strength = int(min(255, abs(w_val) * 255))
                    color = (QColor(c_strength, 0, 255 - c_strength)
                            if w_val > 0 else QColor(255 - c_strength, 0, c_strength))
                    painter.setPen(color)
                    painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        # Neurons
        painter.setFont(QFont("Arial", 9))
        for i, layer in enumerate(layers):
            for x, y, r, j in layer:
                a = 0
                if i < len(self.activations) and j < len(self.activations[i]):
                    a = float(self.activations[i][j])
                a_disp = np.clip(a, 0, 1)
                pulse = 1.0 + 0.15 * np.sin(self.phase + a * np.pi)
                radius = int(r * pulse)
                color_i = int(a_disp * 255)
                painter.setBrush(QColor(color_i, 40, 255 - color_i))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(int(x) - radius, int(y) - radius, 2 * radius, 2 * radius)

                if i == len(layers) - 1:
                    painter.setPen(QColor(255, 255, 255))
                    painter.drawText(int(x) + r + 4, int(y) + 4, f"{a_disp:.2f}")
                    if j < 10:
                        painter.setPen(QColor(200, 200, 100))
                        painter.drawText(int(x) + r + 40, int(y) + 4, str(j))
        painter.end()


# ==============================================================
# Run
# ==============================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    layer_sizes = [784, 128, 64, 32, 10]
    mlp = SimpleMLP(layer_sizes)

    # Load trained weights if available
    import os
    if os.path.exists("weights_mnist.npz"):
        data = np.load("weights_mnist.npz")
        for key in data.files:
            mlp.params[key] = data[key]
        print("✅ Loaded trained MNIST weights!")
    else:
        print("⚠️ No weights_mnist.npz found, using random weights.")

    win = NNVisualizer(mlp)
    win.show()
    sys.exit(app.exec())
