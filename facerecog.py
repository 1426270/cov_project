'''
import face_recognition

image = face_recognition.load_image_file("data/Fam4a/369538590_a47ad7a104_166_10375311@N00.jpg")
face_locations = face_recognition.face_locations(image)
print(face_locations)
'''

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton

e = 3

def buttonClick(button):
    print(button.text())
    global e
    e = 4

app = QApplication(sys.argv)
widget = QWidget()

button1 = QPushButton(widget)
button1.setText("Button1")
button1.move(64, 32)
button1.clicked.connect(lambda: buttonClick(button1))

button2 = QPushButton(widget)
button2.setText("Button2")
button2.move(64, 64)
button2.clicked.connect(lambda: buttonClick(button2))

widget.setGeometry(50, 50, 320, 200)
widget.setWindowTitle("PyQt5 Button Click Example")
widget.show()
sys.exit(app.exec_())


