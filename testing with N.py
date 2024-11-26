import pygame
import cv2
import numpy as np
from tkinter import filedialog, Tk
from tensorflow.keras.models import load_model
from PIL import Image
from glob import glob
from tensorflow.keras.preprocessing.image import img_to_array
from pygame.locals import QUIT


# Pygame
pygame.init()

WIDTH, HEIGHT = 1100, 720
BG_COLOR = (255, 255, 255)
TEXT_COLOR = (0, 0, 0)
BUTTON_COLOR = (70, 130, 180)
BUTTON_HOVER_COLOR = (100, 149, 237)
FONT = pygame.font.Font(None, 36)
HEADER_FONT = pygame.font.Font(None, 50)

# screensetup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("ECG Arrhythmia Detection")

# background
bg_image = pygame.image.load("wave.jpg")
bg_image = pygame.transform.scale(bg_image, (WIDTH, HEIGHT))


def render_text(text, pos, font, color=TEXT_COLOR):
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, pos)


def button(x, y, w, h, text, action=None):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()

    if x < mouse[0] < x + w and y < mouse[1] < y + h:
        pygame.draw.rect(screen, BUTTON_HOVER_COLOR, (x, y, w, h))
        if click[0] == 1 and action is not None:
            action()
    else:
        pygame.draw.rect(screen, BUTTON_COLOR, (x, y, w, h))

    text_surface = FONT.render(text, True, (255, 255, 255))
    screen.blit(text_surface, (x + (w - text_surface.get_width()) // 2, y + (h - text_surface.get_height()) // 2))


def upload_and_analyze():
    global analyzed_image, prediction_text

    # File dialog to choose a file
    file_paths = filedialog.askopenfilenames(title='Select ECG Signal', filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    
    # Check if a file is selected
    if not file_paths:
        prediction_text = "No file selected."
        return

    # Use the first selected file
    file_path = file_paths[0]

    # Read the image with OpenCV
    img = cv2.imread(file_path)
    if img is None:
        prediction_text = "Unable to read the selected file."
        return

    # Resize and process the image
    img = cv2.resize(img, (224, 224))
    analyzed_image = pygame.image.fromstring(img.tobytes(), img.shape[1::-1], 'RGB')

    try:
        model = load_model('trained_model_DNN1.h5')
        test_tensor = img_to_array(Image.fromarray(img)) / 255.0
        test_tensor = np.expand_dims(test_tensor, axis=0)
        pred = model.predict(test_tensor)
        predicted_class_index = np.argmax(pred)
        findex = pred[0][predicted_class_index]
        print('score:', findex)

        classes = [item[10:-1] for item in sorted(glob("./dataset/*/"))]
        if predicted_class_index < len(classes):
            predicted_class = classes[predicted_class_index]
            prediction_text = f"Predicted Class"

            #'ab.txt'
            with open('ab.txt', 'r') as file:
                lines = file.readlines()
                for line in lines:
                    columns = line.strip().split()  
                    if columns[0] == predicted_class:  
                        prediction_text += f"\n: {line.strip()}"
                        break
        else:
            prediction_text = "Prediction out of range."
    except Exception as e:
        prediction_text = f"Error in prediction: {str(e)}"
    file_dialog_open = False


# Initialize variables
analyzed_image = None
prediction_text = ""
file_dialog_open = False

# Initialize Tkinter root window only once
root = Tk()
root.withdraw()

# Main loop
running = True
while running:
    screen.blit(bg_image, (0, 0))

    render_text("ECG Arrhythmia Detection", (WIDTH // 2 - 200, 20), HEADER_FONT, TEXT_COLOR)

    if analyzed_image:
        screen.blit(analyzed_image, (WIDTH // 2 - 112, 250))

    if prediction_text:
        render_text(prediction_text, (50, HEIGHT - 100), FONT, TEXT_COLOR)

    button(WIDTH // 2 - 100, HEIGHT - 200, 200, 50, "Upload Signal", upload_and_analyze)
    
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

    pygame.display.update()

pygame.quit()
