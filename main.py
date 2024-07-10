import streamlit as st
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf

# Завантаження моделей без компіляції
vgg16_model = load_model('best_model.h5', compile=False)
dense_model = load_model('saved_model_dence.h5', compile=False)

# Перекомпіляція dense моделі
dense_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Завантаження історії навчання
with open('training_history.pkl', 'rb') as file:
    history_vgg16 = pickle.load(file)

with open('training_history_dence.pkl', 'rb') as file:
    history_dense = pickle.load(file)

# Відображення опису моделі
def display_model_summary(model, model_name):
    st.subheader(f'Характеристики моделі {model_name}')
    model.summary(print_fn=lambda x: st.text(x))

# Відображення графіків історії навчання
def plot_history(history, model_name):
    st.subheader(f'Історія навчання моделі {model_name}')
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(history['loss'], label='Train Loss')
    ax[0].plot(history['val_loss'], label='Validation Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[0].set_title('Втрати')

    ax[1].plot(history['accuracy'], label='Train Accuracy')
    ax[1].plot(history['val_accuracy'], label='Validation Accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()
    ax[1].set_title('Точність')

    st.pyplot(fig)

# Препроцесінг для VGG16
def classify_image_vgg16(uploaded_file, model):
    target_size = (48, 48)  # Вказуємо розмір зображення без каналу
    img = load_img(uploaded_file, target_size=target_size)
    st.image(img, caption='Завантажене зображення для VGG16', use_column_width=True)

    img_array = img_to_array(img)  # Перетворення зображення в масив
    img_array = np.expand_dims(img_array, axis=0)  # Додавання нової осі для пакетного формату
    img_array = preprocess_input(img_array)  # Препроцесинг зображення для VGG16

    predictions = model.predict(img_array)  # Прогноз моделі
    predicted_class = np.argmax(predictions, axis=-1)[0]
    class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']  # Заміна назв класів на відповідні
    predicted_label = class_labels[predicted_class]
    predicted_probability = np.max(predictions) * 100
    return predicted_label, predicted_probability

# Препроцесінг для Dense
def classify_image_dense(uploaded_file, model):
    target_size = (28, 28)
    img = load_img(uploaded_file, color_mode="grayscale", target_size=target_size)
    st.image(img, caption='Завантажене зображення для Dense', use_column_width=True)

    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)  # Додавання каналу

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=-1)[0]
    class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']  # Заміна назв класів на відповідні
    predicted_label = class_labels[predicted_class]
    predicted_probability = np.max(predictions) * 100
    return predicted_label, predicted_probability

# Інтерфейс Streamlit
st.title('Порівняння моделей VGG16 та Dense')

# Відображення характеристик та історії навчання моделей
display_model_summary(vgg16_model, 'VGG16')
plot_history(history_vgg16, 'VGG16')

display_model_summary(dense_model, 'Dense')
plot_history(history_dense, 'Dense')

# Завантаження зображення для класифікації
st.subheader('Класифікація зображення')
uploaded_file = st.file_uploader("Завантажте зображення для класифікації", type=["jpg", "png"])

if uploaded_file is not None:
    st.subheader('Класифікація за допомогою VGG16')
    predicted_label_vgg16, predicted_probability_vgg16 = classify_image_vgg16(uploaded_file, vgg16_model)
    st.write(f"Прогнозований клас: {predicted_label_vgg16} ({predicted_probability_vgg16:.2f}%)")

    st.subheader('Класифікація за допомогою Dense')
    predicted_label_dense, predicted_probability_dense = classify_image_dense(uploaded_file, dense_model)
    st.write(f"Прогнозований клас: {predicted_label_dense} ({predicted_probability_dense:.2f}%)")