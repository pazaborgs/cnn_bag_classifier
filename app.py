import io
import numpy as np
import pandas as pd
import PIL
import plotly.express as px
import streamlit as st
import tensorflow as tf
from huggingface_hub import hf_hub_download


@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="pazaborgs/bag_classifier",
        filename="bag_class_optimized.tflite",
        repo_type="model"
    )
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def load_image():
    uploaded_file = st.file_uploader(
        'Arraste e solte uma imagem aqui ou clique para selecionar uma',
        type=['png', 'jpg', 'jpeg']
    )

    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image = PIL.Image.open(io.BytesIO(image_data)).convert('RGB')
        st.image(image, caption='Imagem carregada', width=300)

        image = image.resize((300, 300))  # Resize para o tamanho esperado pelo modelo
        image = np.array(image, dtype=np.float32) / 255.0
        image = np.expand_dims(image, axis=0)

        return image


def prev(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    classes = ['Garbage Bag', 'Paper Bag', 'Plastic Bag']
    df = pd.DataFrame({
        'classes': classes,
        'probabilidades (%)': 100 * output_data[0]
    })

    fig = px.bar(
        df,
        y='classes',
        x='probabilidades (%)',
        orientation='h',
        text='probabilidades (%)',
        title='Probabilidade por tipo de sacola'
    )
    st.plotly_chart(fig)


def main():
    st.set_page_config(
        page_title='Classificador de Sacolas',
        page_icon='🛍️',
    )

    st.write('# Classificador de Sacolas para Reciclagem Inteligente 🛍️')

    interpreter = load_model()
    image = load_image()

    if image is not None:
        prev(interpreter, image)


if __name__ == '__main__':
    main()
