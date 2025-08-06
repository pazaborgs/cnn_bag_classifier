# Classificador de Sacolas para Reciclagem Inteligente 🛍️

Este projeto implementa um classificador de imagens de sacolas plásticas, de papel e de lixo, utilizando um modelo TensorFlow Lite otimizado para inferência rápida.

A interface foi construída no Streamlit Cloud, permitindo o uso diretamente no navegador e facilitando a visualização das previsões do modelo.

O modelo foi treinado utilizando o dataset *Plastic - Paper - Garbage Bag Synthetic Images*, criado pelo usuário **vencerlanz09** na plataforma Kaggle. Disponível em:

[Dataset do Projeto](https://www.kaggle.com/datasets/vencerlanz09/plastic-paper-garbage-bag-synthetic-images)

## Objetivo

O principal objetivo é criar uma aplicação prática e leve para classificar tipos de sacolas a partir de imagens enviadas pelo usuário. O projeto visa não apenas demonstrar as capacidades dessa tecnologia, mas também fomentar caminhos para possíveis aplicações no âmbito da reciclagem inteligente.

## Dataset

O dataset *Plastic - Paper - Garbage Bag Synthetic Images* é uma coleção sintética de imagens geradas para três classes de sacolas:

- Plastic Bag (sacolas plásticas)  
- Paper Bag (sacolas de papel)  
- Garbage Bag (sacolas de lixo)

Características principais:

- Aproximadamente 5.000 imagens por classe, totalizando cerca de 15.000 imagens  
- Imagens com tamanho 300 x 300 pixels e 3 canais de cor (RGB)  
- Formato JPEG (.jpg)

## Como usar

Você pode acessar a aplicação online no Streamlit Cloud:

👉 [Link para acessar a aplicação](https://cnnbagclassifier-fsauitjhxdpkyvapdpavhn.streamlit.app/)


## Notebook do Colab

O notebook `cnn_bag_classifier.ipynb` está disponível no repositório para quem quiser reproduzir o treinamento, explorar o dataset e entender o processo por trás do modelo.

---

# Como Rodar o App Localmente

Siga os passos abaixo para executar a aplicação no seu computador:

Clone este repositório:

```bash
git clone https://github.com/pazaborgs/cnn_bag_classifier.git
cd seurepositorio
```

Cria um ambiente virtual (Windows):

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute o app streamlit:

```bash
streamlit run app.py
```


