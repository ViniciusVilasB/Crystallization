import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from fastapi import FastAPI, HTTPException
import mysql.connector


# 1. Configurações e Carregamento da IA
CLASSES = {0: "Clear", 1: "Crystal", 2: "Other", 3: "Precipitate"}

# Descobre a pasta exata onde o main.py está e junta com o nome do modelo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'marco_model.keras')

print("Iniciando API e carregando modelo da IA (isso pode levar alguns segundos)...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Modelo carregado com sucesso!")


# 2. Inicializando o FastAPI
app = FastAPI(title="Crystal Prediction API")


# 3. Função de Conexão com o Banco
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",  # Altere se a sua senha for diferente
        database="crystaldata"
    )


# 4. Rota GET para buscar no banco e prever
@app.get("/predict/{sample_id}")
def prever_amostra_do_banco(sample_id: int):
    # PASSO A: Busca o NOME da imagem no MySQL usando o ID
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT image_path FROM samples WHERE id = %s", (sample_id,))
        amostra = cursor.fetchone()
        cursor.close()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao conectar no banco: {e}")

    if not amostra:
        raise HTTPException(status_code=404, detail="Amostra não encontrada no banco de dados.")

    # PASSO B: Monta o caminho da imagem
    nome_arquivo = amostra['image_path']
    DIR_ATUAL = os.path.dirname(os.path.abspath(__file__))
    PASTA_IMAGENS = os.path.abspath(os.path.join(DIR_ATUAL, '..', 'images'))
    caminho_imagem = os.path.join(PASTA_IMAGENS, nome_arquivo)

    if not os.path.exists(caminho_imagem):
        raise HTTPException(
            status_code=400,
            detail=f"O Python procurou em: {caminho_imagem}, mas a imagem '{nome_arquivo}' não existe."
        )

    # PASSO D: Roda a imagem na IA
    try:
        img = image.load_img(caminho_imagem, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        previsoes = model.predict(img_array)

        indice_vencedor = int(np.argmax(previsoes[0]))
        classe_vencedora = CLASSES[indice_vencedor]
        confianca = float(previsoes[0][indice_vencedor] * 100)

        # PASSO E: Retorna o resultado
        return {
            "id_buscado": sample_id,
            "arquivo": nome_arquivo,
            "resultado": {
                "classificacao": classe_vencedora,
                "confianca_percentual": round(confianca, 2),
                "probabilidades_brutas": previsoes[0].tolist()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar imagem na IA: {e}")