import tensorflow as tf
import numpy as np
import os

CLASSES = {0: "Clear", 1: "Crystal", 2: "Other", 3: "Precipitate"}

def testar_imagem(caminho_imagem, caminho_modelo):
    print("Carregando modelo...")
    model = tf.keras.models.load_model(caminho_modelo)

    print(f"Processando imagem: {caminho_imagem}")
    

    img_raw = tf.io.read_file(caminho_imagem)
    img = tf.image.decode_image(img_raw, channels=3)
    img = tf.image.resize(img, [224, 224])
    

    img_array = tf.cast(img, tf.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0) 


    previsoes = model.predict(img_array)
    
    indice_vencedor = np.argmax(previsoes[0])
    classe_vencedora = CLASSES[indice_vencedor]
    confianca = previsoes[0][indice_vencedor] * 100

  
    print("\n--- RESULTADO DA IA ---")
    print(f"A imagem foi classificada como: {classe_vencedora}")
    print(f"Nível de confiança: {confianca:.2f}%")
    print("Probabilidades brutas retornadas (Clear, Crystal, Other, Precipitate):")
    print(previsoes[0])

if __name__ == '__main__':

    testar_imagem('../images/amostra_001.jpg', 'marco_model.keras')