import tensorflow as tf
from data_pipeline import load_dataset
from model import build_model
import os
import glob


TRAIN_FILES = glob.glob('data/train-*')
VAL_FILES = glob.glob('data/test-*')


print(f"Arquivos de treino encontrados: {len(TRAIN_FILES)}")
print(f"Arquivos de validação encontrados: {len(VAL_FILES)}")

if len(TRAIN_FILES) == 0:
    raise FileNotFoundError("Nenhum arquivo 'train-*' encontrado na pasta data/!")
def main():
    print("Carregando datasets...")
    train_ds = load_dataset(TRAIN_FILES)
    val_ds = load_dataset(VAL_FILES)

    print("Montando o modelo...")
    model = build_model()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Iniciando treinamento (isso pode levar um tempo na GPU)...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5 
    )


    model.save('marco_model.keras')
    print("Modelo salvo como 'marco_model.keras'!")

if __name__ == '__main__':
    main()