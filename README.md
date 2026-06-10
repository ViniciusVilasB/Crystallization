# FastCrystal - Visão Computacional para Monitoramento Espacial

## Descrição do Projeto

O **FastCrystal** é um sistema de monitoramento em tempo real voltado para experimentos de cristalização de proteínas em ambientes de microgravidade (como a Estação Espacial Internacional - ISS). O desenvolvimento de novos medicamentos depende da análise estrutural dessas proteínas, e o espaço oferece um ambiente ideal para o crescimento de cristais com menos imperfeições. 

Devido ao altíssimo custo de lançamento, limitações de carga e longo período de incubação, perder uma amostra representa um enorme prejuízo financeiro e científico. Para resolver este problema, desenvolvemos um pipeline de **Visão Computacional e Inteligência Artificial** que processa o feed de vídeo de webcams nas estações de experimento, monitorando continuamente o estado das gotas e classificando-as automaticamente.

## Objetivos

### Objetivo de Negócio
* **Redução de Desperdícios:** Evitar a perda de amostras caras enviadas ao espaço através de monitoramento proativo.
* **Automação e Otimização:** Reduzir a carga de trabalho dos astronautas e pesquisadores, automatizando a detecção visual de amostras promissoras.
* **Aceleração da Pesquisa:** Auxiliar a tomada de decisão para identificar rapidamente quais experimentos estão gerando cristais válidos para a formulação de novos medicamentos.

### Objetivo Técnico
* Desenvolver um script em Python capaz de capturar vídeo via webcam em tempo real.
* Processar os frames aplicando técnicas de Visão Computacional e uma Rede Neural Convolucional (TensorFlow/Keras) para classificar o estado da amostra em quatro categorias: `Clear`, `Crystal`, `Other`, e `Precipitate`.
* Garantir a resiliência do sistema lidando com intempéries do ambiente (ruídos de imagem, variações de iluminação) e tratando exceções de hardware (como queda de conexão da câmera).

---

## Arquitetura e Lógica da Solução

O funcionamento do nosso pipeline de Visão Computacional segue o diagrama lógico abaixo:

1. **Captura de Stream (OpenCV):** O sistema inicia a captura de vídeo utilizando a webcam. Foi implementado um bloco `try/except` para garantir que, caso a câmera desconecte ou falhe, o sistema tente a reconexão sem quebrar a aplicação (Tratamento de Exceções de Hardware).
2. **Pré-processamento (OpenCV/NumPy):** 
   * Captura do frame atual.
   * Redimensionamento da imagem para a resolução exigida pelo modelo (224x224 pixels).
   * Normalização dos tensores (conversão para `float32` e divisão por 255.0) para alinhar com os pesos do modelo treinado.
3. **Inferência (TensorFlow/Keras):** O frame pré-processado é passado para o modelo `marco_model.keras`. O modelo calcula as probabilidades brutas e retorna o índice de maior confiança.
4. **Pós-processamento e Feedback Visual:**
   * A classe vencedora (`Clear`, `Crystal`, `Other`, `Precipitate`) e sua porcentagem de confiança são extraídas.
   * O texto contendo a predição e o FPS (Frames Per Second) atual são sobrepostos no frame original usando `cv2.putText`.
   * O sistema exibe o resultado visualmente na tela em tempo real, demonstrando robustez contra oclusões parciais e variações de luz presentes no ambiente.

---

## Stack Tecnológica / Bibliotecas Utilizadas

* **Python 3.12**: Linguagem base do projeto.
* **OpenCV (`cv2`)**: Captura do stream de vídeo, manipulação de frames, cálculo de FPS e exibição do HUD visual.
* **TensorFlow / Keras**: Carregamento do modelo de Deep Learning e execução das inferências em tempo real.
* **NumPy**: Manipulação de matrizes, reestruturação de dimensões (`expand_dims`) e cálculos de probabilidade (`argmax`).

---

## Instruções de Execução (Setup de Ambiente)

Para garantir a reprodutibilidade exata do ambiente, siga os passos abaixo:

**1. Clone o repositório:**
```bash
git clone [https://github.com/SEU_USUARIO/FastCrystal-CV.git](https://github.com/SEU_USUARIO/FastCrystal-CV.git)
cd FastCrystal-CV
```

2. Crie e ative um ambiente virtual (Recomendado):
No Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

No Linux/Mac:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Instale as dependências:
Certifique-se de estar com o ambiente virtual ativado e instale os pacotes mapeados no arquivo de controle de dependências:
```bash
pip install -r requirements.txt
```

4. Arquivo de Modelo:
Verifique se o arquivo contendo os pesos treinados (marco_model.keras) está presente no diretório raiz do projeto ou na pasta especificada dentro do script.

5. Execução do Script Principal:
```bash
python main_webcam.py
```

## Integrantes do Grupo

*   Gabriel Luni Nakashima - RM558096
*   Gustavo Henrique - RM556712
*   Milena Garcia - RM555111
*   Renan Simões Gonçalves - RM555584
*   Vinicius Vilas Boas - RM557843

## Video

https://youtu.be/Vs8-cNZU1Kw

