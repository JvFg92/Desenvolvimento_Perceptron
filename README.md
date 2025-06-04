# 🧠 Perceptron Acelerado por C para Classificação Binária ⚙️

Este projeto implementa um algoritmo Perceptron para classificação binária, com suas funções de cálculo principais escritas em C para otimização de desempenho e envolvidas em uma classe Python para fácil uso, manipulação de dados e visualização. 📊 O projeto inclui funcionalidades para usar o conjunto de dados Iris (convertido para um problema binário) ou gerar dados sintéticos para treinamento e teste.

🎯 **Objetivo:** Criar um classificador Perceptron eficiente com um backend em C e uma interface Python amigável.

✨ **Visualização de Exemplo:**
<p align="center">
  <img src="https://github.com/user-attachments/assets/23f1dddd-ba94-48d6-bdd1-3421bb57614e" alt="Exemplo de Limite de Decisão" width="600"/>
  </p>

## 🌟 Visão Geral

O Perceptron é um dos algoritmos de aprendizado de máquina supervisionado mais simples para classificação binária. Este projeto demonstra:
* Implementação do algoritmo Perceptron.
* Uso de C para as operações computacionalmente intensivas (cálculo do neurônio, ajuste de pesos, avaliação de precisão) via `ctypes` em Python.
* Uma classe Python `Perceptron` que encapsula a lógica de treinamento, previsão, avaliação e plotagem.
* Carregamento e pré-processamento de dados para o conjunto de dados Iris e geração de dados sintéticos. 🌸
* Divisão de dados em conjuntos de treinamento e teste.
* Dimensionamento de características (normalização Z-score).
* Treinamento com uma taxa de aprendizado definida e um limite de precisão.
* Cálculo de erro, acuracia e revocação. 📈
* Validação cruzada K-fold.
* Visualização de:
    * Dados de treinamento e teste 📍
    * Limite de decisão do modelo treinado 🗺️
    * Precisão do modelo ao longo das épocas 🎯
    * Evolução dos pesos durante o treinamento 🏋️
    * Erro do modelo ao longo das épocas 📉

## ✨ Funcionalidades Principais

* **Núcleo em C ⚙️:** Funções `neuron`, `fit`, `evaluate_accuracy`, `predict` e `recall` implementadas em C para eficiência.
* **Wrapper Python 🐍:** Classe `Perceptron` em Python fácil de usar.
* **Fontes de Dados 💾:**
    * Utiliza o conjunto de dados Iris (filtrado para duas classes e duas características).
    * Gera dados sintéticos para problemas de classificação (linearmente separáveis ou com ruído).
* **Pré-processamento 🧹:**
    * Converte problemas multiclasse para binários.
    * Dimensiona características usando a média e o desvio padrão.
* **Treinamento 🏋️‍♀️:**
    * Itera até que uma precisão de referência seja atingida no conjunto de teste ou um número máximo de épocas seja alcançado.
    * Armazena o histórico de pesos, erros e precisão.
* **Avaliação 📊:**
    * Calcula a precisão nos conjuntos de treinamento e teste.
    * Realiza validação cruzada k-fold.
* **Visualização 🖼️:** Utiliza `matplotlib` para plotar:
    * Distribuição de dados.
    * Limite de decisão.
    * Curvas de aprendizado (precisão, erro, pesos).
* **Flexibilidade 🛠️:** Permite a configuração da taxa de aprendizado, precisão de referência e parâmetros de geração de dados.

## 📂 Estrutura do Projeto

├── 📄 perceptron.c        # Implementação em C das funções principais do Perceptron

├── 📄 perceptron.h        # Arquivo de cabeçalho para o código C

├── 🔗 perceptron.so       # Biblioteca compartilhada compilada (gerada após a compilação)

├── 🐍 data_treatment.py   # Funções para importação de dados, geração e plotagem

├── 🐍 training.py         # Classe Perceptron em Python e interface ctypes para C

└── 📖 main.py             # Script principal para executar o treinamento e avaliação

## 🛠️ Pré-requisitos

* Python 3.12.3 🐍
* Compilador C (como GCC) ⚙️
* Bibliotecas Python:
    * `numpy`
    * `matplotlib`
    * `scikit-learn` (usado em `data_treatment.py` para `load_iris` e `make_classification`)

## 🚀 Configuração e Instalação

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/JvFg92/Perceptron_Data_Classify
    cd Perceptron_Data_Classify
    ```

2.  **Compile o código C para criar a biblioteca compartilhada (`perceptron.so`):**
    No Linux ou macOS:
    ```bash
    gcc -shared -o perceptron.so -fPIC perceptron.c
    ```
    No Windows (pode exigir ajustes dependendo do seu compilador, por exemplo, com MinGW):
    ```bash
    gcc -shared -o perceptron.so perceptron.c -Wl,--add-stdcall-alias
    ```
    ℹ️ Certifique-se de que o arquivo `perceptron.so` (ou `perceptron.dll` no Windows) resultante esteja no mesmo diretório que os scripts Python.

3.  **Instale as dependências Python:**
     ```bash
    pip install numpy matplotlib scikit-learn
    ```
    
    ⚠️ Para Linux pode ser necessário:
    ```bash
    sudo apt install python3-numpy
    sudo apt install python3-matplotlib
    sudo apt install python3-sklearn
    ```
    
    ⚠️ Para Windows pode ser necessário:
    ```bash
    py -m pip install numpy matplotlib scikit-learn
    ```
    
    ✅ Pronto para começar!

## ▶️ Uso

O script principal para executar o modelo é `main.py`.
```bash
python main.py
```

⚠️ Para Linux pode ser necessário:
```bash
python3 main.py
```
⚠️ Para Windows pode ser necessário:
```bash
py main.py
```
