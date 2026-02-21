# Detecção de Pneumonia (Classificação de Imagens Médicas).

Este projeto apresenta uma solução construída em PyTorch para a classificação de radiografias de tórax em duas categorias: **Normal** e **Pneumonia**. 

# Objetivo:
- Classificar o diagnóstico de imagens de raio-x em uma classificação binária(Normal ou Pneumonia) e transformar em uma distribuição probabilística (função SoftMax).

- Comparar o desempenho de diferentes arquiteturas avançadas de visão computacional (CNNs vs. Transformers).

  
- Garantir a interpretabilidade das decisões do modelo utilizando mapas de ativação de classe (Grad-CAM).

# Base de Dados
- Fonte: <a href="https://www.kaggle.com/competitions/ligia-compviz/overview">Kaggle – Lígia - CV
  
- Distribuição dos dados:
  - Conjunto de Treinamento e Validação: 5232 imagens (1349 Normal, 3883 Pneumonia).
  - Conjunto de Teste: 624 imagens.

- Pré-processamento e Augmentation:
  - Redimensionamento para 224x224 pixels.
  - Normalização utilizando médias e desvios-padrão do ImageNet.
  - Transformações sintéticas (RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter) para mitigar *overfitting*.

## Estrutura do Projeto

```text

  ├── 📂 Código dos modelos/            # Implementações e pesos das arquiteturas
  │   ├── 📂 grad_cam/
  │   │   └── 📓 gradcam.ipynb          # Notebook de explicabilidade (Grad-CAM)
  │   ├── 📂 resnet50/
  │   │   ├── 📓 resnet50.ipynb         # Treinamento e avaliação ResNet
  │   │   ├── 📓 gerar_csv_resnet.ipynb # Script para gerar submissão Kaggle
  │   │   └── 📄 modelo_colab_resnet50.pth
  │   └── 📂 vision_transformers/
  │       ├── 📓 vision_transformers.ipynb # Treinamento do modelo ViT
  │       ├── 📓 gerar_csv_vision_transformers.ipynb
  │       └── 📄 modelo_colab_vision_transformers.pth
  │
  ├── 📂 dataset/                       # Base de dados (Não versionada - 1.2 GB)
  │   ├── 📂 train/                     # Imagens rotuladas (NORMAL/PNEUMONIA)
  │   ├── 📂 test_images/               # Imagens de teste sem rótulo
  │   ├── 📄 train.csv
  │   └── 📄 test.csv
  │
  ├── 📂 gradcam_results/               # Resultados dos mapas de calor
  ├── 📂 graficos resnet50/             # Métricas visuais do modelo ResNet
  ├── 📂 Gráficos vision transformers/  # Métricas visuais do modelo ViT
  │
  ├── 📄 .gitignore                     # Configurado para ignorar venv e dataset
  ├── 📄 README.md                      # Documentação do projeto
  └── 📄 requirements.txt               # Dependências do ambiente
```
## 💻Programador:

<table align="center">
  <tr>
    <td align="center">
      <a href="https://github.com/arthlz">
        <img src="https://avatars.githubusercontent.com/u/173482833?v=4" width="120px;" alt="Arthur Luz"/><br>
        <sub><b>Arthur Luz</b></sub>
      </a>
    </td>
  </tr>
</table>

## Tecnologias Utilizadas:
<div align="left">
<img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white" />
<img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" />
<img src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white" />
<img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white" />
<img src="https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black" />
</div>
