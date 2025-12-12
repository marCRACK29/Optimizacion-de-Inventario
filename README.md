# Optimización de Inventario 📦

> **Proyecto Final de Inteligencia Artificial** > Aplicación de técnicas de Aprendizaje por Refuerzo (Reinforcement Learning) para la gestión eficiente de inventarios.

## 📄 Descripción

Este proyecto aborda el problema de la **optimización de inventarios** utilizando algoritmos de Inteligencia Artificial. El objetivo principal es encontrar una política óptima de reposición que minimice los costos asociados (almacenamiento, pedidos y fallas de stock) mientras se maximiza el nivel de inventario.

El sistema utiliza algoritmos de **Reinforcement Learning (RL)** (como DQN) implementados con *Stable Baselines3* y *Gymnasium*. 

## 📊 Metodología y Datos

El proyecto se basa en el dataset `retail_store_inventory.csv`. Tras realizar un **Análisis Exploratorio de Datos (EDA)**, se tomaron las siguientes decisiones de diseño para el entrenamiento:

* **Enfoque en Producto Único (P0001):** Se detectó que el comportamiento de precios y demanda del producto `P0001` era consistente entre las diferentes tiendas.
* **Simulación de Historial Extendido:** Para reducir la complejidad y aumentar los datos disponibles para el agente, se filtraron los datos de `P0001` y se concatenaron las series temporales de las 5 tiendas (S001-S005). Esto genera una "super-tienda" con un historial secuencial extenso para el entrenamiento.
* **Variables de Estado (Observación):** El agente toma decisiones basándose en:
    * Nivel de Inventario   
    * Precio y Descuento
    * Precios de la competencia
    * Factores externos: Clima, Festivos y Estacionalidad.
    * Categoría del producto

## 👥 Autores

Desarrollado por:
* **Matías Figueroa**
* **Gabriel Castillo**
* **Daniel Támaro**
* **Marcos Martínez**

## 📂 Estructura del Proyecto

El repositorio está organizado de la siguiente manera:

| Carpeta/Archivo | Descripción |
|-----------------|-------------|
| `📂 configs/` | Archivos de configuración e hiperparámetros para el entrenamiento. |
| `📂 data/` | Conjuntos de datos utilizados para la simulación y validación. |
| `📂 notebooks/` | Jupyter Notebooks con análisis exploratorio, entrenamiento y experimentos. |
| `📂 src/` | Código fuente del entorno (`Gymnasium`) y lógica del negocio. |
| `📄 requirements.txt` | Lista de dependencias y librerías necesarias. |

## ⚙️ Instalación

Sigue estos pasos para configurar el entorno de desarrollo local:

1. **Clonar el repositorio:**
   ```bash
   git clone git@github.com:marCRACK29/Optimizacion-de-Inventario.git
   cd Optimizacion-de-Inventario
   ```

2. **Crear un entorno virtual:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Linux/Mac
   .\venv\Scripts\activate  # En Windows
   ```

3. **Instalar las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Uso

La ejecución principal del proyecto se realiza a través de de los cuadernos de Jupyter Notebooks. Para iniciar: 

1. **Abre tu editor de código:**
   ```bash
   # opciones
   code . # para Visual Studio Code
   antigravity . 
   cursor . 
   ```

2. **Navega a la carpeta `📂 notebooks/`.**

3. **Ejecuta el notebook `training_dqn.ipynb` para entrenar el agente.**

4. **Ejecutar los otros notebooks para analizar los resultados.**
    - `EDA.ipynb`: (Opcional) Visualiza la limpieza y preparación de los datos. Genera los archivos de entrenamiento.
    - `sanity_cheks.ipynb`
    - `benchmarking.ipynb` 
    - `train_test_split.ipynb`
    - `visualizar_episodio.ipynb`

## 🧠 Tecnologías Utilizadas

- **Python 3.12**
- **Stable Baselines3**: Algoritmos de RL.
- **Gymnasium (OpenAI Gym)**: Creación del entorno de simulación.
- **Pandas & NumPy**: Procesamiento de datos.
- **Matplotlib**: Visualización de resultados.