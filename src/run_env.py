"""
Este script ejecuta una simulación básica del entorno RetailEnv.
Utiliza un agente aleatorio para demostrar cómo interactuar con el entorno,
cómo cargar los datos y cómo funciona el bucle principal de simulación.
"""
import pandas as pd
from environment import RetailEnv

def main():
    # 1. Cargar el dataset
    print("Cargando datos...")
    df = pd.read_csv('../data/data_train.csv')

    datos_numpy = df.values
    
    # 2. Inicializar el entorno
    print("Inicializando entorno...")
    env = RetailEnv(datos_numpy)
    
    # 3. Resetear el entorno para empezar un nuevo episodio
    obs, info = env.reset()
    print(f"Estado inicial: {obs}")
    
    done = False
    total_reward = 0
    steps = 0
    
    print("\nComenzando simulación...")
    while not done:
        # 4. Elegir una acción (Aquí usamos una aleatoria como ejemplo)
        # 0: Nada, 1: Pequeño, 2: Mediano, 3: Grande
        action = env.action_space.sample()
        
        # 5. Ejecutar el paso
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        steps += 1
        
        # Mostrar progreso cada 50 pasos
        if steps % 50 == 0:
            print(f"Paso {steps}: Inventario={obs[0]:.2f}, Recompensa={reward:.2f}")
            
        done = terminated or truncated

    print(f"\nSimulación terminada en {steps} pasos.")
    print(f"Recompensa Total: {total_reward:.2f}")

if __name__ == "__main__":
    main()
