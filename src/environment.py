import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

IDX_INVENTORY_LEVEL = 0 
IDX_FORECAST = 1
IDX_PRICE = 2
IDX_DISCOUNT = 3
IDX_HOLIDAY = 4
IDX_COMPETITOR = 5
IDX_WEATHER = 6
IDX_SEASONALITY = 7
IDX_UNITS_SOLD = 8  # Esta columna no la puede ver el agente (target)

class RetailEnv(gym.Env):
    def __init__(self, data_array):
        super(RetailEnv, self).__init__()
        
        self.data = data_array # dataset como arreglo 
        self.current_step = 0
        
        # Define qué acciones puede tomar el agente
        # 0: Nada, 1: Pequeño (10 uds), 2: Mediano (30 uds), 3: Grande (50 uds)
        self.action_mapping = {0: 0, 1: 10, 2: 30, 3: 50} 
        self.action_space = spaces.Discrete(4) 
        
        # Define qué puede ver el agente
        # Qué ve el agente: [Inventario Actual, Pronóstico Demanda, Precio, Descuento, ...]
        self.observation_space = spaces.Box(
            # 8 números positivos decimales en cada paso
            low=0, high=np.inf, shape=(8,), dtype=np.float32
        )
        
        # Estado inicial del inventario
        self.inventory = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0 # resetea al primer día
        self.inventory = 50 # Inventario inicial arbitrario
        
        # Obtener la primera fila de datos
        obs = self._get_observation()
        return obs, {} # con ello el agente puede hacer su primera decisión

    def step(self, action):
        # 1. Aplicar la acción (Pedir inventario)
        order_qty = self.action_mapping[action]
        self.inventory += order_qty
        
        # 2. Simular la demanda del día (Usando datos históricos)
        actual_demand = self.data[self.current_step, IDX_UNITS_SOLD]
        
        # 3. Calcular ventas y nuevo inventario
        sold_qty = min(self.inventory, actual_demand) # no se puede vender más de lo que se tiene
        self.inventory -= sold_qty
        
        # 4. CALCULAR RECOMPENSA 
        price = self.data[self.current_step, IDX_PRICE]
        
        revenue = sold_qty * price # ingresos
        cost_of_goods = order_qty * (price * 0.4) # Ej: Asumir costo es 40% del precio
        holding_cost = self.inventory * 0.1 # Costo por guardar lo que sobró
        lost_sales_penalty = (actual_demand - sold_qty) * 2 # Penalización por no tener stock
        
        reward = revenue - cost_of_goods - holding_cost - lost_sales_penalty
        
        # 5. Avanzar al siguiente día
        self.current_step += 1
        terminated = self.current_step >= self.data.shape[0] - 1 # termina cuando se acaben los datos
        
        obs = self._get_observation()
        info = {'sold': sold_qty, 'demand': actual_demand}
        
        return obs, reward, terminated, False, info

    def _get_observation(self):
        """ Obtiene la observación actual del entorno, es un vector de 8 números decimales """
        row = self.data[self.current_step] # fila actual
        # Selecciona las columnas clave para que el agente decida
        return np.array([
            self.inventory, # inventario actual
            row[IDX_FORECAST], # pronóstico de demanda
            row[IDX_PRICE], # precio
            row[IDX_DISCOUNT], # descuento
            row[IDX_HOLIDAY], # holiday/promotion
            row[IDX_COMPETITOR], # precio competidor
            row[IDX_WEATHER], # condiciones climáticas
            row[IDX_SEASONALITY] # temporada
        ], dtype=np.float32)