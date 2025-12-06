import gymnasium as gym
from gymnasium import spaces
import numpy as np


IDX_INVENTORY_LEVEL = 0
IDX_FORECAST = 1
IDX_PRICE = 2
IDX_DISCOUNT = 3
IDX_HOLIDAY = 4
IDX_COMPETITOR = 5
IDX_WEATHER = 6
IDX_SEASONALITY = 7
IDX_UNITS_SOLD = 8 # Target, no lo puede ver el agente


class RetailEnvDQN(gym.Env):
    """Entorno optimizado para DQN en control de inventarios."""

    def __init__(self, data_array, ventana=7, duracion_simulacion=90):
        super().__init__()

        self.data = data_array
        self.ventana = ventana # 7 días, es decir, el agente ve el día actual y 6 días atrás
        self.duracion_simulacion = duracion_simulacion # 90 días de gestión del inventario

        # Normalización basada en el dataset
        self.inv_max = 800.0
        self.forecast_max = 400.0
        self.price_max = 100.0
        self.discount_max = 50.0
        self.competitor_max = 100.0
        self.weather_max = 3.0
        self.season_max = 105.0

        # Acción = unidades pedidas hoy
        # Son las unidades que puede pedir (hasta 800 unidades)
        self.action_values = np.array([0, 50, 100, 200, 300, 500, 800], dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.action_values))

        # Observación = ventana (ventana × 8 features normalizadas)
        self.obs_dim = 8 * ventana
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )

        # Parámetros logísticos
        self.max_inventario = 1500 # tope físico de la bodega
        self.tiempo_entrega = 2 # los días que tarda en llegar el pedido
        self.cola_pedidos = None # cola de pedidos en camino 

        # Para penalizar oscilaciones
        self.ultimo_pedido_qty = 0 # guarda la cantidad del pedido anterior


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Selección aleatoria del inicio (respetando ventana + episodio)
        max_start = len(self.data) - (self.duracion_simulacion + self.ventana) - 1
        self.start_step = self.np_random.integers(self.ventana, max_start)
        
        self.current_step = self.start_step
        self.steps = 0

        # Estado inicial
        self.inventario = 200   # punto inicial 
        self.cola_pedidos = [0] * self.tiempo_entrega
        self.ultimo_pedido_qty = 0

        obs = self._get_window_observation()
        return obs, {}


    def step(self, action):
        pedido_qty = self.action_values[action]

        # 1. Arriban pedidos previos
        llegando = self.cola_pedidos.pop(0)
        self.inventario = min(self.inventario + llegando, self.max_inventario)

        # 2. Colocar el pedido de hoy
        self.cola_pedidos.append(pedido_qty)

        # 3. Demanda real hoy
        demanda = self.data[self.current_step, IDX_UNITS_SOLD]
        vendido = min(self.inventario, demanda)
        self.inventario -= vendido

        # 4. Recompensa económica
        precio = self.data[self.current_step, IDX_PRICE]
        unit_cost = 0.4 * precio  # puedes cambiarlo si estimas un costo fijo real

        ingresos = vendido * precio
        costo_pedido = pedido_qty * unit_cost
        costo_guardado = self.inventario * 0.05
        costo_stockout = (demanda - vendido) * precio * 0.3

        # Penalizar inventario excesivo
        exceso = max(0, self.inventario - self.max_inventario * 0.7) * 0.1

        # Penalizar cambios abruptos en pedidos
        penalizacion_suave = abs(pedido_qty - self.ultimo_pedido_qty) * 0.01
        self.ultimo_pedido_qty = pedido_qty

        recompensa = ingresos - costo_pedido - costo_guardado - costo_stockout - exceso - penalizacion_suave

        # Avanzar
        self.current_step += 1
        self.steps += 1

        terminated = self.steps >= self.duracion_simulacion
        truncated = False

        obs = self._get_window_observation()

        info = {
            "vendido": vendido,
            "demanda": demanda,
            "inventario": self.inventario,
            "pedido_qty": pedido_qty,
            "raw_reward": recompensa,
        }

        return obs, recompensa, terminated, truncated, info


    def _normalize_row(self, row):
        """Normaliza una fila de features."""
        return np.array([
            self.inventario / self.inv_max,
            row[IDX_FORECAST] / self.forecast_max,
            row[IDX_PRICE] / self.price_max,
            row[IDX_DISCOUNT] / self.discount_max,
            row[IDX_HOLIDAY],
            row[IDX_COMPETITOR] / self.competitor_max,
            row[IDX_WEATHER] / self.weather_max,
            row[IDX_SEASONALITY] / self.season_max
        ], dtype=np.float32)

    def _get_window_observation(self):
        """Devuelve una ventana completa de N días como un vector 1D."""
        window_rows = []
        for i in range(self.ventana):
            row_idx = self.current_step - (self.ventana - 1 - i)
            row = self.data[row_idx]
            window_rows.append(self._normalize_row(row))

        return np.concatenate(window_rows, axis=0)
