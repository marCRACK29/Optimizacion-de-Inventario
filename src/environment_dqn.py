import gymnasium as gym
from gymnasium import spaces
from collections import deque
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
        self.season_max = 3.0

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

        # Variables de estado
        self.inventario = 0
        self.cola_pedidos = []
        self.history_inv = None
        self.ultimo_pedido_qty = 0 # guarda la cantidad del pedido anterior


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Selección aleatoria del inicio (respetando ventana + episodio)
        max_start = len(self.data) - self.duracion_simulacion - 1
        self.start_step = self.np_random.integers(self.ventana, max_start)
        
        self.current_step = self.start_step
        self.steps = 0

        # Estado inicial
        self.inventario = 200   # punto inicial 
        self.cola_pedidos = [0] * self.tiempo_entrega
        self.ultimo_pedido_qty = 0

        # Inicializar historial de inventario
        # Llenamos el historial pasado con el valor inicial para no romper la ventana
        self.history_inv = deque([self.inventario] * self.ventana, maxlen=self.ventana)
        
        obs = self._get_window_observation()
        return obs, {}

    def step(self, action):
        # 1. Acción seleccionada (cantidad pedida hoy)
        pedido_qty = self.action_values[action]

        # 2. Arriban pedidos previos (lead time = 2 días)
        llegando = self.cola_pedidos.pop(0)
        self.inventario = min(self.inventario + llegando, self.max_inventario)

        # 3. Registrar el pedido de hoy (llegará en 2 días)
        self.cola_pedidos.append(pedido_qty)

        # 4. Demanda real del día y ventas efectivas
        demanda = self.data[self.current_step, IDX_UNITS_SOLD]
        vendido = min(self.inventario, demanda)
        self.inventario -= vendido

        # Guardar inventario del día para la ventana histórica
        self.history_inv.append(self.inventario)

        # 5. Recompensa económica base
        precio = self.data[self.current_step, IDX_PRICE]
        unit_cost = 0.15 * precio   # costo de reposición reducido para mejor estabilidad

        ingresos = vendido * precio
        costo_pedido = pedido_qty * unit_cost
        costo_guardado = self.inventario * 0.02
        costo_stockout = (demanda - vendido) * precio

        # 6. Recompensas adicionales por comportamiento
        self.ultimo_pedido_qty = pedido_qty
   
        beneficio = ingresos - costo_pedido - costo_guardado - costo_stockout

        # Escala simple
        recompensa_normalizada = beneficio / 2000.0 

        recompensa_normalizada = np.clip(recompensa_normalizada, -1.0, 1.0)

        # 8. Avanzar el ambiente
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
            "raw_reward": beneficio,
        }

        return obs, recompensa_normalizada, terminated, truncated, info


    def _normalize_row(self, row, inv_historico):
        """Normaliza una fila de features."""
        return np.array([
            inv_historico / self.inv_max,
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
            # i=0 es el día más antiguo, i=ventana-1 es hoy
            
            # Índice en el dataframe global
            # Si hoy es step 100 y ventana es 7:
            # i=0 -> step 94
            # i=6 -> step 100 (hoy)
            row_idx = self.current_step - (self.ventana - 1 - i)
            row = self.data[row_idx]
            
            # Dato del inventario simulado (recuperado del deque)
            inv_val = self.history_inv[i]
            window_rows.append(self._normalize_row(row, inv_val))
        
        # Aplanar todo en un solo vector 1D
        return np.concatenate(window_rows, axis=0)
