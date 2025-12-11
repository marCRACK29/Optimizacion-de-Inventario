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
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        # Parámetros logísticos
        self.max_inventario = 1500 # tope físico de la bodega
        self.tiempo_entrega = 2 # los días que tarda en llegar el pedido

        # Variables de estado
        self.inventario = 0
        self.cola_pedidos = None
        self.history_inv = None
        self.ultimo_pedido_qty = 0 # guarda la cantidad del pedido anterior

        # Estadísticas en tiempo real
        # Utilizadas para el cálculo de la media y varianza 
        # según el algoritmo de Welford

        # Rastrean el promedio y la varianza del nivel de inventario histórico
        self.inv_mean = 200.0
        self.inv_M2 = 1.0
        self.inv_count = 1e-4

        # Rastrean el promedio y la varianza de la cantidad de unidades pedidas
        self.order_mean = 0.0
        self.order_M2 = 1.0
        self.order_count = 1e-4
    
        # Rastrean el promedio y la varianza de la tasa de satisfacción
        self.fill_mean = 0.5
        self.fill_M2 = 1e-4
        self.fill_count = 1e-4

        # Coeficientes base (ajustables)
        # Son los costos por defecto de la función de recompensa 
        # antes de cualquier ajuste dinámico
        self.base_hold = 0.40           # costo de mantenimiento por unidad en inventario
        self.base_order_pen = 0.0005    # penalización cuadrática por cantidad de pedido
        self.base_smooth = 0.5          # penalización por cambio de pedido respecto al día anterior

        # Ganancias de adaptación (ajustables)
        # Son los "sensores" que deciden cuánto modificar los costos base
        # dependiendo de la situación actual.
        self.k_hold = 0.6 # más inventario que el objetivo -> aumenta castigo por mantener inventario
        self.k_order = 2.0 # alta volatilidad en pedidos -> aumenta castigo por pedir
        self.k_smooth = 1.0 # cambios bruscos en pedidos -> aumenta castigo por cambios de pedido

        # Límites de seguridad
        self.hold_clip = (0.10, 1.2)    # min, max multiplier on base_hold
        self.order_clip = (0.2, 5.0)    # multiplier on base_order_pen
        self.smooth_clip = (0.1, 5.0)   # multiplier on base_smooth



    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Modo evaluación (Test)
        # Si el entorno fue creado con disable_random_reset=True
        # entonces el episodio SIEMPRE empieza al inicio del dataset
        if getattr(self, "disable_random_reset", False):
            self.start_step = getattr(self, "fixed_start", 0)

        # Modo entrenamiento (Train)
        else:
            # Selección aleatoria del inicio (respetando ventana + duración)
            max_start = len(self.data) - self.duracion_simulacion - 1
            self.start_step = self.np_random.integers(self.ventana, max_start)

        self.current_step = self.start_step
        self.steps = 0

        # Estado inicial
        self.inventario = 200
        self.cola_pedidos = [0] * self.tiempo_entrega
        self.ultimo_pedido_qty = 0

        # Historial de inventario para la ventana
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
        unit_cost = 0.20 * precio
        ingresos = vendido * precio
        costo_pedido = pedido_qty * unit_cost

        
        # 6. Actualización de estadísticas
        # valores actuales de inventario, pedido y satisfacción 
        self._update_running("inv", float(self.inventario))
        self._update_running("order", float(pedido_qty))
        fill_rate = float(vendido) / max(demanda, 1)
        self._update_running("fill", fill_rate)

        # Cálculo de desviaciones
        # calcular qué tan desviado está el agente respecto a lo ideal
        inv_std = self._running_std("inv")
        order_std = self._running_std("order")
        fill_std = self._running_std("fill")

        # z-score del inventario
        forecast = self.data[self.current_step, IDX_FORECAST]
        target = max(1.0, forecast * 1.2) # definimos en el entorno que tener un inventario del 120% del pronostico es "ideal"
        z_inv = (self.inv_mean - target) / (inv_std + 1e-8)
        z_inv = float(np.clip(z_inv, -3.0, 3.0))

        # Volatilidad del pedido
        # cuanto más volátil es el pedido, más castigo se le da (evitar que sea erratico)
        order_vol = order_std / (self.order_mean + 1e-6)
        order_vol = float(np.clip(order_vol, 0.0, 5.0))

        # Deficiencia de satisfacción
        # si se están perdiendo ventas, este valor sube
        fill_def = max(0.0, 0.5 - self.fill_mean)  # si fill_mean < 0.5, entonces fill_def es positivo

        # Multiplicadores dinámicos
        # Aquí es donde el entorno ajusta los castigos basándose en cálculos anteriores
        hold_mult = 1.0 + self.k_hold * z_inv - 0.5 * fill_def
        hold_mult = float(np.clip(hold_mult, self.hold_clip[0], self.hold_clip[1]))

        order_mult = 1.0 + self.k_order * order_vol
        order_mult = float(np.clip(order_mult, self.order_clip[0], self.order_clip[1]))

        smooth_mult = 1.0 + self.k_smooth * order_vol
        smooth_mult = float(np.clip(smooth_mult, self.smooth_clip[0], self.smooth_clip[1]))

        # 7. Aplicar los multiplicadores
        costo_guardado = self.inventario * (self.base_hold * hold_mult)

        costo_pedidos_grandes = self.base_order_pen * order_mult * (pedido_qty ** 2)

        costo_cambio = smooth_mult * self.base_smooth * abs(pedido_qty - getattr(self, "ultimo_pedido_qty", 0))

        costo_stockout = (demanda - vendido) * (precio * 1.5)

        exceso = max(0, self.inventario - self.max_inventario * 0.75)
        costo_exceso_nl = 0.0025 * (exceso ** 1.45)

        beneficio = ingresos - costo_pedido - costo_guardado - costo_pedidos_grandes - costo_cambio - costo_stockout - costo_exceso_nl

        # Pequeño recompensa por la tasa de satisfacción/cumplimiento
        reward_fill = 0.5 * fill_rate
        beneficio += reward_fill

        # 8. Escalar la recompensa 
        recompensa = beneficio / 4000.0
        recompensa = float(np.clip(recompensa, -3.0, 3.0))

        # 9. Actualizar el último pedido y el paso actual
        self.ultimo_pedido_qty = pedido_qty
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
            "hold_mult": hold_mult,
            "order_mult": order_mult,
            "smooth_mult": smooth_mult,
            "fill_rate": fill_rate,
        }
        return obs, recompensa, terminated, truncated, info

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
    
    def _update_running(self, name, value):
        # Algoritmo incremental de Welford para calcular la media y la varianza
        # mantiene la media y M2
        if name == "inv":
            c = self.inv_count + 1.0
            delta = value - self.inv_mean
            self.inv_mean += delta / c
            delta2 = value - self.inv_mean
            self.inv_M2 = self.inv_M2 + delta * delta2
            self.inv_count = c
        elif name == "order":
            c = self.order_count + 1.0
            delta = value - self.order_mean
            self.order_mean += delta / c
            delta2 = value - self.order_mean
            self.order_M2 = self.order_M2 + delta * delta2
            self.order_count = c
        elif name == "fill":
            c = self.fill_count + 1.0
            delta = value - self.fill_mean
            self.fill_mean += delta / c
            delta2 = value - self.fill_mean
            self.fill_M2 = self.fill_M2 + delta * delta2
            self.fill_count = c
    
    def _running_std(self, name):
        if name == "inv":
            if self.inv_count < 2: return 1.0
            var = max(self.inv_M2 / (self.inv_count - 1), 1e-6)
            return var ** 0.5
        if name == "order":
            if self.order_count < 2: return 1.0
            var = max(self.order_M2 / (self.order_count - 1), 1e-6)
            return var ** 0.5
        if name == "fill":
            if self.fill_count < 2: return 0.1
            var = max(self.fill_M2 / (self.fill_count - 1), 1e-6)
            return var ** 0.5


