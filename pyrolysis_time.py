import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 1. SISTEMA DINÁMICO ADIMENSIONAL CON MÁS TIEMPO

class TransientDimensionlessPyrolysis:
    """Sistema de pirólisis en modo transiente con tiempo extendido"""
    
    def __init__(self, reference_conditions=None):
        self.ref = reference_conditions or {
            'mass_flow': 1000, 'temperature': 500, 'pressure': 101325,
            'length': 1.0, 'time': 3600, 'energy': 1000, 'volume': 1.0
        }
        
        # PARÁMETROS CINÉTICOS OPTIMIZADOS PARA VER EVOLUCIÓN
        self.k0_star = 5.0       # REDUCIDO: Para hacer la reacción más lenta
        self.Ea_star = 1.0        # REDUCIDO: Para activar más fácilmente
        self.Da = 0.8            # AUMENTADO: Para ver mejor los efectos
        
        # Condiciones iniciales adimensionales
        self.initial_conditions = {
            'biomass_star': 1.0,      # Concentración de biomasa
            'biooil_star': 0.0,       # Concentración de bio-oil
            'gas_star': 0.0,          # Concentración de gas
            'char_star': 0.0,         # Concentración de char
            'temp_star': 0.5,         # REDUCIDO: Temperatura inicial más baja
            'moisture_star': 0.4      # AUMENTADO: Más humedad para ver secado
        }
        
        # TIEMPO ADIMENSIONAL EXTENDIDO
        self.tau_max = 10.0       # AUMENTADO SIGNIFICATIVAMENTE
        self.n_steps = 2000       # MÁS PUNTOS PARA MEJOR RESOLUCIÓN
    
    def pyrolysis_kinetics(self, t_star, y_star):
        """
        Ecuaciones diferenciales con cinética más lenta para ver evolución
        """
        biomass, biooil, gas, char, temp, moisture = y_star
        
        # CORRECCIÓN: Evitar división por cero en Arrhenius
        temp_safe = max(temp, 0.1)
        
        # Constante cinética adimensional - MÁS LENTA
        k_star = self.k0_star * np.exp(-self.Ea_star / temp_safe)
        
        # TASAS DE REACCIÓN MÁS LENTAS PARA VER EVOLUCIÓN
        # 1. Secado 
        drying_rate = 1.5 * moisture * temp_safe if temp_safe > 0.4 else 0.0
        
        # 2. Pirólisis primaria - MÁS LENTA
        pyrolysis_rate = k_star * biomass * self.Da * 0.1 /(1 + 0.1*char + 0.1*biooil) # FACTOR REDUCIDO
        
        # 3. Pirólisis secundaria - MUCHO MÁS LENTA
        secondary_rate = 0.02 * k_star * biooil * self.Da /(1 + 0.1*char + 0.1*biooil)
        
        # SISTEMA DE EDOS CON EVOLUCIÓN MÁS GRADUAL
        dbiomass_dt = -pyrolysis_rate
        
        dbiooil_dt = (0.5 * pyrolysis_rate - secondary_rate)
        
        dgas_dt = (0.3 * pyrolysis_rate + 0.4 * secondary_rate)
        
        dchar_dt = (0.2 * pyrolysis_rate + 0.6 * secondary_rate)
        
        # BALANCE DE ENERGÍA MÁS LENTO
        heat_supplied = 0.5 * (1.0 - temp_safe)   # REDUCIDO: Calentamiento más lento
        heat_reaction = 0.1 * pyrolysis_rate      # REDUCIDO
        heat_losses = 0.2 * (temp_safe - 0.5)     # AJUSTADO
        
        dtemp_dt = heat_supplied - heat_reaction - heat_losses
        
        # Secado
        dmoisture_dt = -drying_rate
        
        return [dbiomass_dt, dbiooil_dt, dgas_dt, dchar_dt, dtemp_dt, dmoisture_dt]
    
    def solve_transient(self, t_span=None, method='BDF'):
        """Resuelve el sistema transiente con tiempo extendido"""
        if t_span is None:
            t_span = [0, self.tau_max]
        
        # Condiciones iniciales
        y0 = list(self.initial_conditions.values())
        
        # MÁS PUNTOS DE EVALUACIÓN PARA MEJOR RESOLUCIÓN
        t_eval = np.linspace(t_span[0], t_span[1], self.n_steps)
        
        print(f" Resolviendo desde τ=0 hasta τ={self.tau_max} con {self.n_steps} puntos...")
        
        # Resolver sistema de EDOs
        solution = solve_ivp(
            fun=self.pyrolysis_kinetics,
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method=method,
            rtol=1e-8,
            atol=1e-10
        )
        
        if not solution.success:
            print(f" Advertencia: {solution.message}")
        
        self.solution = solution
        return solution

    # ... (el resto de los métodos de esta clase se mantienen igual)
    def analyze_transient_results(self):
        """Analiza resultados de la simulación transiente"""
        if not hasattr(self, 'solution'):
            print("Primero ejecuta solve_transient()")
            return
        
        t_star = self.solution.t
        biomass = self.solution.y[0]
        biooil = self.solution.y[1]
        gas = self.solution.y[2]
        char = self.solution.y[3]
        temperature = self.solution.y[4]
        moisture = self.solution.y[5]
        
        print("\n" + "="*70)
        print("ANÁLISIS DE RESULTADOS TRANSITORIOS - TIEMPO EXTENDIDO")
        print("="*70)
        
        # Encontrar tiempos característicos
        biooil_max_idx = np.argmax(biooil)
        tau_biooil_max = t_star[biooil_max_idx]
        
        drying_complete_idx = np.where(moisture < 0.01)[0]
        tau_drying = t_star[drying_complete_idx[0]] if len(drying_complete_idx) > 0 else t_star[-1]
        
        # Encontrar tiempo de conversión significativa
        conversion_90_idx = np.where(biomass < 0.1)[0]
        tau_90_conversion = t_star[conversion_90_idx[0]] if len(conversion_90_idx) > 0 else t_star[-1]
        
        print(f"⏱️  TIEMPOS CARACTERÍSTICOS:")
        print(f"   Máximo producción bio-oil: τ = {tau_biooil_max:.3f}")
        print(f"   Secado completo: τ = {tau_drying:.3f}")
        print(f"   90% conversión: τ = {tau_90_conversion:.3f}")
        
        # Rendimientos finales
        final_yields = {
            'Bio-oil': biooil[-1],
            'Gas': gas[-1],
            'Char': char[-1],
            'Biomasa residual': biomass[-1]
        }
        
        print(f"\n RENDIMIENTOS FINALES:")
        for product, yield_val in final_yields.items():
            print(f"   {product}: {yield_val:.3f}")
        
        # Balance de masa final
        mass_balance = biooil[-1] + gas[-1] + char[-1] + biomass[-1]
        print(f"   Balance de masa: {mass_balance:.3f}")
        
        return {
            'time_star': t_star,
            'biomass': biomass,
            'biooil': biooil,
            'gas': gas,
            'char': char,
            'temperature': temperature,
            'moisture': moisture,
            'characteristic_times': {
                'biooil_max': tau_biooil_max,
                'drying_complete': tau_drying,
                '90_conversion': tau_90_conversion
            },
            'final_yields': final_yields
        }

# 2. REACTOR CON TIEMPO EXTENDIDO

class TransientDimensionlessReactor:
    """Reactor de pirólisis con dinámica temporal extendida"""
    
    def __init__(self, name, ref_conditions):
        self.name = name
        self.ref = ref_conditions
        
        # Estado transiente del reactor
        self.state = {
            'tau': 0.0,           
            'theta_reactor': 0.5,  # REDUCIDO: Temperatura inicial más baja
            'conversion': 0.0,    
            'pressure_star': 1.0  
        }
        
        self.design_params = self.calculate_initial_design()
    
    def calculate_initial_design(self):
        return {
            'volume_star': 1.0,
            'heat_area_star': 2.5,
            'cross_area_star': 0.8,
            'power_star': 1.5
        }
    
    def update_design_with_time(self, tau, conversion, temperature):
        """Actualiza parámetros de diseño con límites físicos"""
        volume_reduction = min(conversion * 0.5, 0.5)
        volume_effective = self.design_params['volume_star'] * (1 - volume_reduction)
        
        area_increase = min(temperature * 0.3, 0.5)
        heat_area_effective = self.design_params['heat_area_star'] * (0.8 + area_increase)
        
        power_increase = min(conversion * 0.3 + temperature * 0.2, 0.8)
        power_required = self.design_params['power_star'] * (1 + power_increase)
        
        efficiency_reduction = min(conversion * 0.3, 0.4)
        efficiency = max(0.7 * (1 - efficiency_reduction), 0.3)
        
        return {
            'volume_star': max(volume_effective, 0.5),
            'heat_area_star': max(heat_area_effective, 1.5),
            'cross_area_star': self.design_params['cross_area_star'],
            'power_star': min(power_required, 3.0),
            'efficiency_star': min(efficiency, 0.9)
        }
    
    def reactor_dynamics(self, tau, state):
        """Dinámica del reactor más lenta"""
        theta, conversion, pressure = state
        
        theta_safe = max(theta, 0.1)
        
        # DINÁMICA MÁS LENTA
        dtheta_dtau = 0.6 * (1.0 - theta_safe) - 0.2 * conversion  # REDUCIDO
        
        k_rate = 2.0 * np.exp(-10.0 / theta_safe)  # REDUCIDO
        dconversion_dtau = k_rate * (1 - conversion)
        
        dpressure_dtau = 0.03 * (conversion - pressure)  # REDUCIDO
        
        return [dtheta_dtau, dconversion_dtau, dpressure_dtau]
    
    def solve_reactor_dynamics(self, tau_max=10.0):  # TIEMPO EXTENDIDO
        """Resuelve la dinámica del reactor con tiempo extendido"""
        y0 = [self.state['theta_reactor'], self.state['conversion'], self.state['pressure_star']]
        
        tau_eval = np.linspace(0, tau_max, 1000)  # MÁS PUNTOS
        
        solution = solve_ivp(
            fun=self.reactor_dynamics,
            t_span=[0, tau_max],
            y0=y0,
            t_eval=tau_eval,
            method='BDF'
        )
        
        self.reactor_solution = solution
        return solution
    
    def get_design_evolution(self):
        if not hasattr(self, 'reactor_solution'):
            print("Primero ejecuta solve_reactor_dynamics()")
            return
        
        tau = self.reactor_solution.t
        temperature = self.reactor_solution.y[0]
        conversion = self.reactor_solution.y[1]
        
        design_evolution = []
        for i in range(len(tau)):
            design_at_tau = self.update_design_with_time(tau[i], conversion[i], temperature[i])
            design_evolution.append(design_at_tau)
        
        return tau, design_evolution

# 3. SISTEMA COMPLETO CON TIEMPO EXTENDIDO

class CompleteTransientPyrolysis:
    """Sistema completo de pirólisis con tiempo extendido"""
    
    def __init__(self, ref_conditions=None):
        self.ref = ref_conditions or {
            'mass_flow': 1000, 'temperature': 500, 'pressure': 101325,
            'length': 1.0, 'time': 3600, 'energy': 1000, 'volume': 1.0
        }
        
        self.pyrolysis_system = TransientDimensionlessPyrolysis(self.ref)
        self.reactor = TransientDimensionlessReactor("Reactor_Transiente", self.ref)
        self.results = {}
    
    def run_complete_transient_simulation(self, tau_max=10.0):  # TIEMPO EXTENDIDO
        """Ejecuta simulación transiente completa con tiempo extendido"""
        print("=" * 70)
        print(f"SIMULACIÓN CON TIEMPO EXTENDIDO (τ_max = {tau_max})")
        print("=" * 70)
        
        # Actualizar tau_max en subsistemas
        self.pyrolysis_system.tau_max = tau_max
        
        print("\n1. SIMULANDO CINÉTICA DE PIRÓLISIS...")
        pyrolysis_solution = self.pyrolysis_system.solve_transient(t_span=[0, tau_max])
        pyrolysis_results = self.pyrolysis_system.analyze_transient_results()
        
        print("\n2. SIMULANDO DINÁMICA DEL REACTOR...")
        reactor_solution = self.reactor.solve_reactor_dynamics(tau_max)
        tau_reactor, design_evolution = self.reactor.get_design_evolution()
        
        self.results = {
            'pyrolysis': pyrolysis_results,
            'reactor': {
                'time_star': tau_reactor,
                'temperature': reactor_solution.y[0],
                'conversion': reactor_solution.y[1],
                'pressure': reactor_solution.y[2],
                'design_evolution': design_evolution
            }
        }
        
        return self.results
    
    def plot_transient_results(self):
        """Genera gráficas de los resultados con tiempo extendido"""
        if not self.results:
            print("Primero ejecuta run_complete_transient_simulation()")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(f'EVOLUCIÓN TEMPORAL EXTENDIDA - τ_max = {self.pyrolysis_system.tau_max}', fontsize=16)
        
        # Datos de pirólisis
        pyro_data = self.results['pyrolysis']
        t_pyro = pyro_data['time_star']
        
        # GRÁFICA 1: Composición vs tiempo
        axes[0, 0].plot(t_pyro, pyro_data['biomass'], 'b-', label='Biomasa', linewidth=2.5)
        axes[0, 0].plot(t_pyro, pyro_data['biooil'], 'g-', label='Bio-oil', linewidth=2.5)
        axes[0, 0].plot(t_pyro, pyro_data['gas'], 'r-', label='Gas', linewidth=2.5)
        axes[0, 0].plot(t_pyro, pyro_data['char'], 'k-', label='Char', linewidth=2.5)
        axes[0, 0].set_xlabel('Tiempo Adimensional (τ)')
        axes[0, 0].set_ylabel('Composición Adimensional')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_title('EVOLUCIÓN DE COMPOSICIÓN (TIEMPO EXTENDIDO)')
        axes[0, 0].set_xlim(0, self.pyrolysis_system.tau_max)  # AJUSTAR LÍMITE
        
        # Gráfica 2: Temperatura y humedad
        axes[0, 1].plot(t_pyro, pyro_data['temperature'], 'r-', linewidth=2, label='Temperatura')
        axes[0, 1].set_xlabel('Tiempo Adimensional (τ)')
        axes[0, 1].set_ylabel('Temperatura Adimensional', color='red')
        axes[0, 1].tick_params(axis='y', labelcolor='red')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xlim(0, self.pyrolysis_system.tau_max)
        
        ax2 = axes[0, 1].twinx()
        ax2.plot(t_pyro, pyro_data['moisture'], 'b-', linewidth=2, label='Humedad')
        ax2.set_ylabel('Humedad Adimensional', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        axes[0, 1].set_title('Temperatura y Humedad')
        
        # Gráfica 3: Conversión del reactor
        reactor_data = self.results['reactor']
        axes[0, 2].plot(reactor_data['time_star'], reactor_data['conversion'], 'purple', linewidth=2)
        axes[0, 2].set_xlabel('Tiempo Adimensional (τ)')
        axes[0, 2].set_ylabel('Conversión')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_title('Conversión de Biomasa')
        axes[0, 2].set_xlim(0, self.pyrolysis_system.tau_max)
        
        # Gráficas restantes...
        volumes = [design['volume_star'] for design in reactor_data['design_evolution']]
        axes[1, 0].plot(reactor_data['time_star'], volumes, 'orange', linewidth=2)
        axes[1, 0].set_xlabel('Tiempo Adimensional (τ)')
        axes[1, 0].set_ylabel('Volumen Adimensional')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_title('Volumen Efectivo del Reactor')
        axes[1, 0].set_xlim(0, self.pyrolysis_system.tau_max)
        
        areas = [design['heat_area_star'] for design in reactor_data['design_evolution']]
        axes[1, 1].plot(reactor_data['time_star'], areas, 'green', linewidth=2)
        axes[1, 1].set_xlabel('Tiempo Adimensional (τ)')
        axes[1, 1].set_ylabel('Área Adimensional')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_title('Área de Transferencia de Calor')
        axes[1, 1].set_xlim(0, self.pyrolysis_system.tau_max)
        
        efficiencies = [design['efficiency_star'] for design in reactor_data['design_evolution']]
        axes[1, 2].plot(reactor_data['time_star'], efficiencies, 'red', linewidth=2)
        axes[1, 2].set_xlabel('Tiempo Adimensional (τ)')
        axes[1, 2].set_ylabel('Eficiencia Adimensional')
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_title('Eficiencia del Proceso')
        axes[1, 2].set_xlim(0, self.pyrolysis_system.tau_max)
        
        plt.tight_layout()
        plt.show()
    
    def generate_transient_report(self):
        """Genera reporte de la simulación con tiempo extendido"""
        if not self.results:
            print("Primero ejecuta la simulación")
            return
        
        pyro_data = self.results['pyrolysis']
        
        print("\n" + "="*70)
        print("REPORTE CON TIEMPO EXTENDIDO")
        print("="*70)
        
        print(f"\n RANGO TEMPORAL: τ = 0 a {self.pyrolysis_system.tau_max}")
        print(f" PUNTOS DE SIMULACIÓN: {len(pyro_data['time_star'])}")
        
        # Análisis de evolución
        biomass_change = pyro_data['biomass'][0] - pyro_data['biomass'][-1]
        biooil_max = np.max(pyro_data['biooil'])
        
        print(f"\n EVOLUCIÓN OBSERVADA:")
        print(f"   Cambio en biomasa: {pyro_data['biomass'][0]:.3f} → {pyro_data['biomass'][-1]:.3f}")
        print(f"   Bio-oil máximo: {biooil_max:.3f}")
        print(f"   Conversión final: {1 - pyro_data['biomass'][-1]:.3f}")

# 4. FUNCIÓN PRINCIPAL CON TIEMPO EXTENDIDO

def main_transient_extended(tau_max=10.0):
    """Función principal con tiempo extendido"""
    print(" INICIANDO SIMULACIÓN CON TIEMPO EXTENDIDO")
    print(f"   τ_max = {tau_max}")
    print("=" * 70)
    
    try:
        transient_system = CompleteTransientPyrolysis()
        results = transient_system.run_complete_transient_simulation(tau_max=tau_max)
        
        # Verificación de evolución
        pyro_data = results['pyrolysis']
        
        print("\n" + "="*50)
        print("VERIFICACIÓN DE EVOLUCIÓN TEMPORAL")
        print("="*50)
        
        # Calcular variación en cada especie
        variations = {
            'Biomasa': np.ptp(pyro_data['biomass']),  # Peak-to-peak variation
            'Bio-oil': np.ptp(pyro_data['biooil']),
            'Gas': np.ptp(pyro_data['gas']),
            'Char': np.ptp(pyro_data['char'])
        }
        
        for species, variation in variations.items():
            status = "BUENA" if variation > 0.1 else "⚠️  BAJA"
            print(f"   {species}: variación = {variation:.3f} {status}")
        
        transient_system.generate_transient_report()
        transient_system.plot_transient_results()
        
        return transient_system, results
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# 5. EJECUTAR CON DIFERENTES TIEMPOS
if __name__ == "__main__":
    print("SIMULACIÓN CON TIEMPO EXTENDIDO")
    
    # Probar diferentes tiempos máximos
    time_options = [5.0, 12.0, 14.0, 16.0, 18.0, 20.0]  # Diferentes τ_max para probar
    
    for tau_max in time_options:
        print(f"\n{'='*60}")
        print(f"EJECUTANDO CON τ_max = {tau_max}")
        print(f"{'='*60}")
        
        system, results = main_transient_extended(tau_max=tau_max)
        
        if system and results:
            pyro_data = results['pyrolysis']
            final_biomass = pyro_data['biomass'][-1]
            
            if final_biomass < 0.1:  # Si se alcanza conversión alta
                print(f"CON τ_max = {tau_max}: Conversión completa alcanzada")
                optimal_tau = tau_max
                break
            else:
                print(f"CON τ_max = {tau_max}: Conversión incompleta ({final_biomass:.3f} biomasa residual)")
        else:
            print(f"CON τ_max = {tau_max}: Simulación falló")
    
    print(f"\n TIEMPO ÓPTIMO RECOMENDADO: τ_max = {optimal_tau if 'optimal_tau' in locals() else 10.0}")
