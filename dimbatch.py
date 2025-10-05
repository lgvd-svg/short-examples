import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class DimensionlessBatchReactor:
    """
    Reactor Batch Adimensional con números adimensionales
    Reacción: A → B + Calor
    """
    
    def __init__(self):
        # ===========================================
        # NÚMEROS ADIMENSIONALES
        # ===========================================
        
        # Número de Damköhler (Da) - Razón tiempo característico flujo/reacción
        self.Da = 2.0
        
        # Número de Arrhenius (γ) - Energía de activación adimensional
        self.gamma = 1.0
        
        # Número de Prater (β) - Generación de calor adimensional
        self.beta = 1.0
        
        # Número de Stanton (St) - Transferencia de calor adimensional
        self.St = 1.0
        
        # Número de Capacitancia (C) - Razón capacitancias térmicas
        self.C = 1.0
        
        # Condición inicial adimensional de temperatura
        self.theta_initial = 0.0
        
    def dimensionless_reaction_rate(self, X, theta):
        """
        Velocidad de reacción adimensional
        X: Conversión adimensional [0-1]
        theta: Temperatura adimensional
        """
        # Término de Arrhenius adimensional
        arrhenius = np.exp((self.gamma * theta) / (1 + theta))
        
        # Velocidad adimensional = Da * (1-X) * exp(gamma*theta/(1+theta))
        rate = self.Da * (1 - X) * arrhenius
        
        return rate
    
    def dimensionless_energy_balance(self, tau, y):
        """
        Sistema de ecuaciones diferenciales adimensionales
        y = [X, theta, theta_j]
        
        donde:
        X: Conversión adimensional
        theta: Temperatura del reactor adimensional  
        theta_j: Temperatura de la jaqueta adimensional
        tau: Tiempo adimensional
        """
        X, theta, theta_j = y
        
        # ===========================================
        # BALANCE DE MATERIA ADIMENSIONAL
        # ===========================================
        rate = self.dimensionless_reaction_rate(X, theta)
        dXdtau = rate
        
        # ===========================================
        # BALANCE DE ENERGÍA ADIMENSIONAL - REACTOR
        # ===========================================
        # Término de generación: β * Da * (1-X) * exp(gamma*theta/(1+theta))
        heat_generation = self.beta * rate
        
        # Término de remoción: St * (theta - theta_j)
        heat_removal = self.St * (theta - theta_j)
        
        dthetadtau = heat_generation - heat_removal
        
        # ===========================================
        # BALANCE DE ENERGÍA ADIMENSIONAL - JAQUETA
        # ===========================================
        # Para jaqueta con flujo continuo
        dtheta_jdtau = self.St * self.C * (theta - theta_j) - self.C * theta_j
        
        return [dXdtau, dthetadtau, dtheta_jdtau]
    
    def simulate(self, tau_span, tau_eval, initial_conditions=None):
        """
        Ejecuta simulación adimensional
        
        Parameters:
        tau_span: tuple - Intervalo de tiempo adimensional (tau0, tauf)
        tau_eval: array - Tiempos adimensionales donde evaluar
        initial_conditions: list - [X0, theta0, theta_j0] (adimensionales)
        """
        if initial_conditions is None:
            initial_conditions = [0.0, 0.0, 0.0]  # X, theta, theta_j
        
        # Resolver el sistema de EDOs adimensionales
        solution = solve_ivp(
            fun=lambda tau, y: self.dimensionless_energy_balance(tau, y),
            t_span=tau_span,
            y0=initial_conditions,
            t_eval=tau_eval,
            method='BDF',
            rtol=1e-8,
            atol=1e-10
        )
        
        return solution
    
    def calculate_dimensionless_variables(self, X, theta):
        """Calcula variables adimensionales adicionales"""
        rates = np.array([self.dimensionless_reaction_rate(x, t) for x, t in zip(X, theta)])
        
        # Calor adimensional generado
        Q_gen_dimensionless = self.beta * rates
        
        return rates, Q_gen_dimensionless

def dimensionless_to_dimensional(reactor, tau, X, theta, theta_j, 
                               T_ref, CA0, t_ref, DH_rxn, parameters):
    """
    Convierte resultados adimensionales a dimensionales
    
    Parameters:
    reactor: Instancia de DimensionlessBatchReactor
    tau, X, theta, theta_j: Variables adimensionales
    T_ref: Temperatura de referencia [K]
    CA0: Concentración inicial A [kmol/m³]
    t_ref: Tiempo característico [s]
    DH_rxn: Entalpía de reacción [J/mol]
    parameters: Diccionario con parámetros dimensionales
    """
    # Conversión a variables dimensionales
    t_dimensional = tau * t_ref  # [s]
    CA_dimensional = CA0 * (1 - X)  # [kmol/m³]
    CB_dimensional = CA0 * X  # [kmol/m³]
    T_dimensional = T_ref * (1 + theta)  # [K]
    T_j_dimensional = T_ref * (1 + theta_j)  # [K]
    
    # Velocidad de reacción dimensional
    rate_dimensional = (CA0 / t_ref) * reactor.dimensionless_reaction_rate(X, theta)
    
    return {
        't': t_dimensional,
        'CA': CA_dimensional,
        'CB': CB_dimensional, 
        'T': T_dimensional,
        'T_j': T_j_dimensional,
        'rate': rate_dimensional,
        'conversion': X * 100  # [%]
    }

def plot_dimensionless_results(tau, X, theta, theta_j, rates, Q_gen):
    """Grafica resultados adimensionales"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Gráfica 1: Conversión adimensional
    ax1.plot(tau, X, 'b-', linewidth=3, label='X (Conversión)')
    ax1.set_xlabel('Tiempo Adimensional (τ)')
    ax1.set_ylabel('Conversión Adimensional (X)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Conversión vs Tiempo Adimensional')
    
    # Gráfica 2: Temperaturas adimensionales
    ax2.plot(tau, theta, 'r-', linewidth=3, label='θ Reactor')
    ax2.plot(tau, theta_j, 'b-', linewidth=3, label='θ Jaqueta')
    ax2.set_xlabel('Tiempo Adimensional (τ)')
    ax2.set_ylabel('Temperatura Adimensional (θ)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Temperaturas Adimensionales')
    
    # Gráfica 3: Velocidad de reacción adimensional
    ax3.plot(tau, rates, 'purple', linewidth=3)
    ax3.set_xlabel('Tiempo Adimensional (τ)')
    ax3.set_ylabel('Velocidad Adimensional (r*)')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Velocidad de Reacción Adimensional')
    
    # Gráfica 4: Calor generado adimensional
    ax4.plot(tau, Q_gen, 'orange', linewidth=3)
    ax4.set_xlabel('Tiempo Adimensional (τ)')
    ax4.set_ylabel('Calor Generado Adimensional (Q*)')
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Calor Generado Adimensional')
    
    plt.tight_layout()
    plt.show()

def plot_comparison_dimensional(dim_results):
    """Grafica resultados convertidos a dimensionales"""
    t = dim_results['t'] / 60  # Convertir a minutos
    CA = dim_results['CA']
    CB = dim_results['CB']
    T = dim_results['T'] - 273.15  # Convertir a °C
    T_j = dim_results['T_j'] - 273.15
    conversion = dim_results['conversion']
    rate = dim_results['rate'] * 1000  # Convertir a mol/m³s
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Gráfica 1: Concentraciones dimensionales
    ax1.plot(t, CA, 'b-', linewidth=2, label='CA [kmol/m³]')
    ax1.plot(t, CB, 'r-', linewidth=2, label='CB [kmol/m³]')
    ax1.set_xlabel('Tiempo [min]')
    ax1.set_ylabel('Concentración [kmol/m³]')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Concentraciones Dimensionales')
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(t, conversion, 'g--', linewidth=2, label='Conversión [%]')
    ax1_twin.set_ylabel('Conversión [%]')
    ax1_twin.legend(loc='lower right')
    
    # Gráfica 2: Temperaturas dimensionales
    ax2.plot(t, T, 'r-', linewidth=2, label='T Reactor [°C]')
    ax2.plot(t, T_j, 'b-', linewidth=2, label='T Jaqueta [°C]')
    ax2.set_xlabel('Tiempo [min]')
    ax2.set_ylabel('Temperatura [°C]')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Temperaturas Dimensionales')
    
    # Gráfica 3: Velocidad de reacción dimensional
    ax3.plot(t, rate, 'purple', linewidth=2)
    ax3.set_xlabel('Tiempo [min]')
    ax3.set_ylabel('Velocidad [mol/m³s]')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Velocidad de Reacción Dimensional')
    
    plt.tight_layout()
    plt.show()

def main():
    # ===========================================
    # CONFIGURACIÓN ADIMENSIONAL
    # ===========================================
    reactor = DimensionlessBatchReactor()
    
    # Condiciones iniciales adimensionales [X0, theta0, theta_j0]
    initial_conditions = [0.0, 0.0, 0.0]  # Sin conversión, temperatura de referencia
    
    # Tiempo adimensional de simulación
    tau_span = (0, 2)  # Rango adimensional típico
    tau_eval = np.linspace(0, 2, 1000)
    
    # ===========================================
    # EJECUTAR SIMULACIÓN ADIMENSIONAL
    # ===========================================
    print("Ejecutando simulación adimensional del reactor batch...")
    solution = reactor.simulate(tau_span, tau_eval, initial_conditions)
    
    # Extraer resultados adimensionales
    tau = solution.t
    X = solution.y[0]      # Conversión adimensional
    theta = solution.y[1]  # Temperatura reactor adimensional
    theta_j = solution.y[2]  # Temperatura jaqueta adimensional
    
    # Calcular variables adicionales adimensionales
    rates, Q_gen = reactor.calculate_dimensionless_variables(X, theta)
    
    # ===========================================
    # MOSTRAR PARÁMETROS ADIMENSIONALES
    # ===========================================
    print("\n" + "="*60)
    print("PARÁMETROS ADIMENSIONALES")
    print("="*60)
    print(f"Número de Damköhler (Da): {reactor.Da}")
    print(f"Número de Arrhenius (γ): {reactor.gamma}")
    print(f"Número de Prater (β): {reactor.beta}")
    print(f"Número de Stanton (St): {reactor.St}")
    print(f"Número de Capacitancia (C): {reactor.C}")
    
    # ===========================================
    # ANÁLISIS DE RESULTADOS ADIMENSIONALES
    # ===========================================
    print("\n" + "="*60)
    print("RESULTADOS ADIMENSIONALES")
    print("="*60)
    print(f"Conversión final adimensional: {X[-1]:.4f}")
    print(f"Temperatura máxima adimensional: {theta.max():.4f}")
    print(f"Velocidad máxima adimensional: {rates.max():.4f}")
    
    # Encontrar tiempo para conversión del 90% adimensional
    idx_90 = np.where(X >= 0.9)[0]
    if len(idx_90) > 0:
        tau_90 = tau[idx_90[0]]
        print(f"Tiempo adimensional para 90% conversión: {tau_90:.2f}")
    else:
        print("No se alcanzó 90% de conversión en el tiempo simulado")
    
    # ===========================================
    # VISUALIZACIÓN ADIMENSIONAL
    # ===========================================
    plot_dimensionless_results(tau, X, theta, theta_j, rates, Q_gen)
    
    # ===========================================
    # CONVERSIÓN A VARIABLES DIMENSIONALES (OPCIONAL)
    # ===========================================
    print("\n" + "="*60)
    print("CONVERSIÓN A VARIABLES DIMENSIONALES")
    print("="*60)
    
    # Parámetros dimensionales de referencia
    T_ref = 293.15  # Temperatura de referencia [K] (20°C)
    CA0 = 2.5       # Concentración inicial [kmol/m³]
    t_ref = 1800    # Tiempo característico [s] (30 minutos)
    DH_rxn = -80000 # Entalpía de reacción [J/mol]
    
    parameters = {
        'T_ref': T_ref,
        'CA0': CA0,
        't_ref': t_ref,
        'DH_rxn': DH_rxn
    }
    
    # Convertir a dimensional
    dim_results = dimensionless_to_dimensional(reactor, tau, X, theta, theta_j,
                                             T_ref, CA0, t_ref, DH_rxn, parameters)
    
    print(f"Conversión final: {dim_results['conversion'][-1]:.2f} %")
    print(f"Temperatura máxima: {dim_results['T'].max() - 273.15:.2f} °C")
    print(f"Concentración final de B: {dim_results['CB'][-1]:.3f} kmol/m³")
    
    # Graficar resultados dimensionales
    plot_comparison_dimensional(dim_results)
    
    # ===========================================
    # ESTUDIO DE SENSIBILIDAD ADIMENSIONAL
    # ===========================================
    def dimensionless_sensitivity_analysis():
        """Estudio de sensibilidad de parámetros adimensionales"""
        print("\n" + "="*60)
        print("ANÁLISIS DE SENSIBILIDAD ADIMENSIONAL")
        print("="*60)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Variar número de Damköhler (Da)
        Da_values = [0.5, 1.0, 2.0, 5.0]
        colors = ['red', 'blue', 'green', 'purple']
        
        for i, (param_name, param_values, param_dict) in enumerate([
            ('Da', Da_values, {'Da': lambda x: x}),
            ('β', [0.1, 0.3, 0.5, 0.7], {'beta': lambda x: x}),
            ('γ', [15, 20, 25, 30], {'gamma': lambda x: x})
        ]):
            
            for j, (param_val, color) in enumerate(zip(param_values, colors)):
                # Crear nuevo reactor con parámetro modificado
                reactor_sens = DimensionlessBatchReactor()
                
                # Actualizar parámetro específico
                param_key = list(param_dict.keys())[0]
                setattr(reactor_sens, param_key, param_dict[param_key](param_val))
                
                # Simular
                sol_sens = reactor_sens.simulate(tau_span, tau_eval, initial_conditions)
                
                # Graficar conversión
                axes[0, i].plot(sol_sens.t, sol_sens.y[0], color=color, linewidth=2,
                              label=f'{param_name} = {param_val}')
                
                # Graficar temperatura
                axes[1, i].plot(sol_sens.t, sol_sens.y[1], color=color, linewidth=2,
                              label=f'{param_name} = {param_val}')
            
            axes[0, i].set_xlabel('τ')
            axes[0, i].set_ylabel('X')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
            axes[0, i].set_title(f'Sensibilidad a {param_name} - Conversión')
            
            axes[1, i].set_xlabel('τ')
            axes[1, i].set_ylabel('θ')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)
            axes[1, i].set_title(f'Sensibilidad a {param_name} - Temperatura')
        
        plt.tight_layout()
        plt.show()
    
    # Ejecutar análisis de sensibilidad
    dimensionless_sensitivity_analysis()

if __name__ == "__main__":
    main()