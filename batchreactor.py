import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd

class BatchReactor:
    """
    Clase que implementa un reactor batch con jaqueta de enfriamiento
    Reacción: A → B + Calor
    """
    
    def __init__(self):
        # Parámetros del reactor - CORREGIDOS
        self.V_total = 2.0                 # Volumen total [m³]
        self.U = 500.0                     # Coeficiente global [W/m²K]
        self.A_heat = 8.0                  # Área de transferencia [m²]
        self.Cp_rxn = 2500.0               # Capacidad calorífica [J/kgK]
        self.Rho_rxn = 850.0               # Densidad [kg/m³]
        
        # Cinética de reacción - CORREGIDOS (valores más realistas)
        self.k0 = 1.2e8                    # Factor pre-exponencial [1/s]
        self.Ea = 65000.0                  # Energía de activación [J/mol]
        self.R_gas = 8.314                 # Constante de gases [J/mol·K]
        self.DH_rxn = -80000.0             # Entalpía de reacción [J/mol]
        
        # Parámetros del coolant
        self.Cp_cool = 4180.0              # [J/kgK]
        self.Rho_cool = 1000.0             # [kg/m³]
        
        # Condiciones de operación
        self.Fc = 0.5                      # Flujo de coolant [kg/s]
        self.T_cool_in = 288.15            # Temperatura entrada coolant [K]
        
        # Condición inicial de concentración A (para cálculo de conversión)
        self.CA0 = 2.5  # [kmol/m³]
        
    def kinetic_rate(self, CA, T):
        """Calcula la velocidad de reacción con Arrhenius"""
        k = self.k0 * np.exp(-self.Ea / (self.R_gas * T))
        return k * CA
    
    def energy_balance(self, t, y, Fc, T_cool_in):
        """
        Sistema de ecuaciones diferenciales del reactor batch
        y = [CA, CB, T, T_j]
        """
        CA, CB, T, T_j = y
        
        # Velocidad de reacción
        rate = self.kinetic_rate(CA, T)
        
        # ===========================================
        # BALANCES DE MATERIA
        # ===========================================
        dCAdt = -rate
        dCBdt = rate
        
        # ===========================================
        # BALANCES DE ENERGÍA
        # ===========================================
        # Calor generado por reacción [W/m³]
        Q_rxn = -rate * self.DH_rxn
        
        # Calor removido por enfriamiento [W/m³]
        Q_cool = self.U * self.A_heat * (T - T_j) / self.V_total
        
        # Balance de energía del reactor
        # Masa total = densidad * volumen [kg]
        masa_total = self.Rho_rxn * self.V_total
        dTdt = (Q_rxn - Q_cool) * self.V_total / (masa_total * self.Cp_rxn)
        
        # Balance de energía para la jaqueta
        # Volumen de la jaqueta asumido
        V_jacket = 0.1 * self.V_total  # [m³]
        masa_jacket = V_jacket * self.Rho_cool  # [kg]
        dT_jdt = (Fc * self.Cp_cool * (T_cool_in - T_j) + 
                 Q_cool * self.V_total) / (masa_jacket * self.Cp_cool)
        
        return [dCAdt, dCBdt, dTdt, dT_jdt]
    
    def simulate(self, t_span, t_eval, initial_conditions, Fc=None, T_cool_in=None):
        """
        Ejecuta la simulación del reactor batch
        """
        if Fc is None:
            Fc = self.Fc
        if T_cool_in is None:
            T_cool_in = self.T_cool_in
            
        # Guardar CA inicial para cálculo de conversión
        self.CA0 = initial_conditions[0]
            
        # Resolver el sistema de EDOs
        solution = solve_ivp(
            fun=lambda t, y: self.energy_balance(t, y, Fc, T_cool_in),
            t_span=t_span,
            y0=initial_conditions,
            t_eval=t_eval,
            method='BDF',
            rtol=1e-6,
            atol=1e-8
        )
        
        return solution
    
    def calculate_additional_variables(self, CA, T):
        """Calcula variables adicionales para análisis"""
        rates = np.array([self.kinetic_rate(ca, temp) for ca, temp in zip(CA, T)])
        Q_rxn = -rates * self.DH_rxn * self.V_total  # [W]
        conversion = (self.CA0 - CA) / self.CA0 * 100
        
        return rates, Q_rxn, conversion

def safe_max_conversion_time(t, conversion, target=90):
    """Calcula el tiempo para alcanzar una conversión objetivo de manera segura"""
    idx = np.where(conversion >= target)[0]
    if len(idx) > 0:
        return t[idx[0]] / 60
    else:
        max_conv = conversion.max()
        return f"No alcanzado (máx: {max_conv:.1f}%)"

def main():
    # ===========================================
    # CONFIGURACIÓN DE LA SIMULACIÓN
    # ===========================================
    reactor = BatchReactor()
    
    # Condiciones iniciales [CA0, CB0, T0, T_j0]
    initial_conditions = [2.5, 0.0, 293.15, 293.15]  # [kmol/m³, kmol/m³, K, K]
    
    # Tiempo de simulación
    t_span = (0, 7200)  # 2 horas de simulación [s]
    t_eval = np.linspace(0, 7200, 1000)
    
    # ===========================================
    # EJECUTAR SIMULACIÓN
    # ===========================================
    print("Ejecutando simulación del reactor batch...")
    solution = reactor.simulate(t_span, t_eval, initial_conditions)
    
    # Extraer resultados
    t = solution.t
    CA = solution.y[0]
    CB = solution.y[1]
    T = solution.y[2]
    T_j = solution.y[3]
    
    # Calcular variables adicionales
    rates, Q_rxn, conversion = reactor.calculate_additional_variables(CA, T)
    
    # ===========================================
    # VISUALIZACIÓN DE RESULTADOS
    # ===========================================
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Gráfica 1: Concentraciones y Conversión
    ax1.plot(t/60, CA, 'b-', linewidth=2, label='CA [kmol/m³]')
    ax1.plot(t/60, CB, 'r-', linewidth=2, label='CB [kmol/m³]')
    ax1.set_xlabel('Tiempo [min]')
    ax1.set_ylabel('Concentración [kmol/m³]')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Evolución de Concentraciones')
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(t/60, conversion, 'g--', linewidth=2, label='Conversión [%]')
    ax1_twin.set_ylabel('Conversión [%]')
    ax1_twin.legend(loc='lower right')
    
    # Gráfica 2: Temperaturas
    ax2.plot(t/60, T - 273.15, 'r-', linewidth=2, label='T Reactor [°C]')
    ax2.plot(t/60, T_j - 273.15, 'b-', linewidth=2, label='T Jaqueta [°C]')
    ax2.set_xlabel('Tiempo [min]')
    ax2.set_ylabel('Temperatura [°C]')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Evolución de Temperaturas')
    
    # Gráfica 3: Velocidad de Reacción
    ax3.plot(t/60, rates * 1000, 'purple', linewidth=2)
    ax3.set_xlabel('Tiempo [min]')
    ax3.set_ylabel('Velocidad de Reacción [mol/m³s]')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Velocidad de Reacción')
    
    # Gráfica 4: Calores
    ax4.plot(t/60, Q_rxn/1000, 'r-', linewidth=2, label='Q Reacción [kW]')
    
    # Calcular calor de enfriamiento
    Q_cool = reactor.U * reactor.A_heat * (T - T_j)
    ax4.plot(t/60, Q_cool/1000, 'b-', linewidth=2, label='Q Enfriamiento [kW]')
    
    ax4.set_xlabel('Tiempo [min]')
    ax4.set_ylabel('Calor [kW]')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Balance de Energía')
    
    plt.tight_layout()
    plt.show()
    
    # ===========================================
    # ANÁLISIS DE RESULTADOS
    # ===========================================
    print("\n" + "="*50)
    print("ANÁLISIS DE RESULTADOS")
    print("="*50)
    print(f"Conversión final: {conversion[-1]:.2f} %")
    print(f"Temperatura máxima: {T.max() - 273.15:.2f} °C")
    print(f"Concentración final de B: {CB[-1]:.3f} kmol/m³")
    
    # Cálculo seguro del tiempo de conversión
    time_90 = safe_max_conversion_time(t, conversion, 90)
    time_50 = safe_max_conversion_time(t, conversion, 50)
    
    print(f"Tiempo para 50% conversión: {time_50:.1f} min")
    print(f"Tiempo para 90% conversión: {time_90}")
    
    # Mostrar información adicional
    print(f"\nVelocidad máxima de reacción: {rates.max()*1000:.4f} mol/m³s")
    print(f"Calor máximo generado: {Q_rxn.max()/1000:.2f} kW")
    
    # ===========================================
    # ESTUDIO DE SENSIBILIDAD
    # ===========================================
    def sensitivity_analysis():
        """Estudio de sensibilidad del flujo de coolant"""
        print("\n" + "="*50)
        print("EJECUTANDO ANÁLISIS DE SENSIBILIDAD")
        print("="*50)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        Fc_values = [0.2, 0.5, 1.0, 2.0]
        colors = ['red', 'blue', 'green', 'purple']
        
        for Fc, color in zip(Fc_values, colors):
            print(f"Simulando con Fc = {Fc} kg/s...")
            sol = reactor.simulate(t_span, t_eval, initial_conditions, Fc=Fc)
            T_sens = sol.y[2] - 273.15
            CA_sens = sol.y[0]
            conversion_sens = (initial_conditions[0] - CA_sens) / initial_conditions[0] * 100
            
            axes[0].plot(sol.t/60, T_sens, color=color, linewidth=2, 
                        label=f'Fc = {Fc} kg/s')
            axes[1].plot(sol.t/60, conversion_sens, color=color, linewidth=2, 
                        label=f'Fc = {Fc} kg/s')
            
            # Mostrar resultados para cada caso
            max_temp = T_sens.max()
            final_conv = conversion_sens[-1]
            print(f"  Fc={Fc} kg/s -> Temp máx: {max_temp:.1f}°C, Conversión final: {final_conv:.1f}%")
        
        axes[0].set_xlabel('Tiempo [min]')
        axes[0].set_ylabel('Temperatura Reactor [°C]')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('Sensibilidad - Temperatura vs Flujo Coolant')
        
        axes[1].set_xlabel('Tiempo [min]')
        axes[1].set_ylabel('Conversión [%]')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title('Sensibilidad - Conversión vs Flujo Coolant')
        
        plt.tight_layout()
        plt.show()
    
    # Ejecutar análisis de sensibilidad
    sensitivity_analysis()

if __name__ == "__main__":
    main()