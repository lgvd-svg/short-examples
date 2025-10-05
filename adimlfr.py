import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.gridspec as gridspec

class DimensionlessLFR:
    """
    Laminar Flow Reactor Adimensional
    Usa números adimensionales para análisis general
    """
    
    def __init__(self):
        # ===========================================
        # NÚMEROS ADIMENSIONALES PRINCIPALES
        # ===========================================
        
        # Número de Damköhler (Da) - Razón reacción/transporte
        self.Da = 1.0
        
        # Número de Péclet (Pe) - Razón transporte convectivo/difusivo
        self.Pe = 100.0
        
        # Número de Thiele modificado (Φ²) - Razón reacción/difusión radial
        self.Phi_sq = 0.1
        
        # Parámetro de velocidad axial (para coordenadas transformadas)
        self.alpha = 1.0
        
    def dimensionless_velocity(self, r_star):
        """
        Perfil de velocidad adimensional
        u*(r*) = 2*(1 - r*²)
        donde r* = r/R (0 ≤ r* ≤ 1)
        """
        return 2.0 * (1.0 - r_star**2)
    
    def dimensionless_governing_equation(self, eta, y, r_star):
        """
        Ecuación gobernante adimensional del LFR
        
        Parameters:
        eta: Coordenada axial adimensional (z/L)
        y: Variable dependiente (concentración adimensional C* = CA/CA0)
        r_star: Posición radial adimensional (r/R)
        """
        C_star = y[0]
        
        # Término convectivo: u*(r*) * dC*/deta
        u_star = self.dimensionless_velocity(r_star)
        convective = u_star
        
        # Término difusivo radial: (1/Pe) * (d²C*/dr*² + (1/r*) * dC*/dr*)
        # Para simplicidad, asumimos gradientes radiales resueltos numéricamente
        
        # Término reactivo: -Da * C*
        reactive = -self.Da * C_star
        
        # Ecuación adimensional completa
        dCstar_deta = (reactive) / u_star  # Simplificado para integración axial
        
        return [dCstar_deta]
    
    def solve_radial_profiles(self, n_axial=100, n_radial=50):
        """
        Resuelve los perfiles radiales adimensionales a lo largo del reactor
        """
        # Coordenadas adimensionales
        eta = np.linspace(0, 1, n_axial)  # Coordenada axial adimensional
        r_star = np.linspace(0, 1, n_radial)  # Coordenada radial adimensional
        
        # Matriz de concentración adimensional
        C_star_profile = np.zeros((n_axial, n_radial))
        C_star_profile[0, :] = 1.0  # Condición inicial: C* = 1 en toda la entrada
        
        # Velocidades adimensionales
        u_star = self.dimensionless_velocity(r_star)
        
        # Resolver para cada posición radial
        for j, r_val in enumerate(r_star):
            if r_val == 0:
                # En el centro, usar aproximación de simetría
                r_val = 1e-6
                
            # Condición inicial para esta línea de corriente
            y0 = [1.0]  # C* = 1 en la entrada
            
            # Integrar a lo largo de la dirección axial
            for i in range(1, n_axial):
                # Paso de integración simple
                deta = eta[i] - eta[i-1]
                dC_deta = self.dimensionless_governing_equation(eta[i], [C_star_profile[i-1, j]], r_val)[0]
                C_star_profile[i, j] = C_star_profile[i-1, j] + dC_deta * deta
                
                # Asegurar que la concentración no sea negativa
                C_star_profile[i, j] = max(0, C_star_profile[i, j])
        
        return eta, r_star, C_star_profile, u_star
    
    def conversion_flow_average(self, C_star_profile, r_star, u_star):
        """
        Calcula la conversión promedio ponderada por flujo
        X_avg = 1 - ∫∫ C* * u* * r* dr* dtheta / ∫∫ u* * r* dr* dtheta
        """
        conversion_avg = np.zeros(C_star_profile.shape[0])
        
        for i in range(C_star_profile.shape[0]):
            # Numerador: ∫ C* * u* * r* dr*
            numerator = np.trapz(C_star_profile[i, :] * u_star * r_star, r_star)
            
            # Denominador: ∫ u* * r* dr*
            denominator = np.trapz(u_star * r_star, r_star)
            
            # Conversión promedio
            conversion_avg[i] = 1.0 - numerator / denominator
        
        return conversion_avg
    
    def dimensionless_residence_time(self, r_star):
        """
        Tiempo de residencia adimensional para cada línea de corriente
        tau* = 1 / u*(r*)
        """
        u_star = self.dimensionless_velocity(r_star)
        return 1.0 / u_star
    
    def segregation_analysis(self):
        """
        Análisis de segregación usando distribución de tiempos de residencia
        """
        # Distribución de tiempos de residencia adimensional
        r_star_detailed = np.linspace(0, 1, 1000)
        tau_star = self.dimensionless_residence_time(r_star_detailed)
        u_star_detailed = self.dimensionless_velocity(r_star_detailed)
        
        # Función densidad de probabilidad E(tau*)
        # Para flujo laminar: E(tau*) = 1/(2*tau*³) para tau* ≥ 0.5
        tau_star_range = np.linspace(0.5, 10, 1000)
        E_tau_star = np.zeros_like(tau_star_range)
        mask = tau_star_range >= 0.5
        E_tau_star[mask] = 1.0 / (2.0 * tau_star_range[mask]**3)
        
        # Normalizar E(tau*)
        E_tau_star = E_tau_star / np.trapz(E_tau_star, tau_star_range)
        
        # Conversión para flujo segregado (reactor batch)
        # C*_batch = exp(-Da * tau*)
        C_star_batch = np.exp(-self.Da * tau_star_range)
        X_batch = 1.0 - C_star_batch
        
        # Conversión promedio segregada
        X_segregated = np.trapz(X_batch * E_tau_star, tau_star_range)
        
        # Conversión para PFR ideal (mismo tiempo de residencia promedio)
        tau_star_mean = 1.0  # Para perfil parabólico, el tiempo medio adimensional es 1
        X_pfr = 1.0 - np.exp(-self.Da * tau_star_mean)
        
        return tau_star_range, E_tau_star, X_batch, X_segregated, X_pfr, tau_star_mean
    
    def calculate_effectiveness_factor(self, C_star_profile, r_star):
        """
        Calcula el factor de efectividad adimensional
        eta = (tasa de reacción real) / (tasa de reacción sin gradientes)
        """
        # Tasa de reacción promedio
        avg_reaction_rate = np.mean(self.Da * C_star_profile[-1, :])
        
        # Tasa de reacción sin gradientes (en la entrada)
        max_reaction_rate = self.Da * 1.0
        
        effectiveness = avg_reaction_rate / max_reaction_rate
        
        return effectiveness

def plot_dimensionless_lfr_results(eta, r_star, C_star_profile, u_star, conversion_avg):
    """Visualiza resultados adimensionales del LFR"""
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(3, 3, figure=fig)
    
    # 1. Perfil de concentración 2D adimensional
    ax1 = fig.add_subplot(gs[0, 0])
    ETA, R_STAR = np.meshgrid(eta, r_star)
    contour = ax1.contourf(ETA, R_STAR, C_star_profile.T, levels=50, cmap='viridis')
    plt.colorbar(contour, ax=ax1, label='C* = CA/CA0')
    ax1.set_xlabel('η = z/L')
    ax1.set_ylabel('r* = r/R')
    ax1.set_title('Perfil de Concentración Adimensional C*(η, r*)')
    
    # 2. Perfiles radiales en diferentes posiciones axiales
    ax2 = fig.add_subplot(gs[0, 1])
    positions = [0, len(eta)//4, len(eta)//2, 3*len(eta)//4, -1]
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    labels = ['η = 0', 'η = 0.25', 'η = 0.5', 'η = 0.75', 'η = 1.0']
    
    for pos, color, label in zip(positions, colors, labels):
        ax2.plot(r_star, C_star_profile[pos, :], color=color, 
                linewidth=2, label=label)
    
    ax2.set_xlabel('r*')
    ax2.set_ylabel('C*')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Perfiles Radiales Adimensionales')
    
    # 3. Perfil de velocidad adimensional
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(r_star, u_star, 'b-', linewidth=3)
    ax3.set_xlabel('r*')
    ax3.set_ylabel('u*')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Perfil de Velocidad Adimensional u*(r*)')
    
    # 4. Conversión promedio vs posición axial
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(eta, conversion_avg * 100, 'r-', linewidth=3)
    ax4.set_xlabel('η')
    ax4.set_ylabel('Conversión [%]')
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Conversión Promedio vs Posición Axial')
    
    # 5. Concentración en centro y pared
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(eta, C_star_profile[:, 0], 'b-', linewidth=2, label='Centro (r* = 0)')
    ax5.plot(eta, C_star_profile[:, -1], 'r-', linewidth=2, label='Pared (r* = 1)')
    ax5.set_xlabel('η')
    ax5.set_ylabel('C*')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_title('Concentración en Centro y Pared')
    
    # 6. Tiempos de residencia adimensionales
    ax6 = fig.add_subplot(gs[1, 2])
    tau_star = 1.0 / u_star
    ax6.plot(r_star, tau_star, 'g-', linewidth=3)
    ax6.set_xlabel('r*')
    ax6.set_ylabel('τ*')
    ax6.grid(True, alpha=0.3)
    ax6.set_title('Tiempo de Residencia Adimensional τ*(r*)')
    
    # 7. Gradientes radiales máximos
    ax7 = fig.add_subplot(gs[2, 0])
    grad_r = np.gradient(C_star_profile, axis=1)
    max_grad = np.max(np.abs(grad_r), axis=1)
    ax7.plot(eta, max_grad, 'purple', linewidth=2)
    ax7.set_xlabel('η')
    ax7.set_ylabel('|∂C*/∂r*|ₘₐₓ')
    ax7.grid(True, alpha=0.3)
    ax7.set_title('Gradiente Radial Máximo')
    
    # 8. Velocidad de reacción adimensional promedio
    ax8 = fig.add_subplot(gs[2, 1])
    reaction_rate_avg = np.mean(C_star_profile * u_star, axis=1)
    ax8.plot(eta, reaction_rate_avg, 'orange', linewidth=2)
    ax8.set_xlabel('η')
    ax8.set_ylabel('⟨r*·C*·u*⟩')
    ax8.grid(True, alpha=0.3)
    ax8.set_title('Velocidad de Reacción Promedio')
    
    plt.tight_layout()
    plt.show()

def plot_dimensionless_rtd_analysis(tau_star_range, E_tau_star, X_batch, X_segregated, X_pfr):
    """Visualiza análisis de DTR adimensional"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Distribución de tiempos de residencia adimensional
    ax1.plot(tau_star_range, E_tau_star, 'b-', linewidth=3, label='E(τ*)')
    ax1.set_xlabel('τ*')
    ax1.set_ylabel('E(τ*)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('DTR Adimensional - Flujo Laminar')
    
    # 2. Análisis de segregación adimensional
    ax2.plot(tau_star_range, X_batch * 100, 'g-', linewidth=2, 
             label='X(τ*) Batch')
    ax2.axhline(X_segregated * 100, color='blue', linestyle='--', 
                linewidth=3, label=f'X_segregated = {X_segregated*100:.1f}%')
    ax2.axhline(X_pfr * 100, color='red', linestyle='--', 
                linewidth=3, label=f'X_PFR = {X_pfr*100:.1f}%')
    ax2.set_xlabel('τ*')
    ax2.set_ylabel('Conversión [%]')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Análisis de Segregación Adimensional')
    
    plt.tight_layout()
    plt.show()

def main():
    # ===========================================
    # SIMULACIÓN DEL LFR ADIMENSIONAL
    # ===========================================
    print("Inicializando Laminar Flow Reactor Adimensional...")
    lfr = DimensionlessLFR()
    
    print("\n" + "="*60)
    print("PARÁMETROS ADIMENSIONALES")
    print("="*60)
    print(f"Número de Damköhler (Da): {lfr.Da}")
    print(f"Número de Péclet (Pe): {lfr.Pe}")
    print(f"Número de Thiele (Φ²): {lfr.Phi_sq}")
    
    # Interpretación física
    print(f"\nINTERPRETACIÓN FÍSICA:")
    if lfr.Da < 0.1:
        print("  Da < 0.1: Régimen controlado por convección")
    elif lfr.Da > 10:
        print("  Da > 10: Régimen controlado por reacción")
    else:
        print("  Da ≈ 1: Régimen mixto")
    
    if lfr.Pe > 100:
        print("  Pe > 100: Difusión axial despreciable")
    else:
        print("  Pe ≤ 100: Difusión axial significativa")
    
    # ===========================================
    # RESOLVER PERFILES ADIMENSIONALES
    # ===========================================
    print("\nResolviendo ecuaciones adimensionales...")
    eta, r_star, C_star_profile, u_star = lfr.solve_radial_profiles()
    
    # ===========================================
    # CÁLCULOS ADICIONALES
    # ===========================================
    conversion_avg = lfr.conversion_flow_average(C_star_profile, r_star, u_star)
    effectiveness = lfr.calculate_effectiveness_factor(C_star_profile, r_star)
    
    # Análisis de DTR
    tau_star_range, E_tau_star, X_batch, X_segregated, X_pfr, tau_star_mean = lfr.segregation_analysis()
    
    # ===========================================
    # VISUALIZACIÓN ADIMENSIONAL
    # ===========================================
    print("\nGenerando gráficas adimensionales...")
    plot_dimensionless_lfr_results(eta, r_star, C_star_profile, u_star, conversion_avg)
    plot_dimensionless_rtd_analysis(tau_star_range, E_tau_star, X_batch, X_segregated, X_pfr)
    
    # ===========================================
    # RESULTADOS NUMÉRICOS
    # ===========================================
    print("\n" + "="*60)
    print("RESULTADOS ADIMENSIONALES")
    print("="*60)
    print(f"Conversión final promedio: {conversion_avg[-1]*100:.2f} %")
    print(f"Conversión PFR ideal: {X_pfr*100:.2f} %")
    print(f"Conversión segregada: {X_segregated*100:.2f} %")
    print(f"Factor de efectividad: {effectiveness:.4f}")
    
    # Efecto de la segregación
    segregation_effect = (X_pfr - X_segregated) / X_pfr * 100
    print(f"Efecto de segregación: {segregation_effect:.2f} %")
    
    # Conversión por posición radial en la salida
    print(f"\nConversión en salida (η = 1) por posición radial:")
    print(f"  Centro (r* = 0): {(1 - C_star_profile[-1, 0])*100:.2f} %")
    mid_idx = len(r_star) // 2
    print(f"  Medio (r* = 0.5): {(1 - C_star_profile[-1, mid_idx])*100:.2f} %")
    print(f"  Pared (r* = 1): {(1 - C_star_profile[-1, -1])*100:.2f} %")
    
    # ===========================================
    # ESTUDIO DE SENSIBILIDAD ADIMENSIONAL
    # ===========================================
    def dimensionless_sensitivity_analysis():
        """Análisis de sensibilidad a parámetros adimensionales"""
        print("\n" + "="*60)
        print("ANÁLISIS DE SENSIBILIDAD ADIMENSIONAL")
        print("="*60)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Sensibilidad al número de Damköhler
        Da_values = [0.1, 0.5, 1.0, 2.0, 5.0]
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for Da_val, color in zip(Da_values, colors):
            lfr_temp = DimensionlessLFR()
            lfr_temp.Da = Da_val
            eta_temp, r_star_temp, C_star_temp, u_star_temp = lfr_temp.solve_radial_profiles()
            conv_temp = lfr_temp.conversion_flow_average(C_star_temp, r_star_temp, u_star_temp)
            
            axes[0, 0].plot(eta_temp, conv_temp * 100, color=color, 
                           linewidth=2, label=f'Da = {Da_val}')
        
        axes[0, 0].set_xlabel('η')
        axes[0, 0].set_ylabel('Conversión [%]')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_title('Sensibilidad al Número de Damköhler')
        
        # Sensibilidad al número de Péclet
        Pe_values = [10, 50, 100, 200, 500]
        for Pe_val, color in zip(Pe_values, colors):
            lfr_temp = DimensionlessLFR()
            lfr_temp.Pe = Pe_val
            eta_temp, r_star_temp, C_star_temp, u_star_temp = lfr_temp.solve_radial_profiles()
            conv_temp = lfr_temp.conversion_flow_average(C_star_temp, r_star_temp, u_star_temp)
            
            axes[0, 1].plot(eta_temp, conv_temp * 100, color=color, 
                           linewidth=2, label=f'Pe = {Pe_val}')
        
        axes[0, 1].set_xlabel('η')
        axes[0, 1].set_ylabel('Conversión [%]')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_title('Sensibilidad al Número de Péclet')
        
        # Comparación con modelos ideales
        eta_range = np.linspace(0, 1, 100)
        axes[1, 0].plot(eta_range, (1 - np.exp(-lfr.Da * eta_range)) * 100, 'k--',
                       linewidth=3, label='PFR Ideal')
        axes[1, 0].plot(eta, conversion_avg * 100, 'r-',
                       linewidth=2, label='LFR Real')
        axes[1, 0].set_xlabel('η')
        axes[1, 0].set_ylabel('Conversión [%]')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_title('Comparación LFR vs PFR Ideal')
        
        # Efecto de segregación vs Da
        Da_range = np.logspace(-2, 2, 50)
        segregation_effects = []
        for Da_val in Da_range:
            lfr_temp = DimensionlessLFR()
            lfr_temp.Da = Da_val
            _, _, _, _, X_seg, X_pfr = lfr_temp.segregation_analysis()
            seg_effect = (X_pfr - X_seg) / X_pfr * 100
            segregation_effects.append(seg_effect)
        
        axes[1, 1].semilogx(Da_range, segregation_effects, 'b-', linewidth=3)
        axes[1, 1].set_xlabel('Da')
        axes[1, 1].set_ylabel('Efecto de Segregación [%]')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_title('Efecto de Segregación vs Número de Damköhler')
        
        plt.tight_layout()
        plt.show()
    
    dimensionless_sensitivity_analysis()

if __name__ == "__main__":
    main()