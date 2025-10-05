import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import odeint

class LaminarFlowReactor:
    """
    LFR - Laminar Flow Reactor
    Modela un reactor tubular con perfil de velocidad parabólico
    Reacción: A → Productos
    """
    
    def __init__(self):
        # Parámetros geométricos del reactor
        self.L = 2.0           # Longitud del reactor [m]
        self.R = 0.01          # Radio del reactor [m] (1 cm)
        
        # Parámetros de flujo
        self.u_avg = 0.1       # Velocidad promedio [m/s]
        self.nu = 1e-6         # Viscosidad cinemática [m²/s] (agua)
        
        # Parámetros de reacción
        self.k = 0.5           # Constante de velocidad [1/s]
        self.CA0 = 1.0         # Concentración inicial [mol/m³]
        
        # Parámetros numéricos
        self.n_radial = 50     # Número de puntos en dirección radial
        self.n_axial = 100     # Número de puntos en dirección axial
        
    def parabolic_velocity(self, r):
        """
        Perfil de velocidad parabólico para flujo laminar
        u(r) = 2*u_avg*(1 - (r/R)^2)
        """
        return 2 * self.u_avg * (1 - (r / self.R)**2)
    
    def residence_time_distribution(self, t):
        """
        Distribución de tiempos de residencia para flujo laminar
        E(t) = 0 para t < t_min
        E(t) = t_mean^2 / (2*t^3) para t >= t_min
        """
        t_mean = self.L / self.u_avg
        t_min = t_mean / 2
        
        E = np.zeros_like(t)
        mask = t >= t_min
        E[mask] = t_mean**2 / (2 * t[mask]**3)
        
        return E, t_mean, t_min
    
    def solve_radial_profile(self, method='finite_difference'):
        """
        Resuelve el perfil radial de concentración en el LFR
        """
        # Coordenadas radiales (de 0 a R)
        r = np.linspace(0, self.R, self.n_radial)
        dr = r[1] - r[0]
        
        # Velocidades en cada punto radial
        u = self.parabolic_velocity(r)
        
        # Condición inicial (concentración uniforme)
        CA = np.ones(self.n_radial) * self.CA0
        
        # Almacenar resultados a lo largo del reactor
        z_positions = np.linspace(0, self.L, self.n_axial)
        CA_profile = np.zeros((len(z_positions), len(r)))
        CA_profile[0, :] = CA
        
        if method == 'finite_difference':
            # Método de diferencias finitas
            for i, z in enumerate(z_positions[1:], 1):
                # Derivadas en dirección radial (condición de simetría)
                dCA_dr = np.zeros_like(r)
                d2CA_dr2 = np.zeros_like(r)
                
                # Puntos internos
                for j in range(1, len(r)-1):
                    dCA_dr[j] = (CA[j+1] - CA[j-1]) / (2 * dr)
                    d2CA_dr2[j] = (CA[j+1] - 2*CA[j] + CA[j-1]) / (dr**2)
                
                # Condición de frontera: simetría en r=0
                dCA_dr[0] = 0
                d2CA_dr2[0] = 2 * (CA[1] - CA[0]) / (dr**2)
                
                # Condición de frontera: pared impermeable en r=R
                dCA_dr[-1] = 0
                d2CA_dr2[-1] = 2 * (CA[-2] - CA[-1]) / (dr**2)
                
                # Ecuación de transporte
                for j in range(len(r)):
                    if r[j] > 0:
                        transport = -u[j] * (CA[j] - CA_profile[i-1, j]) / (z_positions[i] - z_positions[i-1])
                        diffusion = (d2CA_dr2[j] + (1/r[j]) * dCA_dr[j]) if r[j] > 0 else 2 * d2CA_dr2[j]
                        reaction = -self.k * CA[j]
                        
                        CA[j] = CA_profile[i-1, j] + (transport + diffusion + reaction) * (z_positions[i] - z_positions[i-1])
                    else:
                        # En el centro, término singular
                        transport = -u[j] * (CA[j] - CA_profile[i-1, j]) / (z_positions[i] - z_positions[i-1])
                        diffusion = 2 * d2CA_dr2[j]
                        reaction = -self.k * CA[j]
                        
                        CA[j] = CA_profile[i-1, j] + (transport + diffusion + reaction) * (z_positions[i] - z_positions[i-1])
                
                CA_profile[i, :] = CA.copy()
        
        return z_positions, r, CA_profile, u
    
    def conversion_radial_average(self, CA_profile, r):
        """
        Calcula la conversión promedio radial
        """
        # Conversión en cada punto radial: 1 - CA/CA0
        conversion_profile = 1 - CA_profile / self.CA0
        
        # Promedio ponderado por el flujo (velocidad)
        u = self.parabolic_velocity(r)
        conversion_avg = np.zeros(CA_profile.shape[0])
        
        for i in range(CA_profile.shape[0]):
            # Integración numérica para promedio ponderado
            numerator = np.trapz(conversion_profile[i, :] * u * r, r)
            denominator = np.trapz(u * r, r)
            conversion_avg[i] = numerator / denominator
        
        return conversion_avg
    
    def segregation_analysis(self):
        """
        Análisis de segregación usando la DTR
        """
        t = np.linspace(0.1, 10, 1000)
        E_t, t_mean, t_min = self.residence_time_distribution(t)
        
        # Conversión para flujo segregado (batch)
        CA_batch = self.CA0 * np.exp(-self.k * t)
        conversion_batch = 1 - CA_batch / self.CA0
        
        # Conversión promedio segregada
        conversion_segregated = np.trapz(conversion_batch * E_t, t)
        
        # Conversión para flujo mezclado (PFR ideal)
        conversion_pfr = 1 - np.exp(-self.k * t_mean)
        
        return t, E_t, conversion_batch, conversion_segregated, conversion_pfr, t_mean

def plot_lfr_results(z, r, CA_profile, u, conversion_avg):
    """Visualiza resultados del LFR"""
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Perfil de concentración 2D
    ax1 = plt.subplot(2, 3, 1)
    Z, R = np.meshgrid(z, r)
    contour = ax1.contourf(Z, R * 1000, CA_profile.T, levels=50, cmap='viridis')
    plt.colorbar(contour, ax=ax1, label='Concentración [mol/m³]')
    ax1.set_xlabel('Posición Axial [m]')
    ax1.set_ylabel('Radio [mm]')
    ax1.set_title('Perfil de Concentración 2D en el LFR')
    
    # 2. Perfiles radiales en diferentes posiciones
    ax2 = plt.subplot(2, 3, 2)
    positions = [0, len(z)//4, len(z)//2, 3*len(z)//4, -1]
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for pos, color in zip(positions, colors):
        ax2.plot(r * 1000, CA_profile[pos, :], color=color, 
                linewidth=2, label=f'z = {z[pos]:.2f} m')
    
    ax2.set_xlabel('Radio [mm]')
    ax2.set_ylabel('Concentración [mol/m³]')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Perfiles Radiales de Concentración')
    
    # 3. Perfil de velocidad
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(r * 1000, u, 'b-', linewidth=3)
    ax3.set_xlabel('Radio [mm]')
    ax3.set_ylabel('Velocidad [m/s]')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Perfil Parabólico de Velocidad')
    
    # 4. Conversión promedio a lo largo del reactor
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(z, conversion_avg * 100, 'r-', linewidth=3)
    ax4.set_xlabel('Posición Axial [m]')
    ax4.set_ylabel('Conversión [%]')
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Conversión Promedio vs Posición Axial')
    
    # 5. Concentración en el centro y pared
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(z, CA_profile[:, 0], 'b-', linewidth=2, label='Centro (r=0)')
    ax5.plot(z, CA_profile[:, -1], 'r-', linewidth=2, label='Pared (r=R)')
    ax5.set_xlabel('Posición Axial [m]')
    ax5.set_ylabel('Concentración [mol/m³]')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_title('Concentración en Centro y Pared')
    
    # 6. Gradientes radiales
    ax6 = plt.subplot(2, 3, 6)
    gradientes = np.gradient(CA_profile, axis=1)
    max_grad = np.max(np.abs(gradientes), axis=1)
    ax6.plot(z, max_grad, 'purple', linewidth=2)
    ax6.set_xlabel('Posición Axial [m]')
    ax6.set_ylabel('Máximo Gradiente Radial [mol/m⁴]')
    ax6.grid(True, alpha=0.3)
    ax6.set_title('Máximo Gradiente Radial')
    
    plt.tight_layout()
    plt.show()

def plot_rtd_analysis(t, E_t, conversion_batch, conversion_segregated, conversion_pfr, t_mean):
    """Visualiza análisis de DTR y segregación"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Distribución de tiempos de residencia
    ax1.plot(t, E_t, 'b-', linewidth=3, label='E(t)')
    ax1.axvline(t_mean, color='red', linestyle='--', linewidth=2, label=f't_mean = {t_mean:.2f} s')
    ax1.set_xlabel('Tiempo [s]')
    ax1.set_ylabel('E(t) [1/s]')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Distribución de Tiempos de Residencia (DTR)')
    
    # 2. Análisis de segregación
    ax2.plot(t, conversion_batch * 100, 'g-', linewidth=2, label='Conversión Batch')
    ax2.axhline(conversion_segregated * 100, color='blue', linestyle='--', 
               linewidth=3, label=f'Segregado: {conversion_segregated*100:.1f}%')
    ax2.axhline(conversion_pfr * 100, color='red', linestyle='--', 
               linewidth=3, label=f'PFR Ideal: {conversion_pfr*100:.1f}%')
    ax2.set_xlabel('Tiempo [s]')
    ax2.set_ylabel('Conversión [%]')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Análisis de Segregación')
    
    plt.tight_layout()
    plt.show()

def main():
    # ===========================================
    # SIMULACIÓN DEL LAMINAR FLOW REACTOR
    # ===========================================
    print("Inicializando Laminar Flow Reactor...")
    lfr = LaminarFlowReactor()
    
    print(f"Parámetros del LFR:")
    print(f"Longitud: {lfr.L} m")
    print(f"Radio: {lfr.R*1000} mm")
    print(f"Velocidad promedio: {lfr.u_avg} m/s")
    print(f"Constante cinética: {lfr.k} 1/s")
    print(f"Concentración inicial: {lfr.CA0} mol/m³")
    
    # ===========================================
    # RESOLVER PERFIL RADIAL
    # ===========================================
    print("\nResolviendo perfil radial...")
    z, r, CA_profile, u = lfr.solve_radial_profile()
    
    # ===========================================
    # CALCULAR CONVERSIÓN PROMEDIO
    # ===========================================
    conversion_avg = lfr.conversion_radial_average(CA_profile, r)
    
    # ===========================================
    # ANÁLISIS DE DTR Y SEGREGACIÓN
    # ===========================================
    print("\nRealizando análisis de DTR...")
    t, E_t, conversion_batch, conversion_segregated, conversion_pfr, t_mean = lfr.segregation_analysis()
    
    # ===========================================
    # VISUALIZACIÓN
    # ===========================================
    print("\nGenerando gráficas...")
    plot_lfr_results(z, r, CA_profile, u, conversion_avg)
    plot_rtd_analysis(t, E_t, conversion_batch, conversion_segregated, conversion_pfr, t_mean)
    
    # ===========================================
    # RESULTADOS NUMÉRICOS
    # ===========================================
    print("\n" + "="*60)
    print("RESULTADOS DEL LFR")
    print("="*60)
    print(f"Conversión final promedio: {conversion_avg[-1]*100:.2f} %")
    print(f"Conversión PFR ideal: {conversion_pfr*100:.2f} %")
    print(f"Conversión segregada: {conversion_segregated*100:.2f} %")
    print(f"Tiempo medio de residencia: {t_mean:.2f} s")
    
    # Efecto de la segregación
    segregation_effect = (conversion_pfr - conversion_segregated) / conversion_pfr * 100
    print(f"Efecto de segregación: {segregation_effect:.2f} %")
    
    # Conversión en diferentes posiciones radiales
    print(f"\nConversión en salida por posición radial:")
    print(f"  Centro (r=0): {(1 - CA_profile[-1, 0]/lfr.CA0)*100:.2f} %")
    print(f"  Medio (r=R/2): {(1 - CA_profile[-1, len(r)//2]/lfr.CA0)*100:.2f} %")
    print(f"  Pared (r=R): {(1 - CA_profile[-1, -1]/lfr.CA0)*100:.2f} %")
    
    # ===========================================
    # ESTUDIO DE SENSIBILIDAD
    # ===========================================
    def sensitivity_analysis():
        """Análisis de sensibilidad a parámetros clave"""
        print("\n" + "="*60)
        print("ANÁLISIS DE SENSIBILIDAD")
        print("="*60)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Sensibilidad a la velocidad
        u_values = [0.05, 0.1, 0.2, 0.5]
        colors = ['red', 'blue', 'green', 'purple']
        
        for u_val, color in zip(u_values, colors):
            lfr_temp = LaminarFlowReactor()
            lfr_temp.u_avg = u_val
            z_temp, r_temp, CA_temp, u_temp = lfr_temp.solve_radial_profile()
            conv_temp = lfr_temp.conversion_radial_average(CA_temp, r_temp)
            
            axes[0, 0].plot(z_temp, conv_temp * 100, color=color, 
                           linewidth=2, label=f'u = {u_val} m/s')
            
            # Calcular conversión final
            final_conv = conv_temp[-1] * 100
            print(f"Velocidad {u_val} m/s -> Conversión final: {final_conv:.2f} %")
        
        axes[0, 0].set_xlabel('Posición Axial [m]')
        axes[0, 0].set_ylabel('Conversión [%]')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_title('Sensibilidad a Velocidad Promedio')
        
        # Sensibilidad a constante cinética
        k_values = [0.1, 0.5, 1.0, 2.0]
        for k_val, color in zip(k_values, colors):
            lfr_temp = LaminarFlowReactor()
            lfr_temp.k = k_val
            z_temp, r_temp, CA_temp, u_temp = lfr_temp.solve_radial_profile()
            conv_temp = lfr_temp.conversion_radial_average(CA_temp, r_temp)
            
            axes[0, 1].plot(z_temp, conv_temp * 100, color=color, 
                           linewidth=2, label=f'k = {k_val} 1/s')
        
        axes[0, 1].set_xlabel('Posición Axial [m]')
        axes[0, 1].set_ylabel('Conversión [%]')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_title('Sensibilidad a Constante Cinética')
        
        # Sensibilidad a radio del reactor
        R_values = [0.005, 0.01, 0.02, 0.03]
        for R_val, color in zip(R_values, colors):
            lfr_temp = LaminarFlowReactor()
            lfr_temp.R = R_val
            z_temp, r_temp, CA_temp, u_temp = lfr_temp.solve_radial_profile()
            conv_temp = lfr_temp.conversion_radial_average(CA_temp, r_temp)
            
            axes[1, 0].plot(z_temp, conv_temp * 100, color=color, 
                           linewidth=2, label=f'R = {R_val*1000} mm')
        
        axes[1, 0].set_xlabel('Posición Axial [m]')
        axes[1, 0].set_ylabel('Conversión [%]')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_title('Sensibilidad a Radio del Reactor')
        
        # Comparación con PFR ideal
        z_pfr = np.linspace(0, lfr.L, 100)
        conversion_pfr_curve = 1 - np.exp(-lfr.k * z_pfr / lfr.u_avg)
        axes[1, 1].plot(z_pfr, conversion_pfr_curve * 100, 'k--', 
                       linewidth=3, label='PFR Ideal')
        axes[1, 1].plot(z, conversion_avg * 100, 'r-', 
                       linewidth=2, label='LFR Real')
        axes[1, 1].set_xlabel('Posición Axial [m]')
        axes[1, 1].set_ylabel('Conversión [%]')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_title('Comparación LFR vs PFR Ideal')
        
        plt.tight_layout()
        plt.show()
    
    sensitivity_analysis()

if __name__ == "__main__":
    main()