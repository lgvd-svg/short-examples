import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import math

class DimensionlessHexagonalFEM:
    """
    Elementos Finitos Hexagonales Adimensionales para Ecuación de Calor
    Ecuación: -∇*²T* = f* en Ω*
    donde: 
    T* = T/ΔT_ref (Temperatura adimensional)
    x* = x/L (Coordenadas adimensionales)
    f* = fL²/(kΔT_ref) (Fuente adimensional)
    """
    
    def __init__(self, Lx_star=1.0, Ly_star=1.0, nx=10, ny=10):
        # Dimensiones adimensionales del dominio
        self.Lx_star = Lx_star  # Lx/L_ref = 1
        self.Ly_star = Ly_star  # Ly/L_ref = 1
        
        self.nx = nx
        self.ny = ny
        
        # Números adimensionales
        self.Biot = 0.0  # Número de Biot (para condiciones convectivas)
        self.Nusselt = 1.0  # Número de Nusselt
        
        # Generar malla hexagonal adimensional
        self.nodes, self.elements = self.generate_dimensionless_hexagonal_mesh()
        self.n_nodes = len(self.nodes)
        self.n_elements = len(self.elements)
        
    def generate_dimensionless_hexagonal_mesh(self):
        """
        Genera malla hexagonal en coordenadas adimensionales
        x* = x/L_ref, y* = y/L_ref
        """
        nodes = []
        elements = []
        
        # Espaciado adimensional
        dx_star = self.Lx_star / self.nx
        dy_star = self.Ly_star / self.ny
        h_star = dx_star
        r_star = h_star / math.sqrt(3)  # Radio adimensional
        
        # Usaremos una aproximación triangular para la malla hexagonal
        for i in range(self.nx + 1):
            for j in range(self.ny + 1):
                x_star = i * dx_star
                # Desplazamiento para patrón hexagonal
                if j % 2 == 1:
                    x_star += dx_star / 2
                y_star = j * dy_star * math.sqrt(3) / 2
                
                # Asegurar que esté dentro del dominio adimensional
                if x_star <= self.Lx_star and y_star <= self.Ly_star:
                    nodes.append([x_star, y_star])
        
        nodes = np.array(nodes)
        
        # Crear elementos triangulares (aproximación hexagonal)
        for i in range(self.nx):
            for j in range(self.ny):
                # Índices de nodos
                n1 = i * (self.ny + 1) + j
                n2 = n1 + 1
                n3 = (i + 1) * (self.ny + 1) + j
                n4 = n3 + 1
                
                # Verificar que los índices sean válidos
                if n4 < len(nodes):
                    # Dos triángulos formando un rombo (aproximación hexagonal)
                    elements.append([n1, n2, n3])
                    elements.append([n2, n4, n3])
        
        return nodes, np.array(elements)
    
    def dimensionless_shape_functions(self, element_nodes, xi, eta):
        """
        Funciones de forma adimensionales para elemento triangular
        Coordenadas naturales ξ, η ∈ [0,1]
        """
        N = np.zeros(3)
        N[0] = 1 - xi - eta  # N1
        N[1] = xi           # N2  
        N[2] = eta          # N3
        
        return N
    
    def dimensionless_jacobian(self, element_nodes):
        """
        Jacobiano adimensional para elemento triangular
        J* = ∂(x*,y*)/∂(ξ,η)
        """
        x1, y1 = element_nodes[0]
        x2, y2 = element_nodes[1]
        x3, y3 = element_nodes[2]
        
        # Jacobiano constante para elementos triangulares lineales
        J11 = x2 - x1  # ∂x*/∂ξ
        J12 = x3 - x1  # ∂x*/∂η
        J21 = y2 - y1  # ∂y*/∂ξ
        J22 = y3 - y1  # ∂y*/∂η
        
        J = np.array([[J11, J12],
                      [J21, J22]])
        
        detJ = np.linalg.det(J)
        
        return J, detJ
    
    def dimensionless_element_stiffness(self, element_nodes):
        """
        Matriz de rigidez adimensional local
        k*_ij = ∫(∇N_i · ∇N_j) dΩ*
        """
        k_star = np.zeros((3, 3))
        
        J, detJ = self.dimensionless_jacobian(element_nodes)
        
        if abs(detJ) < 1e-12:
            return k_star
        
        # Matriz de derivadas en coordenadas naturales
        dN_dxi = np.array([[-1, 1, 0],   # ∂N/∂ξ
                          [-1, 0, 1]])   # ∂N/∂η
        
        # Matriz B = [∂N/∂x*, ∂N/∂y*]^T
        invJ = np.linalg.inv(J)
        B = invJ @ dN_dxi
        
        # Matriz de rigidez adimensional
        k_star = B.T @ B * abs(detJ) * 0.5  # Área del elemento triangular
        
        return k_star
    
    def dimensionless_element_load(self, element_nodes, f_star=0.0):
        """
        Vector de carga adimensional local
        f*_i = ∫(f* · N_i) dΩ*
        """
        f_star_vec = np.zeros(3)
        
        if abs(f_star) < 1e-12:
            return f_star_vec
        
        J, detJ = self.dimensionless_jacobian(element_nodes)
        
        if abs(detJ) < 1e-12:
            return f_star_vec
        
        # Integración numérica (regla del punto medio para triángulo)
        area_star = abs(detJ) * 0.5
        N_mid = np.array([1/3, 1/3, 1/3])  # Funciones de forma en el centroide
        
        f_star_vec = f_star * N_mid * area_star
        
        return f_star_vec
    
    def dimensionless_convective_boundary(self, boundary_nodes, h_star=1.0, T_inf_star=0.0):
        """
        Condición de frontera convectiva adimensional
        h* = hL/k (Número de Biot)
        """
        n_boundary = len(boundary_nodes)
        k_conv_star = np.zeros((n_boundary, n_boundary))
        f_conv_star = np.zeros(n_boundary)
        
        # Para elementos de frontera lineales
        for i in range(n_boundary - 1):
            node1 = boundary_nodes[i]
            node2 = boundary_nodes[i + 1]
            
            # Longitud adimensional del segmento
            x1, y1 = self.nodes[node1]
            x2, y2 = self.nodes[node2]
            L_star = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            # Matriz de convección local
            k_local = np.array([[2, 1],
                              [1, 2]]) * L_star * h_star / 6
            
            # Vector de convección local
            f_local = np.array([1, 1]) * L_star * h_star * T_inf_star / 2
            
            # Ensamblar
            k_conv_star[i:i+2, i:i+2] += k_local
            f_conv_star[i:i+2] += f_local
        
        return k_conv_star, f_conv_star
    
    def assemble_dimensionless_system(self, f_star=0.0):
        """
        Ensambla el sistema global adimensional K* * T* = F*
        """
        K_star = lil_matrix((self.n_nodes, self.n_nodes))
        F_star = np.zeros(self.n_nodes)
        
        print(f"Ensamblando sistema adimensional: {self.n_elements} elementos, {self.n_nodes} nodos")
        
        for element in self.elements:
            element_nodes = self.nodes[element]
            
            # Matriz de rigidez adimensional
            k_element = self.dimensionless_element_stiffness(element_nodes)
            
            # Vector de carga adimensional
            f_element = self.dimensionless_element_load(element_nodes, f_star)
            
            # Ensamblar
            for i, node_i in enumerate(element):
                for j, node_j in enumerate(element):
                    K_star[node_i, node_j] += k_element[i, j]
                F_star[node_i] += f_element[i]
        
        return csr_matrix(K_star), F_star
    
    def apply_dimensionless_boundary_conditions(self, K_star, F_star, 
                                              T_left_star=1.0, T_right_star=0.0,
                                              h_star=0.0, T_inf_star=0.0):
        """
        Aplica condiciones de frontera adimensionales
        """
        K_mod = K_star.copy().tolil()
        F_mod = F_star.copy()
        
        # Identificar fronteras
        left_nodes = []
        right_nodes = []
        top_nodes = []
        bottom_nodes = []
        
        tol = 1e-10
        
        for i, node in enumerate(self.nodes):
            x_star, y_star = node
            
            if abs(x_star - 0) < tol:
                left_nodes.append(i)
            elif abs(x_star - self.Lx_star) < tol:
                right_nodes.append(i)
            elif abs(y_star - 0) < tol:
                bottom_nodes.append(i)
            elif abs(y_star - self.Ly_star) < tol:
                top_nodes.append(i)
        
        print(f"Fronteras: {len(left_nodes)} izq, {len(right_nodes)} der, "
              f"{len(top_nodes)} sup, {len(bottom_nodes)} inf")
        
        # Condiciones Dirichlet en fronteras izquierda y derecha
        for node in left_nodes:
            K_mod[node, :] = 0
            K_mod[:, node] = 0
            K_mod[node, node] = 1
            F_mod[node] = T_left_star
        
        for node in right_nodes:
            K_mod[node, :] = 0
            K_mod[:, node] = 0
            K_mod[node, node] = 1
            F_mod[node] = T_right_star
        
        # Condiciones convectivas en fronteras superior e inferior (si h* > 0)
        if h_star > 0:
            if len(top_nodes) > 1:
                k_conv_top, f_conv_top = self.dimensionless_convective_boundary(
                    top_nodes, h_star, T_inf_star)
                for i, node_i in enumerate(top_nodes):
                    for j, node_j in enumerate(top_nodes):
                        K_mod[node_i, node_j] += k_conv_top[i, j]
                    F_mod[node_i] += f_conv_top[i]
            
            if len(bottom_nodes) > 1:
                k_conv_bottom, f_conv_bottom = self.dimensionless_convective_boundary(
                    bottom_nodes, h_star, T_inf_star)
                for i, node_i in enumerate(bottom_nodes):
                    for j, node_j in enumerate(bottom_nodes):
                        K_mod[node_i, node_j] += k_conv_bottom[i, j]
                    F_mod[node_i] += f_conv_bottom[i]
        
        return K_mod.tocsr(), F_mod
    
    def solve_dimensionless(self, f_star=0.0, T_left_star=1.0, T_right_star=0.0,
                          h_star=0.0, T_inf_star=0.0):
        """
        Resuelve el problema adimensional completo
        """
        # Ensamblar sistema
        K_star, F_star = self.assemble_dimensionless_system(f_star)
        
        # Aplicar condiciones de frontera
        K_bc, F_bc = self.apply_dimensionless_boundary_conditions(
            K_star, F_star, T_left_star, T_right_star, h_star, T_inf_star)
        
        # Resolver
        print("Resolviendo sistema lineal adimensional...")
        T_star = spsolve(K_bc, F_bc)
        
        return T_star
    
    def calculate_dimensionless_flux(self, T_star):
        """
        Calcula el flujo de calor adimensional q* = -∇*T*
        """
        flux_x_star = np.zeros(self.n_nodes)
        flux_y_star = np.zeros(self.n_nodes)
        
        for element in self.elements:
            element_nodes = self.nodes[element]
            element_T = T_star[element]
            
            if len(element) == 3:  # Elemento triangular
                x1, y1 = element_nodes[0]
                x2, y2 = element_nodes[1]
                x3, y3 = element_nodes[2]
                
                T1, T2, T3 = element_T
                
                # Área adimensional
                area_star = 0.5 * abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
                
                if area_star > 1e-12:
                    # Gradiente adimensional constante en el elemento
                    dT_dx_star = (T1*(y2-y3) + T2*(y3-y1) + T3*(y1-y2)) / (2*area_star)
                    dT_dy_star = (T1*(x3-x2) + T2*(x1-x3) + T3*(x2-x1)) / (2*area_star)
                    
                    # Flujo adimensional (ley de Fourier adimensional)
                    for node in element:
                        flux_x_star[node] += -dT_dx_star
                        flux_y_star[node] += -dT_dy_star
        
        # Promediar en nodos
        node_count = np.zeros(self.n_nodes)
        for element in self.elements:
            for node in element:
                node_count[node] += 1
        
        for i in range(self.n_nodes):
            if node_count[i] > 0:
                flux_x_star[i] /= node_count[i]
                flux_y_star[i] /= node_count[i]
        
        return flux_x_star, flux_y_star

def plot_dimensionless_solution(nodes, elements, T_star, flux_x_star=None, flux_y_star=None):
    """Visualiza la solución adimensional"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Malla adimensional
    ax1 = axes[0, 0]
    for element in elements:
        if len(element) == 3:
            triangle = np.array([nodes[element[0]], 
                              nodes[element[1]], 
                              nodes[element[2]]])
            tri_patch = plt.Polygon(triangle, fill=False, edgecolor='blue', alpha=0.6, linewidth=0.5)
            ax1.add_patch(tri_patch)
    
    ax1.plot(nodes[:, 0], nodes[:, 1], 'ro', markersize=1, alpha=0.6)
    ax1.set_xlabel('x* (Posición adimensional)')
    ax1.set_ylabel('y* (Posición adimensional)')
    ax1.set_title('Malla Hexagonal Adimensional')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # 2. Campo de temperatura adimensional
    ax2 = axes[0, 1]
    if len(elements) > 0 and len(elements[0]) == 3:
        triang = tri.Triangulation(nodes[:, 0], nodes[:, 1], elements)
        contour = ax2.tricontourf(triang, T_star, levels=50, cmap='jet')
        plt.colorbar(contour, ax=ax2, label='T* (Temperatura adimensional)')
        ax2.tricontour(triang, T_star, levels=10, colors='k', linewidths=0.5, alpha=0.5)
    
    ax2.set_xlabel('x*')
    ax2.set_ylabel('y*')
    ax2.set_title('Campo de Temperatura Adimensional T*')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # 3. Líneas de contorno
    ax3 = axes[1, 0]
    if len(elements) > 0 and len(elements[0]) == 3:
        contour = ax3.tricontour(triang, T_star, levels=15, colors='k', linewidths=1)
        ax3.tricontourf(triang, T_star, levels=50, cmap='jet', alpha=0.7)
        ax3.clabel(contour, inline=True, fontsize=8)
        ax3.plot(nodes[:, 0], nodes[:, 1], 'o', markersize=1, alpha=0.3)
    
    ax3.set_xlabel('x*')
    ax3.set_ylabel('y*')
    ax3.set_title('Líneas de Contorno de T*')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    # 4. Flujo de calor adimensional
    ax4 = axes[1, 1]
    if flux_x_star is not None and flux_y_star is not None:
        flux_magnitude_star = np.sqrt(flux_x_star**2 + flux_y_star**2)
        
        if len(elements) > 0 and len(elements[0]) == 3:
            scatter = ax4.scatter(nodes[:, 0], nodes[:, 1], c=flux_magnitude_star,
                                cmap='hot', s=10, alpha=0.7)
            plt.colorbar(scatter, ax=ax4, label='|q*| (Flujo adimensional)')
        
        # Vectores de flujo (muestreados)
        skip = max(1, len(nodes) // 50)
        ax4.quiver(nodes[::skip, 0], nodes[::skip, 1], 
                  flux_x_star[::skip], flux_y_star[::skip], 
                  flux_magnitude_star[::skip], cmap='hot',
                  scale=20, width=0.003, alpha=0.7)
        
        ax4.set_xlabel('x*')
        ax4.set_ylabel('y*')
        ax4.set_title('Campo de Flujo de Calor Adimensional q*')
        ax4.set_aspect('equal')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Flujo adimensional no calculado', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Campo de Flujo de Calor Adimensional')
    
    plt.tight_layout()
    plt.show()

def dimensionless_convergence_study():
    """Estudio de convergencia adimensional"""
    mesh_sizes = [5, 10, 15, 20, 25]
    errors = []
    computational_times = []  # Tiempo relativo
    
    plt.figure(figsize=(12, 5))
    
    for idx, nx in enumerate(mesh_sizes):
        print(f"\n--- Malla {nx}×{nx} ---")
        
        # Crear y resolver problema adimensional
        fem = DimensionlessHexagonalFEM(nx=nx, ny=nx)
        T_star = fem.solve_dimensionless(T_left_star=1.0, T_right_star=0.0)
        
        # Solución analítica adimensional para problema 1D: T*(x*) = 1 - x*
        T_analytical_star = 1.0 - fem.nodes[:, 0]
        
        # Error RMS adimensional
        error_star = np.sqrt(np.mean((T_star - T_analytical_star)**2))
        errors.append(error_star)
        
        # Tiempo computacional relativo (usando número de nodos como proxy)
        computational_times.append(fem.n_nodes)
        
        print(f"Nodos: {fem.n_nodes}, Error: {error_star:.6f}")
    
    # Gráfica de convergencia
    plt.subplot(1, 2, 1)
    plt.loglog(mesh_sizes, errors, 'bo-', linewidth=2, markersize=8, label='Error RMS')
    plt.xlabel('1/h* (Refinamiento de malla)')
    plt.ylabel('Error RMS adimensional')
    plt.title('Convergencia del Error Adimensional')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Gráfica de eficiencia computacional
    plt.subplot(1, 2, 2)
    plt.loglog(computational_times, errors, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Número de Nodos (Coste computacional)')
    plt.ylabel('Error RMS adimensional')
    plt.title('Eficiencia Computacional')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return mesh_sizes, errors

def analyze_dimensionless_parameters():
    """Análisis de sensibilidad a parámetros adimensionales"""
    print("\n" + "="*60)
    print("ANÁLISIS DE PARÁMETROS ADIMENSIONALES")
    print("="*60)
    
    # Diferentes valores de fuente adimensional
    f_star_values = [0.0, 10.0, 50.0, 100.0]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, f_star in enumerate(f_star_values):
        if idx >= len(axes):
            break
            
        fem = DimensionlessHexagonalFEM(nx=12, ny=12)
        T_star = fem.solve_dimensionless(f_star=f_star, T_left_star=1.0, T_right_star=0.0)
        
        if len(fem.elements) > 0 and len(fem.elements[0]) == 3:
            triang = tri.Triangulation(fem.nodes[:, 0], fem.nodes[:, 1], fem.elements)
            contour = axes[idx].tricontourf(triang, T_star, levels=20, cmap='jet')
            plt.colorbar(contour, ax=axes[idx])
            
            T_max = np.max(T_star)
            T_min = np.min(T_star)
            axes[idx].set_title(f'f* = {f_star}\nT*_max = {T_max:.3f}, T*_min = {T_min:.3f}')
            axes[idx].set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    
    # Análisis de número de Biot
    print("\n--- Análisis del Número de Biot ---")
    Biot_values = [0.1, 1.0, 10.0, 100.0]  # h* = hL/k
    
    plt.figure(figsize=(12, 5))
    
    for Biot in Biot_values:
        fem = DimensionlessHexagonalFEM(nx=15, ny=15)
        T_star = fem.solve_dimensionless(T_left_star=1.0, T_right_star=0.0,
                                       h_star=Biot, T_inf_star=0.0)
        
        # Perfil a lo largo de y* = 0.5
        mid_y = 0.5
        mid_nodes = [i for i, node in enumerate(fem.nodes) if abs(node[1] - mid_y) < 0.01]
        if mid_nodes:
            x_mid = [fem.nodes[i, 0] for i in mid_nodes]
            T_mid = [T_star[i] for i in mid_nodes]
            
            # Ordenar por x*
            sorted_indices = np.argsort(x_mid)
            x_sorted = np.array(x_mid)[sorted_indices]
            T_sorted = np.array(T_mid)[sorted_indices]
            
            plt.plot(x_sorted, T_sorted, linewidth=2, label=f'Biot = {Biot}')
    
    plt.xlabel('x*')
    plt.ylabel('T*')
    plt.title('Efecto del Número de Biot en el Perfil de Temperatura')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    # ===========================================
    # SOLUCIÓN ADIMENSIONAL PRINCIPAL
    # ===========================================
    print("=" * 70)
    print("ELEMENTOS FINITOS HEXAGONALES - FORMULACIÓN ADIMENSIONAL")
    print("=" * 70)
    
    print("\nVARIABLES ADIMENSIONALES:")
    print("x* = x/L_ref, y* = y/L_ref")
    print("T* = T/ΔT_ref")
    print("f* = f·L_ref²/(k·ΔT_ref)")
    print("q* = q·L_ref/(k·ΔT_ref)")
    print("h* = h·L_ref/k (Número de Biot)")
    
    # Caso 1: Conducción pura con fuentes
    print("\n" + "="*50)
    print("CASO 1: CONDUCCIÓN CON FUENTE DE CALOR")
    print("="*50)
    
    fem1 = DimensionlessHexagonalFEM(nx=20, ny=20)
    T_star1 = fem1.solve_dimensionless(f_star=25.0, T_left_star=1.0, T_right_star=0.0)
    flux_x_star1, flux_y_star1 = fem1.calculate_dimensionless_flux(T_star1)
    
    print(f"Resultados Caso 1:")
    print(f"T*_máx = {np.max(T_star1):.4f}")
    print(f"T*_mín = {np.min(T_star1):.4f}")
    print(f"T*_prom = {np.mean(T_star1):.4f}")
    print(f"|q*|_máx = {np.max(np.sqrt(flux_x_star1**2 + flux_y_star1**2)):.4f}")
    
    plot_dimensionless_solution(fem1.nodes, fem1.elements, T_star1, flux_x_star1, flux_y_star1)
    
    # Caso 2: Convección en fronteras
    print("\n" + "="*50)
    print("CASO 2: TRANSFERENCIA DE CALOR CON CONVECCIÓN")
    print("="*50)
    
    fem2 = DimensionlessHexagonalFEM(nx=18, ny=18)
    T_star2 = fem2.solve_dimensionless(T_left_star=1.0, T_right_star=0.0,
                                     h_star=5.0, T_inf_star=0.2)
    
    print(f"Resultados Caso 2 (Biot = 5.0):")
    print(f"T*_máx = {np.max(T_star2):.4f}")
    print(f"T*_mín = {np.min(T_star2):.4f}")
    
    # ===========================================
    # ESTUDIOS NUMÉRICOS
    # ===========================================
    print("\n" + "="*50)
    print("ESTUDIO DE CONVERGENCIA ADIMENSIONAL")
    print("="*50)
    
    mesh_sizes, errors = dimensionless_convergence_study()
    
    # Análisis de parámetros adimensionales
    analyze_dimensionless_parameters()
    
    # ===========================================
    # INTERPRETACIÓN FÍSICA
    # ===========================================
    print("\n" + "="*50)
    print("INTERPRETACIÓN FÍSICA DE RESULTADOS ADIMENSIONALES")
    print("="*50)
    
    print("\nPara convertir a variables dimensionales:")
    print("T = T* × ΔT_ref + T_ref")
    print("x = x* × L_ref") 
    print("q = q* × (k·ΔT_ref/L_ref)")
    print("f = f* × (k·ΔT_ref/L_ref²)")
    
    print("\nEjemplo con valores típicos:")
    print("L_ref = 0.1 m, ΔT_ref = 100 K, k = 50 W/m·K")
    print(f"T* = {T_star1[0]:.3f} → T = {T_star1[0] * 100:.1f} °C")
    print(f"|q*| = {np.sqrt(flux_x_star1[0]**2 + flux_y_star1[0]**2):.3f} → |q| = {np.sqrt(flux_x_star1[0]**2 + flux_y_star1[0]**2) * (50 * 100 / 0.1):.1f} W/m²")
    print(f"f* = 25.0 → f = {25.0 * (50 * 100 / 0.1**2):.0f} W/m³")

if __name__ == "__main__":
    main()
