import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.tri as tri

class MeshfreeGalerkinHeat:
    """
    Meshfree Galerkin (Element Free Galerkin) para Ecuación de Calor Estacionaria
    -∇·(k∇T) = f en Ω
    T = g en ∂Ω
    """
    
    def __init__(self, Lx=1.0, Ly=1.0, num_nodes=400, support_radius=0.15):
        self.Lx = Lx
        self.Ly = Ly
        self.num_nodes = num_nodes
        self.support_radius = support_radius
        
        # Parámetros de forma
        self.k = 1.0  # Conductividad térmica
        
        # Generar nodos distribuidos aleatoriamente
        self.nodes = self.generate_random_nodes()
        self.n_nodes = len(self.nodes)
        
        # Construir KDTree para búsqueda eficiente de vecinos
        self.kdtree = KDTree(self.nodes)
        
        # Identificar nodos de frontera
        self.boundary_nodes = self.identify_boundary_nodes()
        self.internal_nodes = self.identify_internal_nodes()
        
        # Parámetros MLS (Moving Least Squares)
        self.basis_type = 'linear'  # 'linear', 'quadratic'
        
    def generate_random_nodes(self):
        """Genera nodos distribuidos aleatoriamente en el dominio"""
        np.random.seed(42)  # Para reproducibilidad
        
        nodes = []
        attempts = 0
        max_attempts = self.num_nodes * 10
        
        while len(nodes) < self.num_nodes and attempts < max_attempts:
            x = np.random.uniform(0, self.Lx)
            y = np.random.uniform(0, self.Ly)
            
            # Verificar distancia mínima
            if len(nodes) == 0:
                nodes.append([x, y])
            else:
                min_dist = np.min(np.linalg.norm(np.array(nodes) - np.array([x, y]), axis=1))
                if min_dist > 0.02:  # Distancia mínima entre nodos
                    nodes.append([x, y])
            
            attempts += 1
        
        # Si no se alcanza el número deseado, agregar nodos en grid
        if len(nodes) < self.num_nodes:
            additional_nodes = self.num_nodes - len(nodes)
            nx = int(np.sqrt(additional_nodes * self.Lx / self.Ly))
            ny = int(additional_nodes / nx)
            
            for i in range(nx):
                for j in range(ny):
                    x = (i + 0.5) * self.Lx / nx
                    y = (j + 0.5) * self.Ly / ny
                    nodes.append([x, y])
                    
                    if len(nodes) >= self.num_nodes:
                        break
                if len(nodes) >= self.num_nodes:
                    break
        
        return np.array(nodes)
    
    def identify_boundary_nodes(self):
        """Identifica nodos en la frontera del dominio"""
        boundary_nodes = []
        tol = 0.01 * min(self.Lx, self.Ly)
        
        for i, node in enumerate(self.nodes):
            x, y = node
            if (abs(x - 0) < tol or abs(x - self.Lx) < tol or 
                abs(y - 0) < tol or abs(y - self.Ly) < tol):
                boundary_nodes.append(i)
        
        return boundary_nodes
    
    def identify_internal_nodes(self):
        """Identifica nodos internos (no en la frontera)"""
        all_indices = set(range(self.n_nodes))
        boundary_set = set(self.boundary_nodes)
        internal_set = all_indices - boundary_set
        return list(internal_set)
    
    def mls_basis_functions(self, point, center_nodes):
        """
        Calcula las funciones de forma MLS para un punto dado
        """
        n_nodes = len(center_nodes)
        distances = np.linalg.norm(self.nodes[center_nodes] - point, axis=1)
        
        # Función de peso (weight function)
        weights = self.weight_function(distances, self.support_radius)
        
        # Matriz de momento
        if self.basis_type == 'linear':
            basis_size = 3  # 1, x, y
            P = np.ones((n_nodes, basis_size))
            P[:, 1] = self.nodes[center_nodes, 0] - point[0]
            P[:, 2] = self.nodes[center_nodes, 1] - point[1]
        else:  # quadratic
            basis_size = 6  # 1, x, y, x², xy, y²
            P = np.ones((n_nodes, basis_size))
            dx = self.nodes[center_nodes, 0] - point[0]
            dy = self.nodes[center_nodes, 1] - point[1]
            P[:, 1] = dx
            P[:, 2] = dy
            P[:, 3] = dx**2
            P[:, 4] = dx * dy
            P[:, 5] = dy**2
        
        # Matriz de pesos
        W = np.diag(weights)
        
        # Matriz de momento
        A = P.T @ W @ P
        
        # Vector p para el punto de evaluación
        if self.basis_type == 'linear':
            p_eval = np.array([1, 0, 0])
        else:
            p_eval = np.array([1, 0, 0, 0, 0, 0])
        
        try:
            # Calcular funciones de forma
            A_inv = np.linalg.inv(A + np.eye(A.shape[0]) * 1e-12)
            phi = p_eval @ A_inv @ P.T @ W
        except np.linalg.LinAlgError:
            # Si la matriz es singular, usar aproximación simple
            phi = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones(n_nodes) / n_nodes
        
        return phi
    
    def weight_function(self, r, R):
        """
        Función de peso MLS - función cónica
        """
        weights = np.zeros_like(r)
        mask = r <= R
        s = r[mask] / R
        weights[mask] = 1 - 6*s**2 + 8*s**3 - 3*s**4
        return weights
    
    def mls_shape_function_derivatives(self, point, center_nodes):
        """
        Calcula las derivadas de las funciones de forma MLS
        """
        n_nodes = len(center_nodes)
        distances = np.linalg.norm(self.nodes[center_nodes] - point, axis=1)
        weights = self.weight_function(distances, self.support_radius)
        
        # Derivadas de la función de peso
        dw_dx, dw_dy = self.weight_function_derivatives(point, center_nodes)
        
        if self.basis_type == 'linear':
            basis_size = 3
            P = np.ones((n_nodes, basis_size))
            P[:, 1] = self.nodes[center_nodes, 0] - point[0]
            P[:, 2] = self.nodes[center_nodes, 1] - point[1]
            
            dP_dx = np.zeros((n_nodes, basis_size))
            dP_dy = np.zeros((n_nodes, basis_size))
            dP_dx[:, 1] = -1
            dP_dy[:, 2] = -1
            
        else:  # quadratic
            basis_size = 6
            dx = self.nodes[center_nodes, 0] - point[0]
            dy = self.nodes[center_nodes, 1] - point[1]
            P = np.ones((n_nodes, basis_size))
            P[:, 1] = dx
            P[:, 2] = dy
            P[:, 3] = dx**2
            P[:, 4] = dx * dy
            P[:, 5] = dy**2
            
            dP_dx = np.zeros((n_nodes, basis_size))
            dP_dy = np.zeros((n_nodes, basis_size))
            dP_dx[:, 1] = -1
            dP_dy[:, 2] = -1
            dP_dx[:, 3] = -2 * dx
            dP_dx[:, 4] = -dy
            dP_dy[:, 4] = -dx
            dP_dy[:, 5] = -2 * dy
        
        W = np.diag(weights)
        dW_dx = np.diag(dw_dx)
        dW_dy = np.diag(dw_dy)
        
        A = P.T @ W @ P
        B_x = dP_dx.T @ W @ P + P.T @ dW_dx @ P + P.T @ W @ dP_dx
        B_y = dP_dy.T @ W @ P + P.T @ dW_dy @ P + P.T @ W @ dP_dy
        
        try:
            A_inv = np.linalg.inv(A + np.eye(A.shape[0]) * 1e-12)
            
            if self.basis_type == 'linear':
                p_eval = np.array([1, 0, 0])
                dp_dx = np.array([0, -1, 0])
                dp_dy = np.array([0, 0, -1])
            else:
                p_eval = np.array([1, 0, 0, 0, 0, 0])
                dp_dx = np.array([0, -1, 0, 0, 0, 0])
                dp_dy = np.array([0, 0, -1, 0, 0, 0])
            
            # Funciones de forma
            phi = p_eval @ A_inv @ P.T @ W
            
            # Derivadas de las funciones de forma
            dphi_dx = (dp_dx @ A_inv @ P.T @ W + 
                       p_eval @ (-A_inv @ B_x @ A_inv) @ P.T @ W +
                       p_eval @ A_inv @ dP_dx.T @ W +
                       p_eval @ A_inv @ P.T @ dW_dx)
            
            dphi_dy = (dp_dy @ A_inv @ P.T @ W + 
                       p_eval @ (-A_inv @ B_y @ A_inv) @ P.T @ W +
                       p_eval @ A_inv @ dP_dy.T @ W +
                       p_eval @ A_inv @ P.T @ dW_dy)
            
        except np.linalg.LinAlgError:
            # Aproximación simple si hay problemas numéricos
            phi = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones(n_nodes) / n_nodes
            dphi_dx = np.zeros(n_nodes)
            dphi_dy = np.zeros(n_nodes)
        
        return phi, dphi_dx, dphi_dy
    
    def weight_function_derivatives(self, point, center_nodes):
        """
        Calcula las derivadas de la función de peso
        """
        n_nodes = len(center_nodes)
        nodes_pos = self.nodes[center_nodes]
        
        distances = np.linalg.norm(nodes_pos - point, axis=1)
        dw_dx = np.zeros(n_nodes)
        dw_dy = np.zeros(n_nodes)
        
        R = self.support_radius
        mask = distances <= R
        
        if np.any(mask):
            r = distances[mask]
            s = r / R
            x_diff = nodes_pos[mask, 0] - point[0]
            y_diff = nodes_pos[mask, 1] - point[1]
            
            # Derivada de la función de peso
            dw_dr = (-12*s/R + 24*s**2/R - 12*s**3/R) * mask[mask]
            
            dw_dx[mask] = dw_dr * x_diff / (r + 1e-12)
            dw_dy[mask] = dw_dr * y_diff / (r + 1e-12)
        
        return dw_dx, dw_dy
    
    def assemble_system(self, source_term=0.0):
        """
        Ensambla el sistema global K * T = F usando Meshfree Galerkin
        """
        K_global = lil_matrix((self.n_nodes, self.n_nodes))
        F_global = np.zeros(self.n_nodes)
        
        print(f"Ensamblando sistema Meshfree Galerkin con {self.n_nodes} nodos...")
        
        # Puntos de integración (nodos como puntos de integración)
        integration_points = self.nodes[self.internal_nodes]
        
        for idx, point in enumerate(integration_points):
            if idx % 50 == 0:
                print(f"Procesando punto de integración {idx}/{len(integration_points)}")
            
            # Encontrar nodos en el soporte
            center_nodes = self.kdtree.query_ball_point(point, self.support_radius)
            
            if len(center_nodes) < 3:  # Mínimo para MLS
                continue
            
            # Funciones de forma y sus derivadas
            phi, dphi_dx, dphi_dy = self.mls_shape_function_derivatives(point, center_nodes)
            
            # Matriz B (gradientes)
            B = np.vstack([dphi_dx, dphi_dy])
            
            # Matriz de rigidez local
            k_local = B.T @ B * self.k
            
            # Ensamblar en matriz global
            for i, node_i in enumerate(center_nodes):
                for j, node_j in enumerate(center_nodes):
                    K_global[node_i, node_j] += k_local[i, j]
                
                # Término de fuente
                F_global[node_i] += source_term * phi[i]
        
        return csr_matrix(K_global), F_global
    
    def apply_boundary_conditions(self, K, F, T_left=100.0, T_right=0.0):
        """
        Aplica condiciones de Dirichlet en las fronteras
        """
        K_mod = K.copy().tolil()
        F_mod = F.copy()
        
        # Identificar nodos en fronteras específicas
        left_nodes = []
        right_nodes = []
        tol = 0.01 * min(self.Lx, self.Ly)
        
        for i in self.boundary_nodes:
            x, y = self.nodes[i]
            if abs(x - 0) < tol:
                left_nodes.append(i)
            elif abs(x - self.Lx) < tol:
                right_nodes.append(i)
        
        print(f"Aplicando condiciones: {len(left_nodes)} nodos izquierda, {len(right_nodes)} nodos derecha")
        
        # Condiciones Dirichlet
        for node in left_nodes:
            K_mod[node, :] = 0
            K_mod[:, node] = 0
            K_mod[node, node] = 1
            F_mod[node] = T_left
        
        for node in right_nodes:
            K_mod[node, :] = 0
            K_mod[:, node] = 0
            K_mod[node, node] = 1
            F_mod[node] = T_right
        
        return K_mod.tocsr(), F_mod
    
    def solve(self, source_term=0.0, T_left=100.0, T_right=0.0):
        """
        Resuelve el sistema de ecuaciones
        """
        # Ensamblar sistema
        K, F = self.assemble_system(source_term)
        
        # Aplicar condiciones de frontera
        K_bc, F_bc = self.apply_boundary_conditions(K, F, T_left, T_right)
        
        # Resolver
        print("Resolviendo sistema lineal...")
        T_solution = spsolve(K_bc, F_bc)
        
        return T_solution
    
    def calculate_flux(self, T_solution):
        """
        Calcula el flujo de heat q = -k * ∇T
        """
        flux_x = np.zeros(self.n_nodes)
        flux_y = np.zeros(self.n_nodes)
        
        for i in range(self.n_nodes):
            point = self.nodes[i]
            center_nodes = self.kdtree.query_ball_point(point, self.support_radius)
            
            if len(center_nodes) >= 3:
                _, dphi_dx, dphi_dy = self.mls_shape_function_derivatives(point, center_nodes)
                
                # Calcular gradiente
                dT_dx = np.dot(dphi_dx, T_solution[center_nodes])
                dT_dy = np.dot(dphi_dy, T_solution[center_nodes])
                
                flux_x[i] = -self.k * dT_dx
                flux_y[i] = -self.k * dT_dy
        
        return flux_x, flux_y

def plot_meshfree_solution(nodes, T_solution, flux_x=None, flux_y=None, boundary_nodes=None):
    """Visualiza la solución Meshfree"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Distribución de nodos
    ax1 = axes[0, 0]
    ax1.scatter(nodes[:, 0], nodes[:, 1], c='blue', s=10, alpha=0.6, label='Nodos')
    
    if boundary_nodes is not None:
        ax1.scatter(nodes[boundary_nodes, 0], nodes[boundary_nodes, 1], 
                   c='red', s=20, alpha=0.8, label='Frontera')
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Distribución de Nodos Meshfree')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # 2. Campo de temperatura
    ax2 = axes[0, 1]
    scatter = ax2.scatter(nodes[:, 0], nodes[:, 1], c=T_solution, 
                         cmap='jet', s=30, alpha=0.8)
    plt.colorbar(scatter, ax=ax2, label='Temperatura')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Campo de Temperatura - Meshfree')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # 3. Contornos de temperatura
    ax3 = axes[1, 0]
    # Crear malla para contornos (interpolación)
    xi = np.linspace(0, 1, 100)
    yi = np.linspace(0, 1, 100)
    Xi, Yi = np.meshgrid(xi, yi)
    
    from scipy.interpolate import griddata
    Zi = griddata(nodes, T_solution, (Xi, Yi), method='cubic')
    
    contour = ax3.contourf(Xi, Yi, Zi, levels=50, cmap='jet')
    ax3.contour(Xi, Yi, Zi, levels=10, colors='k', linewidths=0.5, alpha=0.5)
    plt.colorbar(contour, ax=ax3, label='Temperatura')
    ax3.scatter(nodes[:, 0], nodes[:, 1], c='k', s=1, alpha=0.3)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Contornos de Temperatura (Interpolados)')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    # 4. Flujo de calor
    ax4 = axes[1, 1]
    if flux_x is not None and flux_y is not None:
        flux_magnitude = np.sqrt(flux_x**2 + flux_y**2)
        scatter_flux = ax4.scatter(nodes[:, 0], nodes[:, 1], c=flux_magnitude,
                                 cmap='hot', s=20, alpha=0.8)
        plt.colorbar(scatter_flux, ax=ax4, label='|Flujo de Calor|')
        
        # Vectores de flujo
        skip = max(1, len(nodes) // 50)
        ax4.quiver(nodes[::skip, 0], nodes[::skip, 1], 
                  flux_x[::skip], flux_y[::skip], 
                  flux_magnitude[::skip], cmap='hot',
                  scale=100, width=0.005, alpha=0.7)
        
        ax4.set_xlabel('x')
        ax4.set_ylabel('y')
        ax4.set_title('Campo de Flujo de Calor')
        ax4.set_aspect('equal')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Flujo no calculado', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Campo de Flujo de Calor')
    
    plt.tight_layout()
    plt.show()

def convergence_study_meshfree():
    """Estudio de convergencia para método Meshfree"""
    node_counts = [100, 200, 400, 600]
    errors = []
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, n_nodes in enumerate(node_counts):
        if idx >= len(axes):
            break
            
        print(f"\n--- Meshfree con {n_nodes} nodos ---")
        
        # Radio de soporte proporcional al espaciado de nodos
        support_radius = 0.15 * (400 / n_nodes)**0.5
        
        meshfree = MeshfreeGalerkinHeat(num_nodes=n_nodes, support_radius=support_radius)
        T_solution = meshfree.solve(T_left=100.0, T_right=0.0)
        
        # Solución analítica 1D
        T_analytical = 100 - 100 * meshfree.nodes[:, 0]
        
        # Error
        error = np.sqrt(np.mean((T_solution - T_analytical)**2))
        errors.append(error)
        
        # Graficar
        axes[idx].scatter(meshfree.nodes[:, 0], meshfree.nodes[:, 1], c=T_solution, 
                         cmap='jet', s=10, alpha=0.8)
        axes[idx].set_title(f'{n_nodes} nodos\nError RMS: {error:.4f}')
        axes[idx].set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    
    # Gráfica de convergencia
    plt.figure(figsize=(10, 6))
    plt.loglog(node_counts, errors, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Número de Nodos')
    plt.ylabel('Error RMS')
    plt.title('Convergencia del Método Meshfree')
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    # ===========================================
    # SOLUCIÓN PRINCIPAL MESHFREE
    # ===========================================
    print("=" * 60)
    print("MESHFREE GALERKIN - ECUACIÓN DE CALOR ESTACIONARIA")
    print("=" * 60)
    
    # Crear solver meshfree
    meshfree = MeshfreeGalerkinHeat(num_nodes=400, support_radius=0.15)
    
    print(f"Dominio: {meshfree.Lx} x {meshfree.Ly}")
    print(f"Nodos totales: {meshfree.n_nodes}")
    print(f"Nodos internos: {len(meshfree.internal_nodes)}")
    print(f"Nodos frontera: {len(meshfree.boundary_nodes)}")
    print(f"Radio de soporte: {meshfree.support_radius}")
    print(f"Función base: {meshfree.basis_type}")
    
    # Resolver ecuación de calor
    T_solution = meshfree.solve(source_term=0.0, T_left=100.0, T_right=0.0)
    
    # Calcular flujo de calor
    flux_x, flux_y = meshfree.calculate_flux(T_solution)
    
    # Resultados
    print(f"\nRESULTADOS:")
    print(f"Temperatura mínima: {np.min(T_solution):.2f} °C")
    print(f"Temperatura máxima: {np.max(T_solution):.2f} °C")
    print(f"Temperatura promedio: {np.mean(T_solution):.2f} °C")
    
    # Visualizar
    plot_meshfree_solution(meshfree.nodes, T_solution, flux_x, flux_y, 
                          meshfree.boundary_nodes)
    
    # ===========================================
    # ESTUDIO DE CONVERGENCIA
    # ===========================================
    print("\n" + "=" * 60)
    print("ESTUDIO DE CONVERGENCIA")
    print("=" * 60)
    
    convergence_study_meshfree()
    
    # ===========================================
    # CASO CON FUENTE DE CALOR
    # ===========================================
    print("\n" + "=" * 60)
    print("CASO CON FUENTE DE CALOR INTERNA")
    print("=" * 60)
    
    meshfree_source = MeshfreeGalerkinHeat(num_nodes=400, support_radius=0.15)
    T_source = meshfree_source.solve(source_term=50.0, T_left=50.0, T_right=0.0)
    
    print(f"Temperatura con fuente:")
    print(f"Mín: {np.min(T_source):.2f}, Máx: {np.max(T_source):.2f}")
    
    # Visualizar caso con fuente - CORREGIDO
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Gráfica 1: Campo de temperatura
    scatter1 = ax1.scatter(meshfree_source.nodes[:, 0], meshfree_source.nodes[:, 1], 
               c=T_source, cmap='jet', s=20, alpha=0.8)
    plt.colorbar(scatter1, ax=ax1, label='Temperatura')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Temperatura con Fuente de Calor - Meshfree')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')  # CORREGIDO: usar set_aspect en el axes
    
    # Gráfica 2: Perfil de temperatura
    mid_mask = np.abs(meshfree_source.nodes[:, 1] - 0.5) < 0.05
    if np.any(mid_mask):
        x_mid = meshfree_source.nodes[mid_mask, 0]
        T_mid = T_source[mid_mask]
        sorted_idx = np.argsort(x_mid)
        ax2.plot(x_mid[sorted_idx], T_mid[sorted_idx], 'r-', linewidth=2)
        ax2.set_xlabel('x')
        ax2.set_ylabel('Temperatura')
        ax2.set_title('Perfil en y ≈ 0.5')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
