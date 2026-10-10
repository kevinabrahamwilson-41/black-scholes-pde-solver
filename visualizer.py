import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
import pyqtgraph.opengl as gl
import config
import mesh
import payoff
import engine

class BlackScholesVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time 2D Black-Scholes PDE Surface Solver")
        self.setGeometry(100, 100, 800, 600)

        # Create basic container layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Initialize the hardware-accelerated 3D OpenGL Viewport
        self.viewer = gl.GLViewWidget()
        self.layout.addWidget(self.viewer)
        self.viewer.setCameraPosition(distance=400, elevation=30, azimuth=45)

        # Initialize spatial grids (convert coordinates to numpy arrays for presentation)
        self.s_x, self.s_y = mesh.generate_spatial_domains()
        self.S1, self.S2 = mesh.compute_tensor_mesh()
        self.dx = self.s_x[1] - self.s_x[0]
        self.dy = self.s_y[1] - self.s_y[0]
        self.dt, _ = mesh.generate_temporal_mesh()

        # Extract terminal option pricing vector profiles
        self.V = payoff.evaluate_basket_call(self.S1, self.S2)
        
        # Draw the initial 3D Surface grid layout mesh
        self.surface = gl.GLSurfacePlotItem(
            x=np.array(self.s_x), 
            y=np.array(self.s_y), 
            z=np.array(self.V), 
            shader='shaded', 
            color=(0.2, 0.6, 1.0, 1.0)
        )
        self.viewer.addItem(self.surface)

        # Set up a temporal loop to update the graph frames dynamically
        self.current_step = 0
        self.start_simulation()

    def start_simulation(self):
        """Triggers the recursive frame updates using a lightweight thread callback."""
        from PyQt6.QtCore import QTimer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_surface_frame)
        self.timer.start(1) # Fire as fast as the GPU can loop memory

    def update_surface_frame(self):
        """Pushes single temporal slices backwards through time and updates display."""
        if self.current_step < config.TIME_STEPS:
            # Step the mathematical backend matrix on your hardware
            self.V = engine.step_explicit_euler(self.V, self.S1, self.S2, self.dx, self.dy, self.dt)
            
            # Send current pricing frame straight to the OpenGL shader mapping pipeline
            z_data = np.array(self.V)
            self.surface.setData(z=z_data)
            self.current_step += 1
        else:
            self.timer.stop()
            print("PDE Solution space successfully reached convergence state.")

def main():
    app = QApplication(sys.argv)
    window = BlackScholesVisualizer()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
