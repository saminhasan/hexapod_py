from numpy import sqrt, sin, cos, dot, pi, arctan2, array, radians, set_printoptions, append, newaxis, repeat, einsum, zeros, linalg, matmul, arcsin,degrees, mean, vstack
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
set_printoptions(precision=32, suppress=True)
def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
    ax: a matplotlib axis, e.g., as output from plt.gca().
    """
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    z_limits = ax.get_zlim()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.45 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def rotX(theta):
    rotx = array([
        [1,     0    ,    0    ],
        [0,  cos(theta), -sin(theta)],
        [0,  sin(theta), cos(theta)] ])
    return rotx

def rotY(theta):    
    roty = array([
        [cos(theta), 0,  sin(theta) ],
        [0         , 1,     0       ],
        [-sin(theta), 0,  cos(theta) ] ])   
    return roty
    
def rotZ(theta):    
    rotz = array([
        [ cos(theta),-sin(theta), 0 ],
        [ sin(theta), cos(theta), 0 ],
        [   0        ,     0      , 1 ] ])   
    return rotz
    

class Excenter:
    def __init__(self) -> None:
        self.R = 0.08

class Coupler:
    def __init__(self) -> None:
        self.L = (0.02 + 0.12) * 2

class Base:
    def __init__(self) -> None:
        self.L = 0.3
        self.h = sqrt(self.L**2 - (self.L / 2)**2)
        self.ri = self.h / 3
        self.ro = 2 * self.ri
        self.bearingwidth = 0.03
        self.l = 0.12
        self.center = array([0, 0, 0])
        self.orientation = radians(array([0, 120, 240]))

        self.corner1 = self.center + self.ro * array([cos(self.orientation[0]), sin(self.orientation[0]), 0])
        self.corner2 = self.center + self.ro * array([cos(self.orientation[1]), sin(self.orientation[1]), 0])
        self.corner3 = self.center + self.ro * array([cos(self.orientation[2]), sin(self.orientation[2]), 0])

        self.corners = array([self.corner1, self.corner2, self.corner3])
        
        self.P12L = self.corner1 + (self.corner2 - self.corner1) / sqrt(dot(self.corner2 - self.corner1, self.corner2 - self.corner1)) * self.l - self.bearingwidth / 2 * array([cos(self.orientation[2]), sin(self.orientation[2]), 0])
        self.P12R = self.corner2 - (self.corner2 - self.corner1) / sqrt(dot(self.corner2 - self.corner1, self.corner2 - self.corner1)) * self.l - self.bearingwidth / 2 * array([cos(self.orientation[2]), sin(self.orientation[2]), 0])

        self.P23L = self.corner2 + (self.corner3 - self.corner2) / sqrt(dot(self.corner3 - self.corner2, self.corner3 - self.corner2)) * self.l - self.bearingwidth / 2 * array([cos(self.orientation[0]), sin(self.orientation[0]), 0])
        self.P23R = self.corner3 - (self.corner3 - self.corner2) / sqrt(dot(self.corner3 - self.corner2, self.corner3 - self.corner2)) * self.l - self.bearingwidth / 2 * array([cos(self.orientation[0]), sin(self.orientation[0]), 0])

        self.P31L = self.corner3 + (self.corner1 - self.corner3) / sqrt(dot(self.corner1 - self.corner3, self.corner1 - self.corner3)) * self.l - self.bearingwidth / 2 * array([cos(self.orientation[1]), sin(self.orientation[1]), 0])
        self.P31R = self.corner1 - (self.corner1 - self.corner3) / sqrt(dot(self.corner1 - self.corner3, self.corner1 - self.corner3)) * self.l - self.bearingwidth / 2 * array([cos(self.orientation[1]), sin(self.orientation[1]), 0])

        self.bearings = array([self.P12L, self.P12R, self.P23L, self.P23R, self.P31L, self.P31R]).T

        self.beta = radians(array([-30,150, 90, 270, 210, 390]))


class Platform:
    def __init__(self) -> None:
        self.L = 0.3
        self.h = sqrt(self.L**2 - (self.L / 2)**2)
        self.ri = self.h / 3
        self.ro = 2 * self.ri
        self.bearingwidth = 0.03
        self.l = 0.13
        self.center = array([0, 0, 0])
        self.orientation = radians(array([0, 120, 240]))

        self.corner1 = self.center + self.ro * array([cos(self.orientation[0]), sin(self.orientation[0]), 0])
        self.corner2 = self.center + self.ro * array([cos(self.orientation[1]), sin(self.orientation[1]), 0])
        self.corner3 = self.center + self.ro * array([cos(self.orientation[2]), sin(self.orientation[2]), 0])

        self.corners = array([self.corner1, self.corner2, self.corner3])
        
        self.P12L = self.corner1 + (self.corner2 - self.corner1) / sqrt(dot(self.corner2 - self.corner1, self.corner2 - self.corner1)) * self.l + self.bearingwidth / 2 * array([cos(self.orientation[2]), sin(self.orientation[2]), 0])
        self.P12R = self.corner2 - (self.corner2 - self.corner1) / sqrt(dot(self.corner2 - self.corner1, self.corner2 - self.corner1)) * self.l + self.bearingwidth / 2 * array([cos(self.orientation[2]), sin(self.orientation[2]), 0])

        self.P23L = self.corner2 + (self.corner3 - self.corner2) / sqrt(dot(self.corner3 - self.corner2, self.corner3 - self.corner2)) * self.l + self.bearingwidth / 2 * array([cos(self.orientation[0]), sin(self.orientation[0]), 0])
        self.P23R = self.corner3 - (self.corner3 - self.corner2) / sqrt(dot(self.corner3 - self.corner2, self.corner3 - self.corner2)) * self.l + self.bearingwidth / 2 * array([cos(self.orientation[0]), sin(self.orientation[0]), 0])

        self.P31L = self.corner3 + (self.corner1 - self.corner3) / sqrt(dot(self.corner1 - self.corner3, self.corner1 - self.corner3)) * self.l + self.bearingwidth / 2 * array([cos(self.orientation[1]), sin(self.orientation[1]), 0])
        self.P31R = self.corner1 - (self.corner1 - self.corner3) / sqrt(dot(self.corner1 - self.corner3, self.corner1 - self.corner3)) * self.l + self.bearingwidth / 2 * array([cos(self.orientation[1]), sin(self.orientation[1]), 0])

        self.bearings = array([self.P12L, self.P12R, self.P23L, self.P23R, self.P31L, self.P31R]).T


class Hexapod:
    def __init__(self) -> None:
        self.base = Base()
        self.excenter = Excenter()
        self.coupler = Coupler()
        self.platform = Platform()

        servo_arm = array([self.excenter.R, 0, 0])
        servo_tips= array([t + rotZ(r)@servo_arm for r, t in zip(self.base.beta, self.base.bearings.T)])
        dist_xy = (servo_tips[:,0]- self.platform.bearings.T[:,0])**2 + (servo_tips[:,1]- self.platform.bearings.T[:,1])**2
        z_zero = mean(sqrt(self.coupler.L**2  - dist_xy))
        self.home_pos = array([0, 0, z_zero])

        self.B = self.base.bearings
        self.P = self.platform.bearings
        # Allocate arrays for variables
        self.angles = zeros((6))
        self.H = zeros((3,6)) 

    def move(self, translation=array([0, 0, 0]), rotation=array([0, 0, 0])):
        l = zeros((3,6))
        lll = zeros((6))
        
        R = matmul(matmul(rotX(rotation[0]), rotZ(rotation[2])), rotY(rotation[1])) # matlab eq

        # Get leg length for each leg
        l = repeat(translation[:, newaxis], 6, axis=1) + repeat(self.home_pos[:, newaxis], 6, axis=1) + matmul(R, self.P) - self.B
        self.P_moved = l + self.B
        lll = linalg.norm(l, axis=0)
        L = l + self.B
        ldl = self.coupler.L
        lhl = self.excenter.R        

        lx = l[0, :]
        ly = l[1, :]
        lz = l[2, :]
        g = lll**2 - (ldl**2 - lhl**2)
        e = 2 * lhl * lz
        fk = 2 * lhl * (cos(self.base.beta) * lx + sin(self.base.beta) * ly)
        # Calculate servo angles for each leg
        angles = arcsin(g / sqrt(e**2 + fk**2)) - arctan2(fk, e)
        
        # Get positions of the point where a spherical joint connects servo arm and rod.
        self.H[0, :] = lhl * cos(angles) * cos(self.base.beta) + self.B[0, :]
        self.H[1, :] = lhl * cos(angles) * sin(self.base.beta) + self.B[1, :]
        self.H[2, :] = lhl * sin(angles)

    
if __name__ == '__main__':
    hexapod = Hexapod()
    # hexapod.move(translation=array([-0.05, 0.05, 0.05]), rotation=-radians(array([0.0, 0.0, 0.0])))
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    def update(val):
        translation = array([slider_x.val, slider_y.val, slider_z.val])
        rotation = radians(array([slider_rx.val, slider_ry.val, slider_rz.val]))
        hexapod.move(translation, rotation)
        ax.clear()
        ax.scatter(*hexapod.B, color='k', s=20, label='Base Bearings')
        ax.scatter(*hexapod.P_moved, color='red', s=20, label='Moved Platform Bearings')
        ax.scatter(*hexapod.H, color='purple', s=20, label='Spherical Joints')
        # Draw lines between B and H, and H and P_moved
        for i in range(6):
            ax.plot([hexapod.B[0, i], hexapod.H[0, i]], [hexapod.B[1, i], hexapod.H[1, i]], [hexapod.B[2, i], hexapod.H[2, i]], color='k')
            ax.plot([hexapod.H[0, i], hexapod.P_moved[0, i]], [hexapod.H[1, i], hexapod.P_moved[1, i]], [hexapod.H[2, i], hexapod.P_moved[2, i]], color='red')

        ax.add_collection3d(Poly3DCollection([list(hexapod.P_moved.T)], facecolors='blue', alpha=0.25))
        # ax.legend()
        set_axes_equal(ax)
        plt.draw()

    # Adjust the main plot to make room for the sliders
    plt.subplots_adjust(left=0.25, bottom=0.4)

    # Define sliders for x, y, z, rx, ry, rz
    ax_x = plt.axes([0.25, 0.30, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    ax_y = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    ax_z = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    ax_rx = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    ax_ry = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    ax_rz = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')

    slider_x = Slider(ax_x, 'X', -1.0, 1.0, valinit=0)
    slider_y = Slider(ax_y, 'Y', -1.0, 1.0, valinit=0)
    slider_z = Slider(ax_z, 'Z', -1.0, 1.0, valinit=0)
    slider_rx = Slider(ax_rx, 'Rx', -180.0, 180.0, valinit=0)
    slider_ry = Slider(ax_ry, 'Ry', -180.0, 180.0, valinit=0)
    slider_rz = Slider(ax_rz, 'Rz', -180.0, 180.0, valinit=0)

    # Call update function on slider value change
    slider_x.on_changed(update)
    slider_y.on_changed(update)
    slider_z.on_changed(update)
    slider_rx.on_changed(update)
    slider_ry.on_changed(update)
    slider_rz.on_changed(update)

    # Initial plot
    update(None)
    plt.show()
