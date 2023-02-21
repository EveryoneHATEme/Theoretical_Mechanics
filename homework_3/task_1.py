import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.lines import Line2D
from matplotlib import patches
import numpy as np

from collections import namedtuple
from typing import Callable

matplotlib.use('TkAgg')


class Coordinates:
    Description = namedtuple('Description', ['o1', 'o2', 'o3', 'd', 'o', 'a', 'm'])

    def __init__(self, phi_func: Callable[[float], float], theta_func: Callable[[float], float], l: float, r: float):
        self.r = r
        self.l = l
        self.theta_func = theta_func
        self.phi_func = phi_func

    def o1(self, time: float):
        return np.array([-self.r, 0])

    def o2(self, time: float):
        return np.array([self.r, 0])

    @staticmethod
    def o3(time: float):
        return np.array([0, 0])

    def d(self, time):
        current_phi = self.phi_func(time)
        return self.o3(time) + self.l * np.array([np.cos(current_phi), np.sin(current_phi)])

    def o(self, time: float):
        return self.o1(time) + self.d(time)

    def a(self, time: float):
        return self.o2(time) + self.d(time)

    def m(self, time: float):
        theta = self.theta_func(time)
        return self.d(time) + self.r * np.array([np.cos(np.pi - theta), np.sin(np.pi - theta)])

    def get_current(self, time: float):
        return Coordinates.Description(
            self.o1(time),
            self.o2(time),
            self.o3(time),
            self.d(time),
            self.o(time),
            self.a(time),
            self.m(time)
        )


class Velocities:
    Description = namedtuple('Description', ['m_rel', 'm_tr', 'm_abs'])

    def __init__(self,
                 phi_func: Callable[[float], float], d_phi: Callable[[float], float],
                 theta_func: Callable[[float], float], d_theta: Callable[[float], float],
                 l: float, r: float):
        self.phi_func = phi_func
        self.d_phi = d_phi
        self.theta_func = theta_func
        self.d_theta = d_theta
        self.l = l
        self.r = r

    def m_tr(self, time: float):
        _phi = self.phi_func(time)
        return self.l * self.d_phi(time) * np.array([-np.sin(_phi), np.cos(_phi)])

    def m_rel(self, time: float):
        _theta = self.theta_func(time)
        return self.r * self.d_theta(time) * np.array([np.sin(_theta), np.cos(_theta)])

    def m_abs(self, time: float):
        return self.m_tr(time) + self.m_rel(time)

    def get_current(self, time: float):
        return self.Description(self.m_rel(time), self.m_tr(time), self.m_abs(time))


class Accelerations:
    Description = namedtuple('Description', ['m_rel', 'm_abs', 'm_tr'])

    def __init__(self,
                 phi_func: Callable[[float], float], d_phi: Callable[[float], float], dd_phi: Callable[[float], float],
                 theta_func: Callable[[float], float], d_theta: Callable[[float], float], dd_theta: Callable[[float], float],
                 l: float, r: float):
        self.phi_func = phi_func
        self.d_phi = d_phi
        self.dd_phi = dd_phi
        self.theta_func = theta_func
        self.d_theta = d_theta
        self.dd_theta = dd_theta
        self.l = l
        self.r = r

    def m_rel(self, time: float):
        _theta = self.theta_func(time)
        _d_theta = self.d_theta(time)
        _dd_theta = self.dd_theta(time)
        normal = _d_theta ** 2 * self.r * np.array([np.cos(_theta), -np.sin(_theta)])
        tangential = np.cross(np.array([0, 0, _dd_theta]), self.r * np.array([-np.cos(_theta), np.sin(_theta)]))[:-1]
        return normal + tangential

    def m_tr(self, time: float):
        _phi = self.phi_func(time)
        _d_phi = self.d_phi(time)
        _dd_phi = self.dd_phi(time)
        normal = -_d_phi ** 2 * self.l * np.array([np.cos(_phi), np.sin(_phi)])
        tangential = np.cross(np.array([0, 0, _dd_phi]), self.l * np.array([np.cos(_phi), np.sin(_phi)]))[:-1]
        return normal + tangential

    def m_abs(self, time: float):
        transport = self.m_tr(time)
        relative = self.m_rel(time)
        return transport + relative

    def get_current(self, time: float):
        return self.Description(self.m_rel(time), self.m_abs(time), self.m_tr(time))


frames = 1000
o1o_length, radius = 20, 18

phi = lambda t: np.pi * t ** 3 / 6
theta = lambda t: 6 * np.pi * t ** 2 / radius

d_phi = lambda t: np.pi * t ** 2 / 2
d_theta = lambda t: 12 * np.pi * t / radius

dd_phi = lambda t: np.pi * t
dd_theta = lambda t: 12 * np.pi / radius

coordinates = Coordinates(phi, theta, o1o_length, radius)
velocities = Velocities(phi, d_phi, theta, d_theta, o1o_length, radius)
accelerations = Accelerations(phi, d_phi, dd_phi, theta, d_theta, dd_theta, o1o_length, radius)

timeline = np.linspace(0, np.sqrt(np.pi * radius / (6 * np.pi)), frames)

initial_positions = coordinates.get_current(0)
initial_velocities = velocities.get_current(0)
initial_accelerations = accelerations.get_current(0)

figure, axes = plt.subplots(1, 2, figsize=(15, 5))

points_line_1, = axes[0].plot([], [], 'o')
points_line_2, = axes[1].plot([], [], 'o')
x_lim, y_lim = (-(radius * 2 + 5), radius * 2 + 5), (-(o1o_length + radius + 5), o1o_length + radius + 5)
axes[0].set_autoscale_on(False)
axes[0].set_xlim(*x_lim)
axes[0].set_ylim(*y_lim)
axes[1].set_autoscale_on(False)
axes[1].set_xlim(*x_lim)
axes[1].set_ylim(*y_lim)

velocity_relative_vector = axes[0].quiver(initial_positions.m,
                                          initial_velocities.m_rel,
                                          color='r',
                                          scale=700)
velocity_relative_key = axes[0].quiverkey(velocity_relative_vector,
                                          0.9, 0.95, 50,
                                          'relative velocity',
                                          labelpos='W',
                                          color='r')

velocity_transport_vector = axes[0].quiver(initial_positions.m,
                                           initial_velocities.m_tr,
                                           color='g',
                                           scale=700)
velocity_transport_key = axes[0].quiverkey(velocity_transport_vector,
                                           0.9, 0.9, 50,
                                           'transport velocity',
                                           labelpos='W',
                                           color='g')

velocity_absolute_vector = axes[0].quiver(initial_positions.m,
                                          initial_velocities.m_abs,
                                          color='b',
                                          scale=700)
velocity_absolute_key = axes[0].quiverkey(velocity_absolute_vector,
                                          0.9, 0.85, 50,
                                          'absolute velocity',
                                          labelpos='W',
                                          color='b')

acceleration_relative_vector = axes[1].quiver(initial_positions.m,
                                           initial_accelerations.m_rel,
                                           color='r',
                                            scale=1500)
acceleration_relative_key = axes[1].quiverkey(acceleration_relative_vector,
                                           0.9, 0.95, 125,
                                           'relative acceleration',
                                           labelpos='W',
                                           color='r')

acceleration_transport_vector = axes[1].quiver(initial_positions.m,
                                            initial_accelerations.m_tr,
                                            color='g',
                                            scale=1500)
acceleration_transport_key = axes[1].quiverkey(acceleration_transport_vector,
                                           0.9, 0.9, 125,
                                           'transport acceleration',
                                           labelpos='W',
                                           color='g')

acceleration_absolute_vector = axes[1].quiver(initial_positions.m,
                                           initial_accelerations.m_abs,
                                           color='b',
                                           scale=1500)
acceleration_absolute_key = axes[1].quiverkey(acceleration_absolute_vector,
                                           0.9, 0.85, 125,
                                           'absolute acceleration',
                                           labelpos='W',
                                           color='b')

def animate(frame_index):
    time = timeline[frame_index]
    positions = coordinates.get_current(time)
    current_velocities = velocities.get_current(time)
    current_accelerations = accelerations.get_current(time)
    points_line_1.set_data(*zip(positions.o1, positions.o, positions.a, positions.o2, positions.m))
    points_line_2.set_data(*zip(positions.o1, positions.o, positions.a, positions.o2, positions.m))

    if frame_index != 0:
        velocity_relative_vector.set_offsets(positions.m)
        velocity_relative_vector.set_UVC(*current_velocities.m_rel)

        velocity_transport_vector.set_offsets(positions.m)
        velocity_transport_vector.set_UVC(*current_velocities.m_tr)

        velocity_absolute_vector.set_offsets(positions.m)
        velocity_absolute_vector.set_UVC(*current_velocities.m_abs)

        acceleration_transport_vector.set_offsets(positions.m)
        acceleration_transport_vector.set_UVC(*current_accelerations.m_tr)

    acceleration_relative_vector.set_offsets(positions.m)
    acceleration_relative_vector.set_UVC(*current_accelerations.m_rel)

    acceleration_absolute_vector.set_offsets(positions.m)
    acceleration_absolute_vector.set_UVC(*current_accelerations.m_abs)

    lines = [
        axes[0].add_line(Line2D(*zip(positions.o1, positions.o, positions.a, positions.o2), color='black')),
        axes[1].add_line(Line2D(*zip(positions.o1, positions.o, positions.a, positions.o2), color='black'))
    ]
    arcs = [
        axes[0].add_patch(patches.Arc(positions.d, radius * 2, radius * 2, theta1=0, theta2=180)),
        axes[1].add_patch(patches.Arc(positions.d, radius * 2, radius * 2, theta1=0, theta2=180))
    ]

    return *lines, *arcs, points_line_1, points_line_2, velocity_relative_vector, velocity_transport_vector,\
        velocity_absolute_vector, acceleration_relative_vector, acceleration_transport_vector,\
        acceleration_absolute_vector


anim = animation.FuncAnimation(figure, animate, frames=frames, interval=1, blit=True)

plt.show()
