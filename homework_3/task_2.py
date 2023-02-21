import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import patches
import numpy as np

from collections import namedtuple
from typing import Callable

matplotlib.use('TkAgg')


class Coordinates:
    Description = namedtuple('Description', ['o1', 'o2', 'o', 'm'])

    def __init__(self, phi_func: Callable[[float], float], theta_func: Callable[[float], float], r: float):
        self.phi_func = phi_func
        self.theta_func = theta_func
        self.r = r

    def o1(self, time: float):
        return np.zeros(2)

    def o2(self, time: float):
        phi = self.phi_func(time)
        return self.r * np.array([-np.sin(phi), np.cos(phi)])

    def o(self, time: float):
        phi = self.phi_func(time)
        return 2 ** 0.5 * self.r * np.array([np.cos(phi + np.pi / 4), np.sin(phi + np.pi / 4)])

    def m(self, time: float):
        phi = self.phi_func(time)
        theta = self.theta_func(time)
        return self.o2(time) + self.r * np.array([np.cos(theta + phi), np.sin(theta + phi)])

    def get_current(self, time: float):
        return self.Description(self.o1(time), self.o2(time), self.o(time), self.m(time))


class Velocities:
    Description = namedtuple('Description', ['m_rel', 'm_tr', 'm_abs'])

    def __init__(self,
                 phi_func: Callable[[float], float], d_phi: Callable[[float], float],
                 theta_func: Callable[[float], float], d_theta: Callable[[float], float],
                 r: float, coords: Coordinates):
        self.phi_func = phi_func
        self.d_phi = d_phi
        self.theta_func = theta_func
        self.d_theta = d_theta
        self.r = r
        self.coordinates = coords

    def m_rel(self, time: float):
        theta = self.theta_func(time)
        d_theta = self.d_theta(time)
        phi = self.phi_func(time)
        d_phi = self.d_phi(time)
        return self.r * (d_theta + d_phi) * np.array([-np.sin(theta + phi), np.cos(theta + phi)])

    def m_tr(self, time: float):
        phi = self.phi_func(time)
        d_phi = self.d_phi(time)
        m_pos = self.coordinates.m(time)
        o1_pos = self.coordinates.o1(time)
        distance = np.linalg.norm(m_pos - o1_pos)
        return -distance * d_phi * np.array([np.cos(phi), np.sin(phi)])

    def m_abs(self, time: float):
        relative = self.m_rel(time)
        transport = self.m_tr(time)
        return relative + transport

    def get_current(self, time: float):
        return self.Description(self.m_rel(time), self.m_tr(time), self.m_abs(time))


class Acceleration:
    Description = namedtuple('Description', ['m_rel', 'm_tr', 'm_abs'])

    def __init__(self,
                 phi_func: Callable[[float], float], d_phi: Callable[[float], float], dd_phi: Callable[[float], float],
                 theta_func: Callable[[float], float], d_theta: Callable[[float], float], dd_theta: Callable[[float], float],
                 r: float, coords: Coordinates, _velocities: Velocities):
        self.phi_func = phi_func
        self.d_phi = d_phi
        self.dd_phi = dd_phi
        self.theta_func = theta_func
        self.d_theta = d_theta
        self.dd_theta = dd_theta
        self.r = r
        self.coordinates = coords
        self.velocities = _velocities

    def m_rel(self, time: float):
        phi = self.phi_func(time)
        d_phi = self.d_phi(time)
        dd_phi = self.dd_phi(time)
        theta = self.theta_func(time)
        d_theta = self.d_theta(time)
        dd_theta = self.dd_theta(time)
        return self.r * ((dd_theta + dd_phi) * np.array([-np.sin(theta + phi), np.cos(theta + phi)]) -
                         (d_theta + d_phi) ** 2 * np.array([np.cos(theta + phi), np.sin(theta + phi)]))

    def m_tr(self, time: float):
        d_phi = self.d_phi(time)
        dd_phi = self.dd_phi(time)
        m_pos = self.coordinates.m(time)
        o1_pos = self.coordinates.o1(time)
        rho = m_pos - o1_pos
        angular_velocity = np.array([0, 0, d_phi])
        angular_acceleration = np.array([0, 0, dd_phi])
        return np.cross(angular_acceleration, rho)[:-1] + np.cross(angular_velocity, np.cross(angular_velocity, rho))[:-1]

    def m_cor(self, time: float):
        d_phi = self.d_phi(time)
        angular_velocity = np.array([0, 0, d_phi])
        relative_velocity = self.velocities.m_rel(time)
        return 2 * np.cross(angular_velocity, relative_velocity)[:-1]

    def m_abs(self, time: float):
        return self.m_rel(time) + self.m_tr(time) + self.m_cor(time)

    def get_current(self, time: float):
        return self.Description(self.m_rel(time), self.m_tr(time), self.m_abs(time))



frames = 1000
radius = 30
t_finish = (-1 + (1 + 16 / 5 * radius) ** 0.5) / 6
phi_of_time = lambda t: 2 * t - 0.3 * t ** 2
d_phi_of_time = lambda t: 2 - 0.6 * t
dd_phi_of_time = lambda t: -0.6

theta_of_time = lambda t: 75 * np.pi * (0.1 * t + 0.3 * t ** 2) / radius
d_theta_of_time = lambda t: 75 * np.pi * (0.1 + 0.6 * t) / radius
dd_theta_of_time = lambda t: 75 * np.pi * 0.6 / radius

timeline = np.linspace(0, t_finish, frames)

coordinates = Coordinates(phi_of_time, theta_of_time, radius)
velocities = Velocities(phi_of_time, d_phi_of_time, theta_of_time, d_theta_of_time, radius, coordinates)
accelerations = Acceleration(phi_of_time, d_phi_of_time, dd_phi_of_time, theta_of_time, d_theta_of_time,
                             dd_theta_of_time, radius, coordinates, velocities)

initial_positions = coordinates.get_current(0)
initial_velocities = velocities.get_current(0)
initial_accelerations = accelerations.get_current(0)

figure, axes = plt.subplots(1, 2, figsize=(15, 5))

points_line_1,  = axes[0].plot([], [], 'o')
points_line_2,  = axes[1].plot([], [], 'o')
x_lim, y_lim = (-(2 * radius + 5), 2 * radius + 5), (-(2 * radius + 5), 2 * radius + 5)
axes[0].set_autoscale_on(False)
axes[0].set_xlim(*x_lim)
axes[0].set_ylim(*y_lim)
axes[1].set_autoscale_on(False)
axes[1].set_xlim(*x_lim)
axes[1].set_ylim(*y_lim)

annotations_axes_1 = [
    axes[0].annotate('O1', initial_positions.o1),
    axes[0].annotate('O2', initial_positions.o2),
    axes[0].annotate('O', initial_positions.o),
    axes[0].annotate('M', initial_positions.m)
]

annotations_axes_2 = [
    axes[1].annotate('O1', initial_positions.o1),
    axes[1].annotate('O2', initial_positions.o2),
    axes[1].annotate('O', initial_positions.o),
    axes[1].annotate('M', initial_positions.m)
]

velocity_relative_vector = axes[0].quiver(initial_positions.m,
                                          initial_velocities.m_rel,
                                          color='r')
velocity_relative_key = axes[0].quiverkey(velocity_relative_vector,
                                          0.9, 0.95, 100,
                                          'relative velocity',
                                          labelpos='W',
                                          color='r')

velocity_transport_vector = axes[0].quiver(initial_positions.m,
                                           initial_velocities.m_tr,
                                           color='g')
velocity_transport_key = axes[0].quiverkey(velocity_transport_vector,
                                           0.9, 0.9, 100,
                                           'transport velocity',
                                           labelpos='W',
                                           color='g')

velocity_absolute_vector = axes[0].quiver(initial_positions.m,
                                          initial_velocities.m_abs,
                                          color='b')
velocity_absolute_key = axes[0].quiverkey(velocity_absolute_vector,
                                          0.9, 0.85, 150,
                                          'absolute velocity',
                                          labelpos='W',
                                          color='b')

acceleration_relative_vector = axes[1].quiver(initial_positions.m,
                                              initial_accelerations.m_rel,
                                              color='r')
acceleration_relative_key = axes[1].quiverkey(acceleration_relative_vector,
                                              0.9, 0.95, 200,
                                              'relative acceleration',
                                              labelpos='W',
                                              color='r')

acceleration_transport_vector = axes[1].quiver(initial_positions.m,
                                               initial_accelerations.m_tr,
                                               color='g')
acceleration_transport_key = axes[1].quiverkey(acceleration_relative_vector,
                                               0.9, 0.9, 200,
                                               'transport acceleration',
                                               labelpos='W',
                                               color='g')

acceleration_absolute_vector = axes[1].quiver(initial_positions.m,
                                              initial_accelerations.m_abs,
                                              color='b')
acceleration_absolute_key = axes[1].quiverkey(acceleration_relative_vector,
                                              0.9, 0.85, 200,
                                              'absolute acceleration',
                                              labelpos='W',
                                              color='b')


def animate(frame_index: int):
    time = timeline[frame_index]
    positions = coordinates.get_current(time)
    current_velocities = velocities.get_current(time)
    current_accelerations = accelerations.get_current(time)
    points_line_1.set_data(*zip(*positions))
    points_line_2.set_data(*zip(*positions))

    velocity_relative_vector.set_offsets(positions.m)
    velocity_relative_vector.set_UVC(*current_velocities.m_rel)

    velocity_transport_vector.set_offsets(positions.m)
    velocity_transport_vector.set_UVC(*current_velocities.m_tr)

    velocity_absolute_vector.set_offsets(positions.m)
    velocity_absolute_vector.set_UVC(*current_velocities.m_abs)

    acceleration_relative_vector.set_offsets(positions.m)
    acceleration_relative_vector.set_UVC(*current_accelerations.m_rel)

    acceleration_transport_vector.set_offsets(positions.m)
    acceleration_transport_vector.set_UVC(*current_accelerations.m_tr)

    acceleration_absolute_vector.set_offsets(positions.m)
    acceleration_absolute_vector.set_UVC(*current_accelerations.m_abs)

    circles = [
        axes[0].add_patch(patches.Circle(tuple(positions.o2), radius, fill=False)),
        axes[1].add_patch(patches.Circle(tuple(positions.o2), radius, fill=False))
    ]

    for annotation, position in zip(annotations_axes_1, positions):
        annotation.set_position(position)

    for annotation, position in zip(annotations_axes_2, positions):
        annotation.set_position(position)

    return *circles, velocity_relative_vector, velocity_transport_vector, velocity_absolute_vector,\
        acceleration_relative_vector, acceleration_transport_vector, acceleration_absolute_vector,points_line_1, \
        points_line_2, *annotations_axes_1, *annotations_axes_2


anim = animation.FuncAnimation(figure, animate, frames, blit=True, interval=1)

plt.show()
