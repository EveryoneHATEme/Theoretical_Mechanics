import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from math import radians

from collections import namedtuple
from operator import itemgetter

matplotlib.use('TkAgg')


Lengths = namedtuple('Lengths',
                     ['a', 'b', 'c', 'd', 'e', 'O1A', 'O2D', 'O3E', 'AB', 'AD', 'GH', 'DE', 'GF', 'FH', 'O4G'])

lengths = Lengths(32, 4, 39, 19, 32, 12, 32, 18, 46, 29, 14, 53, 25, 14, 20)

MechanismPositions = namedtuple('MechanismPositions', ['point_O1', 'point_O2', 'point_O3', 'point_O4',
                                                       'point_A', 'point_B', 'point_C', 'point_D',
                                                       'point_E', 'point_F', 'point_G', 'point_H'])
MechanismVelocities = namedtuple('MechanismVelocities', ['point_O1', 'point_O2', 'point_O3', 'point_O4',
                                                         'point_A', 'point_B', 'point_C', 'point_D',
                                                         'point_E', 'point_F', 'point_G', 'point_H'])

MechanismAccelerations = namedtuple('MechanismAccelerations', ['point_O1', 'point_O2', 'point_O3', 'point_O4',
                                                               'point_A', 'point_B', 'point_C', 'point_D',
                                                               'point_E', 'point_F', 'point_G', 'point_H'])

MechanismAngularVelocities = namedtuple('MechanismAngularVelocities', ['O1A', 'AB', 'AD', 'O2D',
                                                                       'DE', 'O3E', 'O4G', 'GF'])

def get_o2_pos(o1_pos: np.ndarray, a: float, b: float, d: float):
    return o1_pos + np.array([a + b, -d])


def get_o3_pos(o1_pos: np.ndarray, a: float, b: float, c: float, e: float):
    return o1_pos + np.array([a + b + c, e])


def get_o4_pos(o1_pos: np.ndarray, a: float, e: float):
    return o1_pos + np.array([a, e])


def get_position_between(first_pos: np.ndarray, second_pos: np.ndarray, length: float):
    hat_vec = (second_pos - first_pos) / np.linalg.norm(second_pos - first_pos)
    return first_pos + length * hat_vec


def get_velocity_between(first_pos: np.ndarray, first_vel: np.ndarray,
                         second_pos: np.ndarray, second_vel: np.ndarray, length: float):
    d = np.linalg.norm(second_pos - first_pos)
    x_a, y_a = first_pos
    x_b, y_b = second_pos
    xd_a, yd_a = first_vel
    xd_b, yd_b = second_vel
    d_prime = ((x_b - x_a) * (xd_b - xd_a) + (y_b - y_a) * (yd_b - yd_a)) / d
    hat_x = ((xd_b - xd_a) * d - (x_b - x_a) * d_prime) / d ** 2
    hat_y = ((yd_b - yd_a) * d - (y_b - y_a) * d_prime) / d ** 2
    hat_vector = np.array([hat_x, hat_y])
    return first_vel + length * hat_vector


def get_h_pos(g_pos: np.ndarray, f_pos: np.ndarray, fh_length: float, gh_length: float, gf_length: float):
    fg_vec = g_pos - f_pos
    cos_f = (gf_length ** 2 + fh_length ** 2 - gh_length ** 2) / (2 * gf_length * fh_length)
    sin_f = (1 - cos_f ** 2) ** 0.5
    rotation_matrix = np.array([[cos_f, sin_f],
                                [-sin_f, cos_f]])
    fg_hat = fg_vec / np.linalg.norm(fg_vec)
    fh_hat = rotation_matrix.dot(fg_hat)
    fh_vec = fh_hat * fh_length
    return fh_vec + f_pos


def get_b_velocity(phi: float, o1a_length: float, ab_length: float) -> np.ndarray:
    first_term = 2 * o1a_length * np.cos(phi)
    second_term = (o1a_length ** 2 * np.sin(phi * 2)) / (ab_length ** 2 - o1a_length ** 2 * np.cos(phi) ** 2) ** 0.5
    return np.array([0, first_term + second_term])


def get_first_intersection(first_center: np.ndarray, first_radius: float, second_center: np.ndarray,
                           second_radius: float):
    distance = np.linalg.norm(first_center - second_center)
    a = (first_radius ** 2 - second_radius ** 2 + distance ** 2) / (2 * distance)
    h = (first_radius ** 2 - a ** 2) ** 0.5

    x_1, y_1 = first_center
    x_2, y_2 = second_center

    vec_1 = np.array([(x_2 - x_1) / distance, (y_2 - y_1) / distance])
    vec_2 = np.array([(y_1 - y_2) / distance, (x_2 - x_1) / distance])

    return first_center + a * vec_1 + h * vec_2


def get_first_intersection_derivative(first_center: np.ndarray, first_center_derivative: np.ndarray, first_radius: float,
                                      second_center: np.ndarray, second_center_derivative: np.ndarray, second_radius: float):
    x_1, y_1 = first_center
    x_2, y_2 = second_center
    xd_1, yd_1 = first_center_derivative
    xd_2, yd_2 = second_center_derivative
    d = np.linalg.norm(second_center - first_center)
    d_prime = ((x_2 - x_1) * (xd_2 - xd_1) + (y_2 - y_1) * (yd_2 - yd_1)) / d
    a = (first_radius ** 2 - second_radius ** 2 + d ** 2) / (2 * d)
    h = (first_radius ** 2 - a ** 2) ** 0.5
    a_prime = (d_prime * (d ** 2 - first_radius ** 2 + second_radius ** 2)) / (2 * d ** 2)
    h_prime = (-a * a_prime) / (first_radius ** 2 - a ** 2) ** 0.5

    vec_1 = np.array([(x_2 - x_1) / d, (y_2 - y_1) / d])
    vec_1_prime = np.array([((xd_2 - xd_1) * d - (x_2 - x_1) * d_prime) / d ** 2,
                            ((yd_2 - yd_1) * d - (y_2 - y_1) * d_prime) / d ** 2])
    vec_2 = np.array([(y_1 - y_2) / d, (x_2 - x_1) / d])
    vec_2_prime = np.array([((yd_1 - yd_2) * d - (y_1 - y_2) * d_prime) / d ** 2,
                            ((xd_2 - xd_1) * d - (x_2 - x_1) * d_prime) / d ** 2])

    return first_center_derivative + a_prime * vec_1 + a * vec_1_prime + h_prime * vec_2 + h * vec_2_prime


def get_second_intersection(first_center: np.ndarray, first_radius: float, second_center: np.ndarray,
                           second_radius: float):
    distance = np.linalg.norm(first_center - second_center)
    a = (first_radius ** 2 - second_radius ** 2 + distance ** 2) / (2 * distance)
    h = (first_radius ** 2 - a ** 2) ** 0.5

    x_1, y_1 = first_center
    x_2, y_2 = second_center

    vec_1 = np.array([(x_2 - x_1) / distance, (y_2 - y_1) / distance])
    vec_2 = np.array([(y_1 - y_2) / distance, (x_2 - x_1) / distance])

    return first_center + a * vec_1 - h * vec_2


def get_second_intersection_derivative(first_center: np.ndarray, first_center_derivative: np.ndarray, first_radius: float,
                                       second_center: np.ndarray, second_center_derivative: np.ndarray, second_radius: float):
    x_1, y_1 = first_center
    x_2, y_2 = second_center
    xd_1, yd_1 = first_center_derivative
    xd_2, yd_2 = second_center_derivative
    d = np.linalg.norm(second_center - first_center)
    d_prime = ((x_2 - x_1) * (xd_2 - xd_1) + (y_2 - y_1) * (yd_2 - yd_1)) / d
    a = (first_radius ** 2 - second_radius ** 2 + d ** 2) / (2 * d)
    h = (first_radius ** 2 - a ** 2) ** 0.5
    a_prime = (d_prime * (d ** 2 - first_radius ** 2 + second_radius ** 2)) / (2 * d ** 2)
    h_prime = (-a * a_prime) / (first_radius ** 2 - a ** 2) ** 0.5

    vec_1 = np.array([(x_2 - x_1) / d, (y_2 - y_1) / d])
    vec_1_prime = np.array([((xd_2 - xd_1) * d - (x_2 - x_1) * d_prime) / d ** 2,
                            ((yd_2 - yd_1) * d - (y_2 - y_1) * d_prime) / d ** 2])
    vec_2 = np.array([(y_2 - y_1) / d, (x_1 - x_2) / d])
    vec_2_prime = np.array([((yd_2 - yd_1) * d - (y_2 - y_1) * d_prime) / d ** 2,
                            ((xd_1 - xd_2) * d - (x_1 - x_2) * d_prime) / d ** 2])

    return first_center_derivative + a_prime * vec_1 + a * vec_1_prime + h_prime * vec_2 + h * vec_2_prime


def get_xy_lims(_lengths: Lengths):
    x_left = -_lengths.O1A - 5
    x_right = _lengths.a + _lengths.b + _lengths.c + 5
    y_bottom = -_lengths.d - 5
    y_top = max(_lengths.O1A + _lengths.AB, _lengths.O3E + _lengths.e, _lengths.O4G + _lengths.e) + 5

    return (min(x_left, y_bottom), max(x_right, y_top)), (min(x_left, y_bottom), max(x_right, y_top))


def get_angular_velocity(first_position: np.ndarray, first_velocity: np.ndarray,
                         second_position: np.ndarray, second_velocity: np.ndarray):
    radius_vector = second_position - first_position
    relative_velocity = second_velocity - first_velocity
    cos_a, sin_a = radius_vector / np.linalg.norm(radius_vector)
    hat_vector = np.array([-sin_a, cos_a])
    return relative_velocity.dot(hat_vector)


def phi_of_time(time: float):
    phi_initial = radians(52)
    omega = 2
    return phi_initial + omega * time


def get_positions(time: float) -> MechanismPositions:
    phi = phi_of_time(time)
    o1_pos = np.array([0, 0])
    a_pos = o1_pos + np.array([lengths.O1A * np.cos(phi), lengths.O1A * np.sin(phi)])
    b_pos = o1_pos + np.array([0, lengths.O1A * np.sin(phi) + (lengths.AB ** 2 - (lengths.O1A * np.cos(phi)) ** 2) ** 0.5])
    c_pos = get_position_between(a_pos, b_pos, 2 * lengths.AB / 3)
    o2_pos = get_o2_pos(o1_pos, lengths.a, lengths.b, lengths.d)
    d_pos = get_first_intersection(a_pos, lengths.AD, o2_pos, lengths.O2D)
    o3_pos = get_o3_pos(o1_pos, lengths.a, lengths.b, lengths.c, lengths.e)
    e_pos = get_first_intersection(d_pos, lengths.DE, o3_pos, lengths.O3E)
    f_pos = get_position_between(e_pos, d_pos, 2 * lengths.DE / 5)
    o4_pos = get_o4_pos(o1_pos, lengths.a, lengths.e)
    g_pos = get_second_intersection(f_pos, lengths.GF, o4_pos, lengths.O4G)
    h_pos = get_h_pos(g_pos, f_pos, lengths.FH, lengths.GH, lengths.GF)

    return MechanismPositions(o1_pos, o2_pos, o3_pos, o4_pos, a_pos, b_pos, c_pos, d_pos, e_pos, f_pos, g_pos, h_pos)


def get_velocities(time: float, positions: MechanismPositions) -> MechanismVelocities:
    phi = phi_of_time(time)
    o1_vel = np.array([0, 0])
    a_vel = o1_vel + np.array([-2 * lengths.O1A * np.sin(phi), 2 * lengths.O1A * np.cos(phi)])
    b_vel = get_b_velocity(phi, lengths.O1A, lengths.AB)
    c_vel = get_velocity_between(positions.point_A, a_vel, positions.point_B, b_vel, 2 * lengths.AB / 3)
    o2_vel = np.zeros(2)
    d_vel = get_first_intersection_derivative(positions.point_A, a_vel, lengths.AD,
                                              positions.point_O2, o2_vel, lengths.O2D)
    o3_vel = np.zeros(2)
    e_vel = get_first_intersection_derivative(positions.point_D, d_vel, lengths.DE,
                                              positions.point_O3, o3_vel, lengths.O3E)
    f_vel = get_velocity_between(positions.point_E, e_vel, positions.point_D, d_vel, 2 * lengths.DE / 5)
    o4_vel = np.zeros(2)
    g_vel = get_second_intersection_derivative(positions.point_F, f_vel, lengths.GF,
                                               positions.point_O4, o4_vel, lengths.O4G)
    h_vel = np.zeros(2)

    return MechanismVelocities(o1_vel, o2_vel, o3_vel, o4_vel, a_vel, b_vel, c_vel, d_vel, e_vel, f_vel, g_vel, h_vel)


def get_accelerations(time: float):
    phi = phi_of_time(time)
    o1_acc = np.zeros(2)
    o2_acc = np.zeros(2)
    o3_acc = np.zeros(2)
    o4_acc = np.zeros(2)
    a_acc = np.array([-4 * lengths.O1A * np.cos(phi), -4 * lengths.O1A * np.sin(phi)])
    wierd_root = (lengths.AB ** 2 - lengths.O1A ** 2 * np.cos(phi) ** 2) ** 0.5
    a_b = -4 * lengths.O1A * np.sin(phi) + \
          (4 * lengths.O1A ** 2 * np.cos(phi * 2) * wierd_root
           - (lengths.O1A ** 2 * np.sin(phi * 2)) ** 2 / wierd_root) / \
          (lengths.AB ** 2 - lengths.O1A ** 2 * np.cos(phi) ** 2)
    b_acc = np.array([0, a_b])
    c_acc = np.zeros(2)
    d_acc = np.zeros(2)
    e_acc = np.zeros(2)
    f_acc = np.zeros(2)
    g_acc = np.zeros(2)
    h_acc = np.zeros(2)

    return MechanismAccelerations(o1_acc, o2_acc, o3_acc, o4_acc, a_acc,
                                  b_acc, c_acc, d_acc, e_acc, f_acc, g_acc, h_acc)


def get_angular_velocities(positions: MechanismPositions, velocities: MechanismVelocities):
    o1a_vel = get_angular_velocity(positions.point_O1, velocities.point_O1,
                                   positions.point_A, velocities.point_A)
    ab_vel = get_angular_velocity(positions.point_A, velocities.point_A,
                                  positions.point_B, velocities.point_B)
    ad_vel = get_angular_velocity(positions.point_A, velocities.point_A,
                                  positions.point_D, velocities.point_D)
    o2d_vel = get_angular_velocity(positions.point_O2, velocities.point_O2,
                                  positions.point_D, velocities.point_D)
    de_vel = get_angular_velocity(positions.point_D, velocities.point_D,
                                  positions.point_E, velocities.point_E)
    o3e_vel = get_angular_velocity(positions.point_O3, velocities.point_O3,
                                   positions.point_E, velocities.point_E)
    o4g_vel = get_angular_velocity(positions.point_O4, velocities.point_O4,
                                  positions.point_G, velocities.point_G)
    gf_vel = get_angular_velocity(positions.point_G, velocities.point_G,
                                  positions.point_F, velocities.point_F)

    return MechanismAngularVelocities(o1a_vel, ab_vel, ad_vel, o2d_vel, de_vel, o3e_vel, o4g_vel, gf_vel)


frames = 2000

timeline = np.linspace(0, np.pi, frames)
dt = np.diff(timeline)
delta_time = dt[0]

figure, axes = plt.subplots()

traces_lines = [axes.plot(list(), list(), '--', color='orange')[0] for _ in range(12)]

traces = np.array(list(map(get_positions, timeline)))

mechanism_line, = axes.plot([], [], 'o-')
axes.set_autoscale_on(False)
x_lim, y_lim = get_xy_lims(lengths)
axes.set_xlim(*x_lim)
axes.set_ylim(*y_lim)

for i, line in enumerate(traces_lines):
    line.set_xdata(traces.T[0, i])
    line.set_ydata(traces.T[1, i])

initial_positions = get_positions(0)
initial_velocities = get_velocities(0, initial_positions)
initial_accelerations = get_accelerations(0)

annotations = [
    axes.annotate('O1', initial_positions.point_O1),
    axes.annotate('O2', initial_positions.point_O2),
    axes.annotate('O3', initial_positions.point_O3),
    axes.annotate('O4', initial_positions.point_O4),
    axes.annotate('A', initial_positions.point_A),
    axes.annotate('B', initial_positions.point_B),
    axes.annotate('C', initial_positions.point_C),
    axes.annotate('D', initial_positions.point_D),
    axes.annotate('E', initial_positions.point_E),
    axes.annotate('F', initial_positions.point_F),
    axes.annotate('G', initial_positions.point_G),
    axes.annotate('H', initial_positions.point_H)
]

velocity_vectors = axes.quiver(list(map(itemgetter(0), initial_positions)),
                               list(map(itemgetter(1), initial_positions)),
                               list(map(itemgetter(0), initial_velocities)),
                               list(map(itemgetter(1), initial_velocities)),
                               color='r')

acceleration_vectors = axes.quiver(list(map(itemgetter(0), initial_positions)),
                                   list(map(itemgetter(1), initial_positions)),
                                   list(map(itemgetter(0), initial_accelerations)),
                                   list(map(itemgetter(1), initial_accelerations)),
                                   color='b', scale=500)


def animate(frame_index):
    time = timeline[frame_index]
    positions = get_positions(time)
    velocities = get_velocities(time, positions)
    accelerations = get_accelerations(time)

    mechanism_line.set_xdata(list(map(itemgetter(0), (positions.point_O1,
                                                      positions.point_A,
                                                      positions.point_C,
                                                      positions.point_B,
                                                      positions.point_A,
                                                      positions.point_D,
                                                      positions.point_O2,
                                                      positions.point_D,
                                                      positions.point_E,
                                                      positions.point_O3,
                                                      positions.point_E,
                                                      positions.point_F,
                                                      positions.point_G,
                                                      positions.point_O4,
                                                      positions.point_G,
                                                      positions.point_H,
                                                      positions.point_F))))

    mechanism_line.set_ydata(list(map(itemgetter(1), (positions.point_O1,
                                                      positions.point_A,
                                                      positions.point_C,
                                                      positions.point_B,
                                                      positions.point_A,
                                                      positions.point_D,
                                                      positions.point_O2,
                                                      positions.point_D,
                                                      positions.point_E,
                                                      positions.point_O3,
                                                      positions.point_E,
                                                      positions.point_F,
                                                      positions.point_G,
                                                      positions.point_O4,
                                                      positions.point_G,
                                                      positions.point_H,
                                                      positions.point_F))))

    velocity_vectors.set_offsets(np.array(positions))
    velocity_vectors.set_UVC(list(map(itemgetter(0), velocities)), list(map(itemgetter(1), velocities)))

    acceleration_vectors.set_offsets(np.array(positions))
    acceleration_vectors.set_UVC(list(map(itemgetter(0), accelerations)), list(map(itemgetter(1), accelerations)))

    for annotation, descr in zip(annotations, positions):
        annotation.set_position(descr)

    return mechanism_line, *annotations, velocity_vectors, acceleration_vectors


anim = animation.FuncAnimation(figure, animate, frames=frames, interval=1000 * delta_time, blit=True)

plt.show()