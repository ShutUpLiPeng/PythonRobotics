"""

Path tracking simulation with Stanley steering control and PID speed control.

author: Atsushi Sakai (@Atsushi_twi)

Ref:
    - [Stanley: The robot that won the DARPA grand challenge](http://isl.ecst.csuchico.edu/DOCS/darpa2005/DARPA%202005%20Stanley.pdf)
    - [Autonomous Automobile Path Tracking](https://www.ri.cmu.edu/pub_files/2009/2/Automatic_Steering_Methods_for_Autonomous_Automobile_Path_Tracking.pdf)

"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import math
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from PathPlanning.CubicSpline import cubic_spline_planner

k = 1  # control gain
Kp = 1.0  # speed proportional gain
dt = 0.2  # [s] time difference
L = 2.9  # [m] Wheel base of vehicle
max_steer = np.radians(45.0)  # [rad] max steering angle

Move_Backward = True
Target_Speed = 5

# Vehicle parameters
LENGTH = 4.9  # [m]
WIDTH = 2.0  # [m]
BACKTOWHEEL = 1.0  # [m]
WHEEL_LEN = 0.3  # [m]
WHEEL_WIDTH = 0.2  # [m]
TREAD = 0.7  # [m]
WB = L  # [m]

show_animation = True

N_IND_SEARCH = 20

class State(object):
    """
    Class representing the state of a vehicle.

    :param x: (float) x-coordinate
    :param y: (float) y-coordinater
    :param yaw: (float) yaw angle
    :param v: (float) speed
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        """Instantiate the object."""
        super(State, self).__init__()
        self.x = x
        self.y = y
        self.yaw = normalize_angle(yaw)
        self.v = v
        self.backward = Move_Backward

    def update(self, acceleration, delta):
        """
        Update the state of the vehicle.

        Stanley Control uses bicycle model.

        :param acceleration: (float) Acceleration
        :param delta: (float) Steering
        """
        delta = np.clip(delta, -max_steer, max_steer)

        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += self.v / L * np.tan(delta) * dt
        self.yaw = normalize_angle(self.yaw)
        self.v += acceleration * dt
        # print("State(x, y, yaw, v) = ", self.x, self.y, self.yaw, self.v)


def pid_control(target, current):
    """
    Proportional control for the speed.

    :param target: (float)
    :param current: (float)
    :return: (float)
    """
    return Kp * (target - current)

def plot_car(x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):  # pragma: no cover

    outline = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
                        [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                         [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])

    rr_wheel = np.copy(fr_wheel)

    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                     [-math.sin(yaw), math.cos(yaw)]])
    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])

    fr_wheel = (fr_wheel.T.dot(Rot2)).T
    fl_wheel = (fl_wheel.T.dot(Rot2)).T
    fr_wheel[0, :] += WB
    fl_wheel[0, :] += WB

    fr_wheel = (fr_wheel.T.dot(Rot1)).T
    fl_wheel = (fl_wheel.T.dot(Rot1)).T

    outline = (outline.T.dot(Rot1)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T

    outline[0, :] += x
    outline[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    plt.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fr_wheel[0, :]).flatten(),
             np.array(fr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rr_wheel[0, :]).flatten(),
             np.array(rr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fl_wheel[0, :]).flatten(),
             np.array(fl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rl_wheel[0, :]).flatten(),
             np.array(rl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(x, y, "*")

def stanley_control(state, cx, cy, cyaw, last_target_idx):
    """
    Stanley steering control.

    :param state: (State object)
    :param cx: ([float])
    :param cy: ([float])
    :param cyaw: ([float])
    :param last_target_idx: (int)
    :return: (float, int)
    """
    global Target_Speed

    angle_diff = math.fabs(normalize_angle(state.yaw - normalize_angle(cyaw[np.minimum(last_target_idx + 50, len(cyaw) - 1)])))
    current_target_idx, error_front_axle, fx, fy = calc_target_index(state, cx, cy, cyaw, last_target_idx, state.backward)

    if last_target_idx >= current_target_idx:
        current_target_idx = last_target_idx

    # theta_e corrects the heading error
    if state.backward:
        theta_e = normalize_angle(normalize_angle(state.yaw + math.pi) - cyaw[current_target_idx])
    else:
        theta_e = normalize_angle(cyaw[current_target_idx] - state.yaw)
    
    # theta_d corrects the cross track error
    theta_d = np.arctan2(k * error_front_axle, math.fabs(Target_Speed))
    theta_d = np.clip(theta_d, -max_steer, max_steer)
    # Steering control
    delta = theta_e + theta_d

    delta = np.clip(delta, -max_steer, max_steer)

    return delta, current_target_idx, fx, fy

def normalize_angle(angle):
    """
    Normalize an angle to [-pi, pi].

    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle

def calc_target_index(state, cx, cy, cyaw, last_idx, backward = False):
    """
    Compute index in the trajectory list of the target.

    :param state: (State object)
    :param cx: [float]
    :param cy: [float]
    :return: (int, float)
    """
    # Calc front axle position
    if backward:
        fx = state.x + L * np.cos(state.yaw + math.pi)
        fy = state.y + L * np.sin(state.yaw + math.pi)
    else:
        fx = state.x + L * np.cos(state.yaw)
        fy = state.y + L * np.sin(state.yaw)

    # Search nearest point index
    start_index = np.maximum(last_idx-N_IND_SEARCH, 0)
    end_index = np.minimum(last_idx+N_IND_SEARCH, len(cx) - 1)
    dx = [fx - icx for icx in cx[start_index: end_index]]
    dy = [fy - icy for icy in cy[start_index: end_index]]
    d = np.hypot(dx, dy)
    target_idx = np.argmin(d)

    delta_x = dx[target_idx]
    delta_y = dy[target_idx]
    target_idx += (start_index + 1)

    # Project RMS error onto front axle vector
    if backward:
        ahead_axle_vec = [np.cos(cyaw[target_idx] + np.pi / 2),
              np.sin(cyaw[target_idx] + np.pi / 2)]
    else:
        ahead_axle_vec = [-np.cos(cyaw[target_idx] + np.pi / 2),
                      -np.sin(cyaw[target_idx] + np.pi / 2)]
    error_front_axle = np.dot([delta_x, delta_y], ahead_axle_vec)

    # print("error_front_axle = ", error_front_axle);
    return target_idx, error_front_axle, fx, fy


def get_straight_course(dl):
    ax = [0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    ay = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    return cx, cy, cyaw, ck


def get_straight_course2(dl):
    ax = [0.0, -10.0, -20.0, -40.0, -50.0, -60.0, -70.0]
    ay = [0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    return cx, cy, cyaw, ck


def get_straight_course3(dl):
    ax = [0.0, -10.0, -20.0, -40.0, -50.0, -60.0, -70.0]
    ay = [0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    cyaw = [i - math.pi for i in cyaw]

    return cx, cy, cyaw, ck


def get_forward_course(dl):
    ax = [0.0, 60.0, 125.0, 50.0, 75.0, 30.0, -10.0]
    ay = [0.0, 0.0, 50.0, 65.0, 30.0, 50.0, -20.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    return cx, cy, cyaw, ck

def get_turn_course(dl):
    ax = [0.0, 5.0, 10.0, 15.0, 15.0, 15.0, 20.0, 35, 50, 55, 55, 55, 65, 65, 65, 55, 40, 20, 10, 0]
    ay = [0.0, 0.0, 0.0, 5.0, 15.0, 30.0, 35.0, 35.0, 35.0, 30.0, 20.0, 5.0, -5.0, -20.0, -45.0, -50.0, -50.0, -50.0, -45.0, -45.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)
    return cx, cy, cyaw, ck

def get_curve_course(dl):
    ax = [0.0, 100.0, 100.0, 50.0, 60.0]
    ay = [0.0, 0.0, -30.0, -20.0, 0.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)
    return cx, cy, cyaw, ck

def get_switch_back_course(dl):
    ax = [0.0, 30.0, 6.0, 20.0, 35.0]
    ay = [0.0, 0.0, 20.0, 35.0, 20.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)
    ax = [35.0, 10.0, 0.0, 0.0]
    ay = [20.0, 30.0, 5.0, 0.0]
    cx2, cy2, cyaw2, ck2, s2 = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)
    # cyaw2 = [normalize_angle(i - math.pi) for i in cyaw2]
    cx.extend(cx2)
    cy.extend(cy2)
    cyaw.extend(cyaw2)
    ck.extend(ck2)
    return cx, cy, cyaw, ck

def get_shape_turn_course(dl):
    ax = [0.0, -50]
    ay = [0.0, 0.0]
    cx, cy, cyaw, ck, s1 = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)
    ax = [-50, 0]
    ay = [0, 25]
    cx2, cy2, cyaw2, ck2, s2 = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)
    ax = [0, -50]
    ay = [25, 25]
    cx3, cy3, cyaw3, ck3, s3 = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    cx.extend(cx2)
    cy.extend(cy2)
    cyaw.extend(cyaw2)
    ck.extend(ck2)
    cx.extend(cx3)
    cy.extend(cy3)
    cyaw.extend(cyaw3)
    ck.extend(ck3)
    return cx, cy, cyaw, ck

def main():

    dl = 0.1  # course tick
    # cx, cy, cyaw, ck = get_straight_course(dl)
    # cx, cy, cyaw, ck = get_straight_course2(dl)
    # cx, cy, cyaw, ck = get_straight_course3(dl)
    # cx, cy, cyaw, ck = get_forward_course(dl)
    # cx, cy, cyaw, ck = get_turn_course(dl)
    # cx, cy, cyaw, ck = get_switch_course(dl)
    cx, cy, cyaw, ck = get_switch_back_course(dl)
    # cx, cy, cyaw, ck = get_shape_turn_course(dl)

     # [m/s]
    global Target_Speed
    if Move_Backward :
        Target_Speed *= -1.0;

    max_simulation_time = 100.0

    # Initial state
    state = State(x=0.0, y=5.0, yaw= np.radians(-50), v=0.0)

    last_idx = len(cx) - 1
    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]
    ahead_x = []
    ahead_y = []
    desired_angle = [0.0]
    target_idx = 0
    last_index = 0

    while max_simulation_time >= time and last_idx > target_idx:
        target_speed = Target_Speed
        ai = pid_control(target_speed, state.v)
        di, target_idx, fx, fy = stanley_control(state, cx, cy, cyaw, last_index)
        last_index = target_idx
        state.update(ai, di)

        time += dt

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)
        ahead_x.append(fx)
        ahead_x=ahead_x[-50:]
        ahead_y.append(fy)
        ahead_y=ahead_y[-50:]
        desired_angle.append(di)

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(cx, cy, ".r", label="course")
            plt.plot(x, y, "-b", label="trajectory")
            plt.scatter(ahead_x, ahead_y, label="ahead pose")
            plt.plot(cx[target_idx], cy[target_idx], "xg", label="target")
            plot_car(state.x, state.y, state.yaw, steer=di)
            plt.axis("equal")
            plt.grid(True)
            plt.title("Speed[km/h]:" + str(state.v * 3.6)[:4] + ", car yaw:" + str(state.yaw / math.pi * 180)[:4] + ", closet yaw:" + str(cyaw[target_idx] / math.pi * 180)[:4])
            plt.pause(0.001)

    # Test
    assert last_idx >= target_idx, "Cannot reach goal"

    if show_animation:  # pragma: no cover
        plt.plot(cx, cy, ".r", label="course")
        plt.plot(x, y, "-b", label="trajectory")
        plt.legend()
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.axis("equal")
        plt.grid(True)

        plt.subplots(1)
        plt.plot(t, [iv * 3.6 for iv in v], "-r")
        plt.xlabel("Time[s]")
        plt.ylabel("Speed[km/h]")
        plt.grid(True)

        plt.subplots(1)
        plt.plot(t, [iv / 3.14 * 180 for iv in desired_angle], "-r")
        plt.xlabel("Time[s]")
        plt.ylabel("Angle[degree]")
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    main()
