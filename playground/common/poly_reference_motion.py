import jax.numpy as jp
from jax import vmap
import pickle


# dimensions_names = [
#     0  "pos left_hip_yaw",
#     1  "pos left_hip_roll",
#     2  "pos left_hip_pitch",
#     3  "pos left_knee",
#     4  "pos left_ankle",
#     5  "pos neck_pitch",
#     6  "pos head_pitch",
#     7  "pos head_yaw",
#     8  "pos head_roll",
#     9  "pos left_antenna",
#     10 "pos right_antenna",
#     11 "pos right_hip_yaw",
#     12 "pos right_hip_roll",
#     13 "pos right_hip_pitch",
#     14 "pos right_knee",
#     15 "pos right_ankle",

#     16 "vel left_hip_yaw",
#     17 "vel left_hip_roll",
#     18 "vel left_hip_pitch",
#     19 "vel left_knee",
#     20 "vel left_ankle",
#     21 "vel neck_pitch",
#     22 "vel head_pitch",
#     23 "vel head_yaw",
#     24 "vel head_roll",
#     25 "vel left_antenna",
#     26 "vel right_antenna",
#     27 "vel right_hip_yaw",
#     28 "vel right_hip_roll",
#     29 "vel right_hip_pitch",
#     30 "vel right_knee",
#     31 "vel right_ankle",

#     32 "foot_contacts left",
#     33 "foot_contacts right",

#     34 "base_linear_vel x",
#     35 "base_linear_vel y",
#     36 "base_linear_vel z",

#     37 "base_angular_vel x",
#     38 "base_angular_vel y",
#     39 "base_angular_vel z",
# ]


class PolyReferenceMotion:
    def __init__(self, polynomial_coefficients: str):
        data = pickle.load(open(polynomial_coefficients, "rb"))
        # data = json.load(open(polynomial_coefficients))
        self.dx_range = [0, 0]
        self.dy_range = [0, 0]
        self.dtheta_range = [0, 0]
        self.dxs = []
        self.dys = []
        self.dthetas = []
        self.data_array = []
        self.period = None
        self.fps = None
        self.frame_offsets = None
        self.startend_double_support_ratio = None
        self.start_offset = None
        self.nb_steps_in_period = None

        self.process(data)

    def process(self, data):
        print("[Poly ref data] Processing ...")
        _data = {}
        for name in data.keys():
            split = name.split("_")
            dx = float(split[0])
            dy = float(split[1])
            dtheta = float(split[2])

            if self.period is None:
                self.period = data[name]["period"]
                self.fps = data[name]["fps"]
                self.frame_offsets = data[name]["frame_offsets"]
                self.startend_double_support_ratio = data[name][
                    "startend_double_support_ratio"
                ]
                self.start_offset = int(self.startend_double_support_ratio * self.fps)
                self.nb_steps_in_period = int(self.period * self.fps)

            if dx not in self.dxs:
                self.dxs.append(dx)

            if dy not in self.dys:
                self.dys.append(dy)

            if dtheta not in self.dthetas:
                self.dthetas.append(dtheta)

            self.dx_range = [min(dx, self.dx_range[0]), max(dx, self.dx_range[1])]
            self.dy_range = [min(dy, self.dy_range[0]), max(dy, self.dy_range[1])]
            self.dtheta_range = [
                min(dtheta, self.dtheta_range[0]),
                max(dtheta, self.dtheta_range[1]),
            ]

            if dx not in _data:
                _data[dx] = {}

            if dy not in _data[dx]:
                _data[dx][dy] = {}

            if dtheta not in _data[dx][dy]:
                _data[dx][dy][dtheta] = data[name]

            _coeffs = data[name]["coefficients"]

            coeffs = []
            for k, v in _coeffs.items():
                coeffs.append(jp.flip(jp.array(v)))
            _data[dx][dy][dtheta] = coeffs

        # print(self.dtheta_range)
        # exit()

        self.dxs = sorted(self.dxs)
        self.dys = sorted(self.dys)
        self.dthetas = sorted(self.dthetas)

        nb_dx = len(self.dxs)
        nb_dy = len(self.dys)
        nb_dtheta = len(self.dthetas)

        self.data_array = nb_dx * [None]
        for x, dx in enumerate(self.dxs):
            self.data_array[x] = nb_dy * [None]
            for y, dy in enumerate(self.dys):
                self.data_array[x][y] = nb_dtheta * [None]
                for th, dtheta in enumerate(self.dthetas):
                    self.data_array[x][y][th] = jp.array(_data[dx][dy][dtheta])

        self.data_array = jp.array(self.data_array)

        print("[Poly ref data] Done processing")

    def vel_to_index(self, dx, dy, dtheta):

        dx = jp.clip(dx, self.dx_range[0], self.dx_range[1])
        dy = jp.clip(dy, self.dy_range[0], self.dy_range[1])
        dtheta = jp.clip(dtheta, self.dtheta_range[0], self.dtheta_range[1])

        ix = jp.argmin(jp.abs(jp.array(self.dxs) - dx))
        iy = jp.argmin(jp.abs(jp.array(self.dys) - dy))
        itheta = jp.argmin(jp.abs(jp.array(self.dthetas) - dtheta))

        return ix, iy, itheta

    def sample_polynomial(self, t, coeffs):
        return vmap(lambda c: jp.polyval(c, t))(coeffs)

    def get_reference_motion(self, dx, dy, dtheta, i):
        ix, iy, itheta = self.vel_to_index(dx, dy, dtheta)
        t = i % self.nb_steps_in_period / self.nb_steps_in_period
        t = jp.clip(t, 0.0, 1.0)  # safeguard
        ret = self.sample_polynomial(t, self.data_array[ix][iy][itheta])
        return ret


if __name__ == "__main__":

    PRM = PolyReferenceMotion(
        "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
    )
    vals = []
    total = 10 * PRM.nb_steps_in_period
    # select_dim = 
    for i in range(total):
        vals.append(PRM.get_reference_motion(-0.05, -0.0, -0.0, i)[34]) # 32

    # plot
    import matplotlib.pyplot as plt
    import numpy as np

    ts = np.arange(0, total)
    plt.plot(ts, vals)
    plt.show()
