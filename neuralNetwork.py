import numpy as np
from progress.bar import IncrementalBar
import random
import csv
import time
import pickle
from multiprocessing import Pool

import activationFunctions as afs


class IncrementalBarCost(IncrementalBar):
    def __init__(self, net, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = net

    @property
    def cost(self):
        return str(round(self.net.last_cost, 5))


class neuralNetwork:
    def __init__(
        self,
        layers,
        input_filter,
        comparison_array_func,
        layers_file="layers.pickle",
        W_file="W.pickle",
        B_file="B.pickle",
        values_file="saved_values.csv",
        af=afs.ELU,
        afdot=afs.ELUDot,
    ):
        self.layers = layers
        self.W = [
            afs.af_weight_adjustment(
                af, np.random.randn(self.layers[i + 1], self.layers[i])
            )
            for i in range(len(self.layers) - 1)
        ]
        self.B = [np.zeros((self.layers[i], 1)) for i in range(1, len(self.layers))]
        self.a = [np.zeros((l, 1)) for l in self.layers]
        self.input_filter = input_filter
        self.last_answer = 0
        self.comparison_array_func = comparison_array_func
        self.layers_file = layers_file
        self.W_file = W_file
        self.B_file = B_file
        self.values_file = values_file
        self.af = af
        self.afdot = afdot
        self.last_cost = 0

    def set_inputs(self, input):
        self.a[0] = self.input_filter(input)

    def get_layer_z(self, layer_number):
        return (
            np.matmul(self.W[layer_number - 1], self.a[layer_number - 1])
            + self.B[layer_number - 1]
        )

    def calc_layer_a(self, layer_number):
        self.a[layer_number] = self.af(self.get_layer_z(layer_number))

    def calc_layers(self):
        for i in range(1, len(self.layers), 1):
            self.calc_layer_a(i)

    def run(self, img, ans):
        self.set_inputs(img)
        self.calc_layers()
        self.last_answer = ans

    def get_output(self):
        return self.a[-1]

    def get_error(self, answer):
        return (
            np.linalg.norm(self.get_output() - self.comparison_array_func(answer)) ** 2
        )

    def calc_cost(self, data):
        cost = 0
        for img, ans in zip(data[0], data[1]):
            self.run(img, ans)
            cost += self.get_error(ans)
        self.last_cost = cost / (2 * len(data[1]))

    def get_dcda_p(self, answer):
        dcda_p = [np.zeros(self.a[i].shape) for i in range(len(self.layers))]
        for i in range(len(self.layers) - 1):
            l = len(self.layers) - 1 - i
            if i == 0:
                dcda_p[l] = 2 * (self.a[l] - self.comparison_array_func(answer))
            else:
                x = np.matmul(
                    self.W[l].transpose(),
                    self.afdot(self.get_layer_z(l + 1)) * dcda_p[l + 1],
                )
                dcda_p[l] = x
        return dcda_p

    def get_dcdw(self, dcda_p):
        dcdw = [np.zeros(self.W[i].shape) for i in range(len(self.layers) - 1)]
        for i in range(len(self.layers) - 1):
            l = len(self.layers) - 1 - i
            dcdw[l - 1] = (
                self.a[l - 1].flatten() * self.afdot(self.get_layer_z(l)) * dcda_p[l]
            )
        return dcdw

    def get_dcdb(self, dcda_p):
        dcdb = [np.zeros(self.B[i].shape) for i in range(len(self.layers) - 1)]
        for i in range(len(self.layers) - 1):
            l = len(self.layers) - 1 - i
            dcdb[l - 1] = self.afdot(self.get_layer_z(l)) * dcda_p[l]
        return dcdb

    def calcBatch(self, imgAns):
        img, ans = imgAns
        self.run(img, ans)
        dcda_p = self.get_dcda_p(ans)
        dcdw_run = self.get_dcdw(dcda_p)
        dcdb_run = self.get_dcdb(dcda_p)
        return (dcdw_run, dcdb_run)

    def train(
        self,
        training_data,
        e=1e-3,
        repeats=1,
        batch_size=1000,
        func_after_it=None,
        save_net=False,
        save_values=False,
        process_count=None,
    ):
        b = int(len(training_data[1]) / batch_size)
        bar = IncrementalBarCost(
            self,
            "Training",
            suffix="%(percent)d%% [%(elapsed_td)s / %(eta_td)s] %(cost)s",
            max=repeats * b,
        )
        bar.goto(0)
        for ii in range(repeats):
            temp = list(zip(training_data[0], training_data[1]))
            random.shuffle(temp)
            training_data[0], training_data[1] = zip(*temp)
            for i in range(b):
                bar.goto(int(ii * b + i))
                dcdw_total, dcdb_total = [
                    np.zeros(self.W[x].shape) for x in range(len(self.layers) - 1)
                ], [np.zeros(self.B[x].shape) for x in range(len(self.layers) - 1)]
                with Pool(processes=process_count) as pool:
                    poolMap = pool.imap_unordered(
                        self.calcBatch,
                        zip(
                            training_data[0][
                                batch_size * i : batch_size * i + batch_size
                            ],
                            training_data[1][
                                batch_size * i : batch_size * i + batch_size
                            ],
                        ),
                    )
                    while True:
                        try:
                            result = next(poolMap)
                        except StopIteration:
                            break

                        for i in range(len(result[0])):
                            dcdw_total[i] += result[0][i]
                            dcdb_total[i] += result[1][i]

                for j in range(len(dcdw_total)):
                    self.W[j] -= e * dcdw_total[j] / batch_size
                    self.B[j] -= e * dcdb_total[j] / batch_size
            
                self.calc_cost(training_data[:300])
                bar.goto(int(ii * b + i + 1))
            if save_net:
                self.save_net()
            if save_values:
                self.save_values()
            if func_after_it:
                func_after_it()
        bar.goto(int(repeats * b))
        bar.finish()

    def save_net(self):
        with open(self.layers_file, "wb") as f:
            pickle.dump(self.layers, f, pickle.HIGHEST_PROTOCOL)
        with open(self.W_file, "wb") as f:
            pickle.dump(self.W, f, pickle.HIGHEST_PROTOCOL)
        with open(self.B_file, "wb") as f:
            pickle.dump(self.B, f, pickle.HIGHEST_PROTOCOL)

    def write_values(self, f):
        writer = csv.writer(f, delimiter=",", quotechar="|", lineterminator="\r")
        for i in range(len(self.a)):
            writer.writerow(self.a[i].flatten())
            writer.writerow([])
        writer.writerow(self.comparison_array_func(self.last_answer).flatten())

    def save_values(self):
        try:
            with open(self.values_file, "w") as f:
                self.write_values(f)
        except PermissionError:
            print("COULD NOT SAVE trying again")
            available = False
            while not available:
                time.sleep(1)
                try:
                    with open(self.values_file, "w") as f:
                        self.write_values(f)
                    available = True
                except PermissionError:
                    pass

    def load_net(self):
        print('LOADING NET')
        with open(self.layers_file, "rb") as f:
            self.layers = pickle.load(f)
        with open(self.W_file, "rb") as f:
            self.W = pickle.load(f)
        with open(self.B_file, "rb") as f:
            self.B = pickle.load(f)
        self.a = [np.zeros((l, 1)) for l in self.layers]
        print('NET LOADED')
