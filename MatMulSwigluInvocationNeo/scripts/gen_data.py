#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

# import numpy as np
# import os


# def gen_golden_data():
#     M = 32768
#     N = 2048
#     K = 22016

#     x1_gm = np.random.randint(1, 10, [M, K]).astype(np.float16)
#     x2_gm = np.random.randint(1, 10, [N, K]).astype(np.float16)
#     golden = np.matmul(x1_gm, x2_gm.T).astype(np.float16)
#     os.system("mkdir -p input")
#     os.system("mkdir -p output")
#     x1_gm.tofile("./input/x1_gm.bin")
#     x2_gm.tofile("./input/x2_gm.bin")
#     golden.tofile("./output/golden.bin")


# if __name__ == "__main__":
#     gen_golden_data()

import numpy as np
import os

def silu_numpy_float16(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    sigmoid_x = 1 / (1 + np.exp(-x))
    result = x * sigmoid_x
    return result.astype(np.float16)


def gen_golden_data_simple():
    M = 1024
    K = 256
    N = 640 * 2

    input_a = np.random.randint(1, 10, [M, K]).astype(np.float16)
    input_b = np.random.randint(1, 10, [N, K]).astype(np.float16)
    golden = np.matmul(input_a, input_b.T)
    golden = silu_numpy_float16(golden)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_a.tofile("./input/x1_gm.bin")
    input_b.tofile("./input/x2_gm.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()