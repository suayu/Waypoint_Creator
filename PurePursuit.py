
import os.path

import numpy as np
from matplotlib import pyplot as plt
import matplotlib

import runtime
import copy

matplotlib.use('Agg')

# 角度使用弧度制

class PurePursuit:
    def __init__(self, lam=runtime.lam, c=runtime.c, L=runtime.L):
        self.lam = lam
        self.c = c
        self.L = L

    def one_step(self, ref_path: np.ndarray, ego_state: np.ndarray) -> np.ndarray:
        """
        1. 计算预瞄距离
        2. 在ref_path上找一个预瞄点(简单实现中,是在ref_path上找一个离ld最近的点)
        3. 根据预瞄点计算控制指令
        :param ref_path: ((x,y)...)
        :param ego_state: (x,y,yaw,v)
        :return:
        """
        ## 计算预瞄距离
        ld = self.lam * ego_state[3] + self.c

        ## 寻找预瞄点
        dis = np.linalg.norm(ref_path - ego_state[:2], 2, axis=1)
        start_idx = np.argmin(dis)
        dis = ld - dis

        ## 给距离大于ld和之前的点加mask
        dis[dis < 0] = float('inf')  
        dis[:start_idx] = float('inf')

        ref_point = ref_path[np.argmin(dis)]

        ## 计算控制指令
        alpha = np.arctan2(ref_point[1] - ego_state[1], ref_point[0] - ego_state[0]) - ego_state[2]
        delta = np.arctan2(2 * self.L * np.sin(alpha), ld)
        return delta

class SimpleTest:
    class CarModel:
        def __init__(self, initial_state=np.array([0, 0, 0, 1]).astype(np.float64), L=runtime.L, dt=runtime.dt):

            # state: x, y, yaw, v
            self.state = initial_state 
            self.L = L
            self.dt = dt

        def update_state(self, accel, delta):
            self.state[0] = self.state[0] + self.state[3] * np.cos(self.state[2]) * self.dt
            self.state[1] = self.state[1] + self.state[3] * np.sin(self.state[2]) * self.dt
            self.state[2] = self.state[2] + self.state[3] / self.L * np.tan(delta) * self.dt
            self.state[3] = self.state[3] + accel * self.dt

    def __init__(self, initial_state=np.array([0, 0, -0.5 * np.pi, 1]).astype(np.float64),
                 lam=runtime.lam, c=runtime.c, L=runtime.L, dt=runtime.dt):
        self.model = SimpleTest.CarModel(initial_state, L, dt)
        self.pure_pursuit = PurePursuit(lam, c, L)

    def offline_test(self, path, test_cnt, x_lim=runtime.x_lim, y_lim=runtime.y_lim,
                     result_dir=runtime.save_fig_dir,result_name = runtime.save_fig_name):
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        save_path = os.path.join(result_dir,result_name)
        # plt.xlim([-x_lim, x_lim])
        # plt.ylim([-y_lim, y_lim])
        past = path[0]
        for idx, p in enumerate(path):
            if idx > 0:
                # plt.plot([past[0], p[0]], [past[1], p[1]], c='black')
                past = p
        
        states = []
        past_s = self.model.state
        for i in range(test_cnt):
            delta = self.pure_pursuit.one_step(path, self.model.state)
            self.model.update_state(0, delta)
            new_state = copy.deepcopy(self.model.state)
            states.append(new_state)
            # plt.plot([past_s[0], self.model.state[0]], [past_s[1], self.model.state[1]], c='red')
            past_s = self.model.state.copy()
        # plt.savefig(save_path)

        return states

    @staticmethod
    def simple_read(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        path = []
        for l in lines:
            str_array = l.split('(')[1].split(')')[0].split(',')
            p = np.array([float(str_array[0]), float(str_array[1])])
            path.append(p)
        return np.array(path)
    
    def get_path(x, y):
        return np.concatenate((x.reshape(-1,1),y.reshape(-1,1)),axis=1)
