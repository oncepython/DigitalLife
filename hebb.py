import numpy as np
import threading
import queue
import time

FPS = 20 # 帧率

class Hebb:
    def __init__(self, d_i: int, d_o: int, mu: float) -> None:
        # 8输入: 4方向食物 + 4方向障碍
        self.W = np.random.randn(d_o, d_i) * 0.1
        self.mu = mu

    def pred(self, i: np.ndarray) -> np.ndarray:
        return i @ self.W.T

    def learn(self, i: np.ndarray, target: np.ndarray, modulation: float) -> None:
        delta = self.mu * modulation * np.outer(target, i)
        self.W += delta

class Env:
    def __init__(self, size: int, n_food: int, n_obsta: int, mu: float) -> None:
        self.world = np.zeros(shape=(size, size))
        self.size = size

        for _ in range(n_food):
            x, y = np.random.randint(0, size, 2)
            self.world[x, y] = 1
        for _ in range(n_obsta):
            x, y = np.random.randint(0, size, 2)
            self.world[x, y] = -1

        self.agx = size // 2
        self.agy = size // 2
        self.model = Hebb(10, 4, mu)
        self.energy = 100
        self.running = True
        self.display_queue = queue.Queue(maxsize=1)
        self.step_count = 0

    def find_nearest(self, start_x: int, start_y: int, dx: int, dy: int, target_type: int) -> float:
        """沿方向(dx,dy)查找最近目标(食物=1或障碍=-1)的距离倒数"""
        x, y = start_x, start_y
        dist = 0
        max_dist = self.size

        while dist < max_dist:
            x = (x + dx) % self.size
            y = (y + dy) % self.size
            dist += 1

            if self.world[x, y] == target_type:
                return 1.0 / dist
            elif self.world[x, y] != 0 and self.world[x, y] != target_type:
                # 遇到其他类型阻挡（找食物时遇到障碍，或反之）
                return 0.0

        return 0.0

    def get_perception(self) -> np.ndarray:
        """8维输入: [上食,下食,左食,右食,上障,下障,左障,右障]"""
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上、下、左、右
        
        food_signals = [self.find_nearest(self.agx, self.agy, dx, dy, 1) for dx, dy in directions]
        obstacle_signals = [self.find_nearest(self.agx, self.agy, dx, dy, -1) for dx, dy in directions]
        
        return np.array(food_signals + obstacle_signals + [np.random.random(), self.energy / 100])

    def step(self) -> None:
        sta = self.get_perception()

        out = self.model.pred(sta)
        logits = out[:4]
        reward = out[-1]

        act = np.argmax(logits)

        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dx, dy = moves[act]
        nx, ny = (self.agx + dx) % self.size, (self.agy + dy) % self.size

        old_energy = self.energy
        cell = self.world[nx, ny]

        if cell == 1:
            self.energy += 1
            self.world[nx, ny] = 0
            self.agx, self.agy = nx, ny
        elif cell == -1:
            self.energy -= 1
        else:
            self.agx, self.agy = nx, ny
            self.energy -= 0.2

        delta_e = self.energy - old_energy
        modulation = delta_e

        target = np.zeros(4)
        target[act] = 1.0

        self.model.learn(sta, target, modulation)

        self.step_count += 1

        if self.step_count % 4 == 0:
            x, y = np.random.randint(0, self.size, 2)
            if self.world[x, y] == 0 and not (x == self.agx and y == self.agy):
                self.world[x, y] = 1

        try:
            self.display_queue.put_nowait((
                self.world.copy(), self.agx, self.agy,
                self.energy, sta, act, modulation
            ))
        except queue.Full:
            pass

    def show(self) -> None:
        cmap = {0: " ", 1: "\033[32m.\033[0m", -1: "\033[31m#\033[0m", 2: "\033[33m@\033[0m"}
        while self.running:
            try:
                world, agx, agy, energy, sta, act, mod = self.display_queue.get(timeout=0.1)
                n = world.copy()
                n[agx, agy] = 2

                buf = []
                for row in n:
                    buf.append("".join([cmap.get(int(i), "?") for i in row]))

                print("\033[2J\033[H", end="")
                print("\n".join(buf))
                print(f"Energy: {round(energy, 1)} | Step: {self.step_count}")
                print(f"Food(up/dn/lt/rt): [{sta[0]:.2f} {sta[1]:.2f} {sta[2]:.2f} {sta[3]:.2f}]")
                print(f"Obst(up/dn/lt/rt): [{sta[4]:.2f} {sta[5]:.2f} {sta[6]:.2f} {sta[7]:.2f}]")
            except queue.Empty:
                continue

    def compute_loop(self) -> None:
        while self.running:
            self.step()
            time.sleep(1/FPS)

env = Env(16, 20, 20, 0.1)

compute_thread = threading.Thread(target=env.compute_loop)
display_thread = threading.Thread(target=env.show)

compute_thread.start()
display_thread.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    env.running = False
    compute_thread.join()
    display_thread.join()
    print("\nStopped")
