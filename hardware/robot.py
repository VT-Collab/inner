# socket client for the Franka controller
import time
import socket
import numpy as np
from termcolor import colored

STATE_LENGTHS = {55: "velocity", 7: "position"}


class FrankaResearch3:

    def __init__(self, host='172.16.0.3', robot_port=8080, gripper_port=8081, robot_home=None, max_speed=0.5):
        self.host = host
        print(colored("[Panda] ", "blue") + 'Connecting to robot...')
        self.conn = self.connect(robot_port)
        print(colored("[Panda] ", "green") + 'Connection complete')

        print(colored("[Panda] ", "blue") + 'Connecting to gripper')
        self.conn_gripper = self.connect(gripper_port)
        print(colored("[Panda] ", "green") + 'Connection complete')

        self.robot_home = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4]) if robot_home is None else np.asarray(robot_home)[:7]
        self.gripper_home = 'o'
        self.num_joints = 7
        # largest joint speed ever sent in velocity mode (rad/s)
        self.max_speed = max_speed
        self.buf, self.state_length = "", None
        self.readState()
        self.mode = STATE_LENGTHS[self.state_length]
        print(colored("[Panda] ", "green") + f'{self.mode} controller ({self.state_length} values per state)')
        self.stop()

    def connect(self, port):
        """wait for the controller to connect"""
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.host, port))
        s.listen()
        conn, addr = s.accept()
        return conn

    def send2gripper(self, command=''):
        """send 'o' (open) or 'c' (close)"""
        if command == '':
            command = self.gripper_home
        send_msg = "s," + command + ","
        self.conn_gripper.send(send_msg.encode())

    def _send(self, values):
        send_msg = np.array2string(np.asarray(values), precision=5, separator=',', suppress_small=True)[1:-1]
        if send_msg == '0.,0.,0.,0.,0.,0.,0.':
            send_msg = '0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000'
        send_msg = "s," + send_msg + ","
        self.conn.send(send_msg.encode())

    def move(self, q, target, dt):
        """move from q toward target over dt seconds"""
        if self.mode == "velocity":
            # scaled, not clipped per joint, so the arm keeps its direction
            qdot = (np.asarray(target) - q) / dt
            self._send(qdot * min(1.0, self.max_speed / max(np.abs(qdot).max(), 1e-12)))
        else:
            self._send(target)

    def stop(self, q=None):
        """stand still, position mode holds q"""
        if self.mode == "velocity":
            self._send(np.zeros(7))
        elif q is not None:
            self._send(q)

    def _last_state(self):
        """newest complete state in the buffer"""
        tokens = self.buf.split(",")
        starts = [i for i, t in enumerate(tokens) if t == "s"]
        if self.state_length is None:
            for i, j in zip(starts, starts[1:] + [len(tokens) - 1]):
                if j - i - 1 in STATE_LENGTHS and (j < len(tokens) - 1 or tokens[-1] == ""):
                    self.state_length = j - i - 1
                    break
            else:
                return None
        n = self.state_length
        for i in reversed(starts):
            if len(tokens) > i + n + 1:
                values = tokens[i + 1:i + 1 + n]
                self.buf = ",".join(tokens[i + 1 + n:])
                return np.asarray(values, dtype=np.float32)
        return None

    def readState(self):
        while True:
            data = self.conn.recv(65536)
            if not data:
                raise ConnectionError("robot controller closed the connection")
            self.buf += data.decode()
            self.conn.setblocking(False)
            try:
                while True:
                    data = self.conn.recv(65536)
                    if not data:
                        break
                    self.buf += data.decode()
            except BlockingIOError:
                pass
            finally:
                self.conn.setblocking(True)
            state = self._last_state()
            if state is not None:
                return {"q": state[:7]}

    def go2position(self, goal=False, total_time=20.0):
        """go to goal in joint space, at most 0.1 rad per step per joint"""
        if goal is False:
            goal = self.robot_home
        goal = np.asarray(goal)
        assert len(goal) == 7
        start_time = time.time()
        q = self.readState()["q"]
        while np.linalg.norm(q - goal) > 0.01 and time.time() - start_time < total_time:
            step = np.clip(goal - q, -0.1, 0.1)
            if self.mode == "velocity":
                self._send(step)
            else:
                self._send(q + step)
            q = self.readState()["q"]
        self.stop(q)
        if np.linalg.norm(q - goal) > 0.01:
            print(colored("[Panda] ", "yellow") + f'Not at the goal after {total_time:g} s (off by {np.linalg.norm(q - goal):.3f} rad)')
