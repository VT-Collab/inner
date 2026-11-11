# receives camera frames over zmq
import time
import struct
from threading import Thread
import zmq
import numpy as np


class CamerasSubscriber():
    """subscribes to camera topics and keeps the latest frame of each"""
    def __init__(self, topics, server_addr='localhost', port=8082):
        self.topics = topics
        self.server_addr = server_addr
        self.port = port

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.connect(f"tcp://{self.server_addr}:{self.port}")

        for topic in self.topics:
            self._socket.setsockopt(zmq.SUBSCRIBE, topic.encode(encoding="utf-8"))

        self.frames = {}

    def receive_messages(self):
        self.done = False
        while not self.done:
            [topic, message] = self._socket.recv_multipart()
            topic = topic.decode(encoding='utf-8')

            if 'img_rgb' in topic:
                h, w, c = struct.unpack(">III", message[:12])
                img = np.frombuffer(message[12:], dtype=np.uint8).reshape(h, w, c)
            elif 'img_depth' in topic:
                h, w = struct.unpack(">II", message[:8])
                img = np.frombuffer(message[8:], dtype=np.uint16).reshape(h, w)

            self.frames[topic] = img

    def get_last_frames(self):
        """latest frame per topic, or None until every topic has arrived"""
        if self.frames == {} or len(self.frames.keys()) != len(self.topics):
            return None
        return self.frames

    def start_thread(self):
        img_recv_thread = Thread(target=self.receive_messages)
        img_recv_thread.daemon = True
        img_recv_thread.start()

        time.sleep(1)
        print("[*] Waiting to receive first message")
        start_time = time.time()
        frames = None
        steps = 0
        while frames is None:
            frames = self.get_last_frames()
            steps += 1
            if (steps % 10000000) == 0:
                print(f"[*] Expecting frames from: {self.topics}, but only got {self.frames.keys()}, waiting time: {time.time() - start_time:.2f}s...")

    def close_subscriber(self):
        self.done = True
        time.sleep(1)
        self._socket.close()
        self._context.term()
