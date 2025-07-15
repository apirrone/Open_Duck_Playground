import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
import time
import logging

logger = logging.getLogger(__name__)


class ControlMode(Enum):
    LOCOMOTION = "locomotion"
    HEAD = "head"


@dataclass
class CommandLimits:
    linear_velocity_x: Tuple[float, float] = (-0.2, 0.2)
    linear_velocity_y: Tuple[float, float] = (-0.2, 0.2)
    angular_velocity: Tuple[float, float] = (-1.0, 1.0)
    neck_pitch: Tuple[float, float] = (-0.34, 1.1)
    head_pitch: Tuple[float, float] = (-0.78, 0.78)
    head_yaw: Tuple[float, float] = (-1.5, 1.5)
    head_roll: Tuple[float, float] = (-0.5, 0.5)


class CommandHandler:
    def __init__(self, limits: Optional[CommandLimits] = None):
        self.limits = limits or CommandLimits()
        self.commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.control_mode = ControlMode.LOCOMOTION
        self.lock = threading.Lock()
        
    def get_commands(self) -> List[float]:
        with self.lock:
            return self.commands.copy()
    
    def set_control_mode(self, mode: ControlMode):
        with self.lock:
            self.control_mode = mode
    
    def get_control_mode(self) -> ControlMode:
        with self.lock:
            return self.control_mode
    
    def set_locomotion_command(self, 
                              forward: float = 0.0, 
                              lateral: float = 0.0, 
                              angular: float = 0.0):
        with self.lock:
            self.commands[0] = np.clip(forward, *self.limits.linear_velocity_x)
            self.commands[1] = np.clip(lateral, *self.limits.linear_velocity_y)
            self.commands[2] = np.clip(angular, *self.limits.angular_velocity)
    
    def set_head_command(self,
                        neck_pitch: float = 0.0,
                        head_pitch: float = 0.0,
                        head_yaw: float = 0.0,
                        head_roll: float = 0.0):
        with self.lock:
            self.commands[3] = np.clip(neck_pitch, *self.limits.neck_pitch)
            self.commands[4] = np.clip(head_pitch, *self.limits.head_pitch)
            self.commands[5] = np.clip(head_yaw, *self.limits.head_yaw)
            self.commands[6] = np.clip(head_roll, *self.limits.head_roll)
    
    def stop(self):
        with self.lock:
            self.commands[:3] = [0.0, 0.0, 0.0]
    
    def reset_head(self):
        with self.lock:
            self.commands[3:] = [0.0, 0.0, 0.0, 0.0]
    
    def emergency_stop(self):
        with self.lock:
            self.commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    def move_forward(self, speed: float = None):
        speed = speed if speed is not None else self.limits.linear_velocity_x[1]
        self.set_locomotion_command(forward=speed)
    
    def move_backward(self, speed: float = None):
        speed = speed if speed is not None else abs(self.limits.linear_velocity_x[0])
        self.set_locomotion_command(forward=-speed)
    
    def move_left(self, speed: float = None):
        speed = speed if speed is not None else self.limits.linear_velocity_y[1]
        self.set_locomotion_command(lateral=speed)
    
    def move_right(self, speed: float = None):
        speed = speed if speed is not None else abs(self.limits.linear_velocity_y[0])
        self.set_locomotion_command(lateral=-speed)
    
    def turn_left(self, speed: float = None):
        speed = speed if speed is not None else self.limits.angular_velocity[1]
        self.set_locomotion_command(angular=speed)
    
    def turn_right(self, speed: float = None):
        speed = speed if speed is not None else abs(self.limits.angular_velocity[0])
        self.set_locomotion_command(angular=-speed)
    
    def process_keyboard_input(self, keycode: int) -> bool:
        if keycode == 32: # space - stop
            self.stop()
            return True
        if keycode == 72:  # h - toggle control mode
            new_mode = (ControlMode.HEAD if self.control_mode == ControlMode.LOCOMOTION 
                       else ControlMode.LOCOMOTION)
            self.set_control_mode(new_mode)
            return True
        
        if self.control_mode == ControlMode.LOCOMOTION:
            if keycode == 265:  # arrow up
                self.move_forward()
            elif keycode == 264:  # arrow down
                self.move_backward()
            elif keycode == 263:  # arrow left
                self.move_left()
            elif keycode == 262:  # arrow right
                self.move_right()
            elif keycode == 81:  # q
                self.turn_left()
            elif keycode == 69:  # e
                self.turn_right()
            else:
                return False
        else:  # HEAD mode
            if keycode == 265:  # arrow up
                self.set_head_command(head_pitch=self.limits.head_pitch[1])
            elif keycode == 264:  # arrow down
                self.set_head_command(head_pitch=self.limits.head_pitch[0])
            elif keycode == 263:  # arrow left
                self.set_head_command(head_yaw=self.limits.head_yaw[1])
            elif keycode == 262:  # arrow right
                self.set_head_command(head_yaw=self.limits.head_yaw[0])
            elif keycode == 81:  # q
                self.set_head_command(head_roll=self.limits.head_roll[1])
            elif keycode == 69:  # e
                self.set_head_command(head_roll=self.limits.head_roll[0])
            else:
                return False
        
        return True
    
    def get_status(self) -> Dict:
        with self.lock:
            return {
                "commands": {
                    "linear_velocity_x": self.commands[0],
                    "linear_velocity_y": self.commands[1],
                    "angular_velocity": self.commands[2],
                    "neck_pitch": self.commands[3],
                    "head_pitch": self.commands[4],
                    "head_yaw": self.commands[5],
                    "head_roll": self.commands[6]
                },
                "control_mode": self.control_mode.value,
                "limits": {
                    "linear_velocity_x": self.limits.linear_velocity_x,
                    "linear_velocity_y": self.limits.linear_velocity_y,
                    "angular_velocity": self.limits.angular_velocity,
                    "neck_pitch": self.limits.neck_pitch,
                    "head_pitch": self.limits.head_pitch,
                    "head_yaw": self.limits.head_yaw,
                    "head_roll": self.limits.head_roll
                },
            }