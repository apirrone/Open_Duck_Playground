import re
import requests
import logging
from typing import Dict, Any, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class CommandType(Enum):
    MOVE = "move"
    TURN = "turn"
    STOP = "stop"
    HEAD = "head"
    MODE = "mode"
    UNKNOWN = "unknown"


class DuckCommandParser:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url.rstrip('/')
        
        # Command patterns
        self.movement_patterns = {
            r'\b(go|move)\s+(forward|forwards?|ahead)\b': ('move', 'forward'),
            r'\b(go|move)\s+(backward|backwards?|back)\b': ('move', 'backward'),
            r'\b(go|move)\s+(left)\b': ('move', 'left'),
            r'\b(go|move)\s+(right)\b': ('move', 'right'),
            r'\b(forward|ahead)\b': ('move', 'forward'),
            r'\b(backward|back)\b': ('move', 'backward'),
            r'\b(left)\b(?!\s*turn)': ('move', 'left'),
            r'\b(right)\b(?!\s*turn)': ('move', 'right'),
        }
        
        self.turn_patterns = {
            r'\bturn\s+(left)\b': ('turn', 'left'),
            r'\bturn\s+(right)\b': ('turn', 'right'),
            r'\b(rotate|spin)\s+(left)\b': ('turn', 'left'),
            r'\b(rotate|spin)\s+(right)\b': ('turn', 'right'),
        }
        
        self.head_patterns = {
            r'\blook\s+(up|upward)\b': ('head', {'head_pitch': 0.3}),
            r'\blook\s+(down|downward)\b': ('head', {'head_pitch': -0.3}),
            r'\blook\s+(left)\b': ('head', {'head_yaw': -0.5}),
            r'\blook\s+(right)\b': ('head', {'head_yaw': 0.5}),
            r'\bhead\s+(up|upward)\b': ('head', {'head_pitch': 0.3}),
            r'\bhead\s+(down|downward)\b': ('head', {'head_pitch': -0.3}),
            r'\bhead\s+(left)\b': ('head', {'head_yaw': -0.5}),
            r'\bhead\s+(right)\b': ('head', {'head_yaw': 0.5}),
        }
        
        self.stop_patterns = {
            r'\b(stop|halt|freeze)\b': 'stop',
            r'\bemergency\s+stop\b': 'emergency_stop',
        }
        
        self.mode_patterns = {
            r'\b(head|look)\s+mode\b': 'head',
            r'\b(walk|locomotion|movement)\s+mode\b': 'locomotion',
            r'\bswitch\s+to\s+(head|look)\s+mode\b': 'head',
            r'\bswitch\s+to\s+(walk|locomotion|movement)\s+mode\b': 'locomotion',
        }
        
        # Speed modifiers
        self.speed_modifiers = {
            r'\b(slow|slowly|careful|carefully)\b': 0.3,
            r'\b(fast|quickly|quick)\b': 0.8,
            r'\b(normal|medium)\b': 0.5,
        }
    
    def parse_command(self, text: str) -> Tuple[CommandType, Dict[str, Any]]:
        text = text.lower().strip()
        logger.debug(f"Parsing command: {text}")
        
        # Check for stop commands first (highest priority)
        for pattern, command in self.stop_patterns.items():
            if re.search(pattern, text):
                return CommandType.STOP, {'command': command}
        
        # Check for mode switching
        for pattern, mode in self.mode_patterns.items():
            if re.search(pattern, text):
                return CommandType.MODE, {'mode': mode}
        
        # Check for movement commands
        for pattern, (cmd_type, direction) in self.movement_patterns.items():
            if re.search(pattern, text):
                speed = self._extract_speed(text)
                return CommandType.MOVE, {'direction': direction, 'speed': speed}
        
        # Check for turn commands
        for pattern, (cmd_type, direction) in self.turn_patterns.items():
            if re.search(pattern, text):
                speed = self._extract_speed(text)
                return CommandType.TURN, {'direction': direction, 'speed': speed}
        
        # Check for head commands
        for pattern, (cmd_type, params) in self.head_patterns.items():
            if re.search(pattern, text):
                return CommandType.HEAD, params
        
        return CommandType.UNKNOWN, {}
    
    def _extract_speed(self, text: str) -> Optional[float]:
        for pattern, speed in self.speed_modifiers.items():
            if re.search(pattern, text):
                return speed
        return None
    
    def execute_command(self, text: str) -> Dict[str, Any]:
        cmd_type, params = self.parse_command(text)
        
        if cmd_type == CommandType.UNKNOWN:
            logger.warning(f"Unknown command: {text}")
            return {"success": False, "message": "Command not recognized"}
        
        try:
            if cmd_type == CommandType.STOP:
                return self._execute_stop(params['command'])
            elif cmd_type == CommandType.MODE:
                return self._execute_mode_switch(params['mode'])
            elif cmd_type == CommandType.MOVE:
                return self._execute_move(params['direction'], params.get('speed'))
            elif cmd_type == CommandType.TURN:
                return self._execute_turn(params['direction'], params.get('speed'))
            elif cmd_type == CommandType.HEAD:
                return self._execute_head(params)
            
        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            return {"success": False, "message": f"API error: {str(e)}"}
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}
    
    def _execute_stop(self, stop_type: str) -> Dict[str, Any]:
        endpoint = f"/command/{stop_type.replace('_', '_')}"
        response = requests.post(f"{self.api_url}{endpoint}")
        response.raise_for_status()
        logger.info(f"Executed {stop_type}")
        return {"success": True, "message": response.json().get("message", "Stopped")}
    
    def _execute_mode_switch(self, mode: str) -> Dict[str, Any]:
        endpoint = "/control_mode"
        response = requests.post(f"{self.api_url}{endpoint}", json={"mode": mode})
        response.raise_for_status()
        logger.info(f"Switched to {mode} mode")
        return {"success": True, "message": f"Switched to {mode} mode"}
    
    def _execute_move(self, direction: str, speed: Optional[float] = None) -> Dict[str, Any]:
        endpoint = f"/command/move/{direction}"
        params = {"speed": speed} if speed is not None else {}
        response = requests.post(f"{self.api_url}{endpoint}", params=params)
        response.raise_for_status()
        logger.info(f"Moving {direction}" + (f" at speed {speed}" if speed else ""))
        return {"success": True, "message": response.json().get("message", f"Moving {direction}")}
    
    def _execute_turn(self, direction: str, speed: Optional[float] = None) -> Dict[str, Any]:
        endpoint = f"/command/turn/{direction}"
        params = {"speed": speed} if speed is not None else {}
        response = requests.post(f"{self.api_url}{endpoint}", params=params)
        response.raise_for_status()
        logger.info(f"Turning {direction}" + (f" at speed {speed}" if speed else ""))
        return {"success": True, "message": response.json().get("message", f"Turning {direction}")}
    
    def _execute_head(self, params: Dict[str, float]) -> Dict[str, Any]:
        endpoint = "/command/head"
        # Fill in defaults for unspecified parameters
        head_command = {
            "neck_pitch": params.get("neck_pitch", 0.0),
            "head_pitch": params.get("head_pitch", 0.0),
            "head_yaw": params.get("head_yaw", 0.0),
            "head_roll": params.get("head_roll", 0.0),
        }
        response = requests.post(f"{self.api_url}{endpoint}", json=head_command)
        response.raise_for_status()
        logger.info(f"Head command executed: {params}")
        return {"success": True, "message": "Head position updated"}