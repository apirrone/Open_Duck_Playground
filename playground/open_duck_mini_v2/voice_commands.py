import re
import requests
import logging
from typing import Dict, Any, Optional, Tuple
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class CommandType(Enum):
    MOVE = "move"
    TURN = "turn"
    STOP = "stop"
    HEAD = "head"
    MODE = "mode"
    AUTO_STOP = "auto_stop"
    UNKNOWN = "unknown"


class DuckCommandParser:
    def __init__(self, api_url: str = "http://localhost:8000", timeout: float = 5.0, auto_stop_delay: float = 5.0):
        self.api_url = api_url.rstrip('/')
        self.timeout = timeout  # Request timeout in seconds
        self.auto_stop_delay = auto_stop_delay  # Auto-stop delay in seconds
        self.auto_stop_timer = None  # Current auto-stop timer
        self.auto_stop_enabled = True
        
        # Test connection to API
        try:
            response = requests.get(f"{self.api_url}/health", timeout=2.0)
            response.raise_for_status()
            logger.info(f"Successfully connected to API at {self.api_url}")
        except Exception as e:
            logger.warning(f"Could not connect to API at {self.api_url}: {e}")
            logger.warning("Make sure mujoco_with_api.py is running first!")
        
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
        
        self.auto_stop_patterns = {
            r'\b(disable|turn\s+off)\s+auto[\s-]?stop\b': ('disable', None),
            r'\b(enable|turn\s+on)\s+auto[\s-]?stop\b': ('enable', None),
            r'\bset\s+auto[\s-]?stop\s+(?:to\s+)?(\d+(?:\.\d+)?)\s*(?:seconds?)?\b': ('timeout', None),
            r'\bauto[\s-]?stop\s+(\d+(?:\.\d+)?)\s*(?:seconds?)?\b': ('timeout', None),
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
        
        # Check for auto-stop commands
        for pattern, (action, _) in self.auto_stop_patterns.items():
            match = re.search(pattern, text)
            if match:
                if action == 'timeout':
                    timeout = float(match.group(1))
                    return CommandType.AUTO_STOP, {'action': 'timeout', 'timeout': timeout}
                else:
                    return CommandType.AUTO_STOP, {'action': action}
        
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
    
    def _cancel_auto_stop(self):
        """Cancel any pending auto-stop timer"""
        if self.auto_stop_timer and self.auto_stop_timer.is_alive():
            self.auto_stop_timer.cancel()
            self.auto_stop_timer = None
    
    def _schedule_auto_stop(self):
        """Schedule an auto-stop after the configured delay"""
        if not self.auto_stop_enabled:
            return
            
        self._cancel_auto_stop()  # Cancel any existing timer
        
        def auto_stop():
            logger.info("Auto-stop triggered")
            try:
                response = requests.post(f"{self.api_url}/command/stop", timeout=self.timeout)
                response.raise_for_status()
                logger.info("Auto-stop executed successfully")
            except Exception as e:
                logger.error(f"Auto-stop failed: {e}")
        
        self.auto_stop_timer = threading.Timer(self.auto_stop_delay, auto_stop)
        self.auto_stop_timer.daemon = True
        self.auto_stop_timer.start()
        logger.debug(f"Auto-stop scheduled in {self.auto_stop_delay} seconds")
    
    def execute_command(self, text: str) -> Dict[str, Any]:
        cmd_type, params = self.parse_command(text)
        
        if cmd_type == CommandType.UNKNOWN:
            logger.warning(f"Unknown command: {text}")
            # Don't cancel auto-stop for unknown commands
            return {"success": False, "message": "Command not recognized"}
        
        # Only cancel auto-stop for actual commands that affect movement
        if cmd_type in [CommandType.STOP, CommandType.MOVE, CommandType.TURN]:
            self._cancel_auto_stop()
        
        try:
            if cmd_type == CommandType.STOP:
                return self._execute_stop(params['command'])
            elif cmd_type == CommandType.MODE:
                return self._execute_mode_switch(params['mode'])
            elif cmd_type == CommandType.MOVE:
                result = self._execute_move(params['direction'], params.get('speed'))
                if result['success']:
                    self._schedule_auto_stop()
                return result
            elif cmd_type == CommandType.TURN:
                result = self._execute_turn(params['direction'], params.get('speed'))
                if result['success']:
                    self._schedule_auto_stop()
                return result
            elif cmd_type == CommandType.HEAD:
                return self._execute_head(params)
            elif cmd_type == CommandType.AUTO_STOP:
                return self._execute_auto_stop(params)
            
        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            return {"success": False, "message": f"API error: {str(e)}"}
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}
    
    def _execute_stop(self, stop_type: str) -> Dict[str, Any]:
        endpoint = f"/command/{stop_type.replace('_', '_')}"
        try:
            response = requests.post(f"{self.api_url}{endpoint}", timeout=self.timeout)
            response.raise_for_status()
            logger.info(f"Executed {stop_type}")
            return {"success": True, "message": response.json().get("message", "Stopped")}
        except requests.Timeout:
            logger.error(f"Request timeout for {stop_type}")
            return {"success": False, "message": "Request timed out"}
        except Exception as e:
            logger.error(f"Failed to execute {stop_type}: {e}")
            return {"success": False, "message": str(e)}
    
    def _execute_mode_switch(self, mode: str) -> Dict[str, Any]:
        endpoint = "/control_mode"
        try:
            response = requests.post(f"{self.api_url}{endpoint}", json={"mode": mode}, timeout=self.timeout)
            response.raise_for_status()
            logger.info(f"Switched to {mode} mode")
            return {"success": True, "message": f"Switched to {mode} mode"}
        except requests.Timeout:
            logger.error(f"Request timeout for mode switch")
            return {"success": False, "message": "Request timed out"}
        except Exception as e:
            logger.error(f"Failed to switch mode: {e}")
            return {"success": False, "message": str(e)}
    
    def _execute_move(self, direction: str, speed: Optional[float] = None) -> Dict[str, Any]:
        endpoint = f"/command/move/{direction}"
        params = {"speed": speed} if speed is not None else {}
        try:
            response = requests.post(f"{self.api_url}{endpoint}", params=params, timeout=self.timeout)
            response.raise_for_status()
            logger.info(f"Moving {direction}" + (f" at speed {speed}" if speed else ""))
            return {"success": True, "message": response.json().get("message", f"Moving {direction}")}
        except requests.Timeout:
            logger.error(f"Request timeout for move {direction}")
            return {"success": False, "message": "Request timed out"}
        except Exception as e:
            logger.error(f"Failed to move {direction}: {e}")
            return {"success": False, "message": str(e)}
    
    def _execute_turn(self, direction: str, speed: Optional[float] = None) -> Dict[str, Any]:
        endpoint = f"/command/turn/{direction}"
        params = {"speed": speed} if speed is not None else {}
        try:
            response = requests.post(f"{self.api_url}{endpoint}", params=params, timeout=self.timeout)
            response.raise_for_status()
            logger.info(f"Turning {direction}" + (f" at speed {speed}" if speed else ""))
            return {"success": True, "message": response.json().get("message", f"Turning {direction}")}
        except requests.Timeout:
            logger.error(f"Request timeout for turn {direction}")
            return {"success": False, "message": "Request timed out"}
        except Exception as e:
            logger.error(f"Failed to turn {direction}: {e}")
            return {"success": False, "message": str(e)}
    
    def _execute_head(self, params: Dict[str, float]) -> Dict[str, Any]:
        endpoint = "/command/head"
        # Fill in defaults for unspecified parameters
        head_command = {
            "neck_pitch": params.get("neck_pitch", 0.0),
            "head_pitch": params.get("head_pitch", 0.0),
            "head_yaw": params.get("head_yaw", 0.0),
            "head_roll": params.get("head_roll", 0.0),
        }
        try:
            response = requests.post(f"{self.api_url}{endpoint}", json=head_command, timeout=self.timeout)
            response.raise_for_status()
            logger.info(f"Head command executed: {params}")
            return {"success": True, "message": "Head position updated"}
        except requests.Timeout:
            logger.error(f"Request timeout for head command")
            return {"success": False, "message": "Request timed out"}
        except Exception as e:
            logger.error(f"Failed to execute head command: {e}")
            return {"success": False, "message": str(e)}

    def _execute_auto_stop(self, params: Dict[str, Any]) -> Dict[str, Any]:
        action = params.get('action')
        if action == 'enable':
            self.auto_stop_enabled = True
            return {"success": True, "message": "Auto-stop enabled"}
        elif action == 'disable':
            self.auto_stop_enabled = False
            self._cancel_auto_stop()
            return {"success": True, "message": "Auto-stop disabled"}
        elif action == 'timeout':
            timeout = params.get('timeout')
            if timeout is not None:
                self.auto_stop_delay = timeout
                return {"success": True, "message": f"Auto-stop timeout set to {timeout} seconds"}
        return {"success": False, "message": "Unknown auto-stop action"}