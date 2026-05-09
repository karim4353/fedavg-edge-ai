"""
mqtt_simulation.py - Lightweight MQTT communication simulation.

Simulates the MQTT-based communication protocol used in the team's
presentation for exchanging model weights between IoT nodes and the
central server.

This is a mock layer — no actual MQTT broker is needed. It simulates:
  - Message serialization/deserialization of model parameters
  - Communication overhead estimation (bytes transferred)
  - Round-trip time simulation for different network conditions

This matches the presentation's architecture:
  '3 nœuds IoT communiquant via protocole MQTT'
"""

import numpy as np
import time
import json


class MQTTMessage:
    """Simulated MQTT message for model weight exchange."""

    def __init__(self, topic, payload, client_id, timestamp=None):
        self.topic = topic
        self.payload = payload
        self.client_id = client_id
        self.timestamp = timestamp or time.time()
        self.size_bytes = len(payload) if isinstance(payload, bytes) else 0


class MQTTSimulator:
    """Mock MQTT broker/client for federated learning communication.

    Simulates the message exchange pattern:
      1. Server publishes global model to 'federated/model/global'
      2. Each client subscribes, trains locally, then publishes updates
         to 'federated/model/update/{client_id}'
      3. Server collects updates and aggregates

    No actual network is used — everything is in-memory.
    """

    def __init__(self, num_clients=3, network_delay_ms=10.0):
        """Initialize the MQTT simulator.

        Args:
            num_clients: number of IoT nodes
            network_delay_ms: simulated network latency per message
        """
        self.num_clients = num_clients
        self.network_delay_ms = network_delay_ms
        self.message_log = []
        self.total_bytes_transferred = 0
        self.total_messages = 0

    def serialize_params(self, params):
        """Serialize model parameters to bytes (simulated).

        Args:
            params: list of (W, b) tuples

        Returns:
            bytes: serialized payload
        """
        # Concatenate all parameters into a single flat array
        flat = np.concatenate([
            np.concatenate([W.flatten(), b.flatten()])
            for W, b in params
        ])
        return flat.tobytes()

    def deserialize_params(self, payload, param_shapes):
        """Deserialize bytes back to parameters.

        Args:
            payload: bytes from serialize_params
            param_shapes: list of ((W_shape), (b_shape)) tuples

        Returns:
            params: list of (W, b) tuples
        """
        flat = np.frombuffer(payload, dtype=np.float64)
        params = []
        offset = 0
        for w_shape, b_shape in param_shapes:
            w_size = np.prod(w_shape)
            b_size = np.prod(b_shape)
            W = flat[offset:offset + w_size].reshape(w_shape)
            offset += w_size
            b = flat[offset:offset + b_size].reshape(b_shape)
            offset += b_size
            params.append((W.copy(), b.copy()))
        return params

    def publish_global_model(self, params, round_num):
        """Server publishes global model to all clients.

        Args:
            params: global model parameters
            round_num: current communication round

        Returns:
            message: MQTTMessage
        """
        payload = self.serialize_params(params)
        msg = MQTTMessage(
            topic='federated/model/global',
            payload=payload,
            client_id='server',
        )
        msg.size_bytes = len(payload)

        # Simulate broadcast to all clients
        total_bytes = len(payload) * self.num_clients
        self.total_bytes_transferred += total_bytes
        self.total_messages += self.num_clients

        self.message_log.append({
            'round': round_num,
            'direction': 'server→clients',
            'bytes': total_bytes,
            'messages': self.num_clients,
        })

        return msg

    def publish_client_update(self, params, client_id, round_num):
        """Client publishes local model update to server.

        Args:
            params: local model parameters
            client_id: client identifier
            round_num: current communication round

        Returns:
            message: MQTTMessage
        """
        payload = self.serialize_params(params)
        msg = MQTTMessage(
            topic=f'federated/model/update/{client_id}',
            payload=payload,
            client_id=client_id,
        )
        msg.size_bytes = len(payload)

        self.total_bytes_transferred += len(payload)
        self.total_messages += 1

        self.message_log.append({
            'round': round_num,
            'direction': f'client_{client_id}→server',
            'bytes': len(payload),
            'messages': 1,
        })

        return msg

    def get_communication_stats(self):
        """Get summary of communication statistics.

        Returns:
            dict with total bytes, messages, and per-round averages
        """
        num_rounds = len(set(m['round'] for m in self.message_log)) if self.message_log else 1
        return {
            'total_bytes': self.total_bytes_transferred,
            'total_kb': self.total_bytes_transferred / 1024,
            'total_mb': self.total_bytes_transferred / (1024 * 1024),
            'total_messages': self.total_messages,
            'num_rounds': num_rounds,
            'avg_bytes_per_round': self.total_bytes_transferred / num_rounds,
            'simulated_total_latency_ms': self.total_messages * self.network_delay_ms,
        }

    def reset(self):
        """Reset all statistics."""
        self.message_log = []
        self.total_bytes_transferred = 0
        self.total_messages = 0
