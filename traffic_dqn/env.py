"""Single 4-way intersection environment (N, S, E, W lanes)."""

import random

import numpy as np


class TrafficIntersection:
    """
    Single 4-way intersection environment.

    State (8 floats, all normalised to [0, 1]):
        [queue_N, queue_S, queue_E, queue_W,
         current_phase, time_in_phase,
         arrival_rate_NS, arrival_rate_EW]

    Actions (int 0-3):
        0 → NS green   (North-South go, East-West stop)
        1 → NS yellow  (transition)
        2 → EW green   (East-West go, North-South stop)
        3 → EW yellow  (transition)

    Reward:
        +2 per vehicle that departs
        −1 × total queue length    (penalise waiting)
        −1 during yellow phases    (penalise unnecessary transitions)
    """

    MAX_QUEUE      = 12   # vehicles per lane
    MAX_PHASE_TIME = 20   # cap used to normalise time_in_phase in the state
    MIN_GREEN_TIME = 5    # minimum green steps before allowed to switch

    # Arrival probabilities per traffic mode
    TRAFFIC_MODES = {
        "uniform":    {"N": 0.30, "S": 0.30, "E": 0.30, "W": 0.30},
        "rush_hour":  {"N": 0.60, "S": 0.60, "E": 0.15, "W": 0.15},
        "heavy":      {"N": 0.55, "S": 0.55, "E": 0.55, "W": 0.55},
        "imbalanced": {"N": 0.20, "S": 0.20, "E": 0.65, "W": 0.45},
    }

    def __init__(self, traffic_mode: str = "uniform"):
        self.arrival_rates = self.TRAFFIC_MODES[traffic_mode]
        self.reset()

    def reset(self) -> np.ndarray:
        self.queues       = {d: random.randint(0, 2) for d in "NSEW"}
        self.phase        = 0   # start on NS green
        self.time_in_phase = 0
        self.total_steps  = 0
        return self._get_state()

    # ── Core methods ──────────────────────────────────────────

    def step(self, action: int):
        """
        Apply action, advance one time step.
        Returns (next_state, reward, done, info).
        """
        # Enforce minimum green time — ignore switches too early
        if self.time_in_phase < self.MIN_GREEN_TIME:
            action = self.phase

        # Handle yellow transitions: can only switch to yellow,
        # then yellow automatically advances to the next green.
        self._apply_phase_transition(action)

        # Stochastic arrivals
        for d in "NSEW":
            if random.random() < self.arrival_rates[d]:
                self.queues[d] = min(self.queues[d] + 1, self.MAX_QUEUE)

        # Departures (2 vehicles/step when green)
        departed = 0
        if self.phase == 0:   # NS green
            for d in ["N", "S"]:
                n = min(self.queues[d], 2)
                self.queues[d] -= n
                departed += n
        elif self.phase == 2: # EW green
            for d in ["E", "W"]:
                n = min(self.queues[d], 2)
                self.queues[d] -= n
                departed += n

        # Reward
        total_queue = sum(self.queues.values())
        reward = (departed * 2.0
              - total_queue * 1.0
              - (1.0 if self.phase in (1, 3) else 0.0))

        self.time_in_phase += 1
        self.total_steps   += 1

        info = {
            "departed":    departed,
            "total_queue": total_queue,
            "phase":       self.phase,
            "queues":      dict(self.queues),
        }
        return self._get_state(), reward, False, info

    # ── Internal helpers ──────────────────────────────────────

    def _apply_phase_transition(self, action: int):
        """
        State machine:
            0 (NS green)  → 1 (NS yellow) → 2 (EW green)
            2 (EW green)  → 3 (EW yellow) → 0 (NS green)
        Yellow phases auto-advance on the next step.
        """
        if action != self.phase:
            if   self.phase == 0:              # NS green  → NS yellow
                self.phase = 1; self.time_in_phase = 0
            elif self.phase == 1:              # NS yellow → EW green
                self.phase = 2; self.time_in_phase = 0
            elif self.phase == 2:              # EW green  → EW yellow
                self.phase = 3; self.time_in_phase = 0
            elif self.phase == 3:              # EW yellow → NS green
                self.phase = 0; self.time_in_phase = 0
        # Force yellow to auto-advance (yellow = 1 step)
        elif self.phase in (1, 3):
            self.phase = 2 if self.phase == 1 else 0
            self.time_in_phase = 0

    def _get_state(self) -> np.ndarray:
        """Normalised 8-dimensional state vector."""
        return np.array([
            self.queues["N"] / self.MAX_QUEUE,
            self.queues["S"] / self.MAX_QUEUE,
            self.queues["E"] / self.MAX_QUEUE,
            self.queues["W"] / self.MAX_QUEUE,
            self.phase / 3.0,
            min(self.time_in_phase, self.MAX_PHASE_TIME) / self.MAX_PHASE_TIME,
            self.arrival_rates["N"],   # static feature
            self.arrival_rates["E"],   # static feature
        ], dtype=np.float32)

    @property
    def state_size(self) -> int:
        return 8

    @property
    def action_size(self) -> int:
        return 4
