"""Interpretable high-level overtake state machine.

Why this exists
---------------
``SensorFusion`` answers "what is around me right now".  The RARL policy used to
turn that answer into a lane change on its own: ``lane_aware`` picked a target
lane and ``overtake_lock`` pinned the steering action until the lane index
changed.  That is enough to *cross* a lane line, but not enough to describe an
**overtake**: nothing remembered the car that forced the manoeuvre, nothing
waited for a safe gap behind, and nothing required the blocker to actually end
up behind the ego.

:class:`OvertakeController` is that missing layer.  It consumes only a
:class:`autodrive.sensors.SensorFrame` and exposes a four-phase state machine::

    FOLLOW ──opportunity──▶ PREPARE_LEFT / PREPARE_RIGHT ──safe hold──▶ COMPLETE
      ▲                         │                                   │
      │                         └──────── unsafe window ──▶ ABORT ◀─┘
      └────────────────────────── cooldown ──────────────────────────┘

Every transition is triggered by a sensor fact, never by an action:

* **FOLLOW -> PREPARE**: the ego lane's lead is closer than
  ``opportunity_gap`` *and* the ego is closing on it by at least
  ``min_closing`` m/s; an adjacent lane must pass the entry gate (front/rear
  gap, ``safe_window`` and front/rear TTC).
* **PREPARE stays prepared** while the target lane's window/TTC stay above the
  (lower) abort thresholds.
* **PREPARE -> ABORT**: the window/TTC fall below those thresholds, the lead
  clears on its own, the ego ends up in a third lane, or the manoeuvre takes
  longer than ``prepare_timeout_steps``.
* **PREPARE -> COMPLETE**: the ego has entered the target lane, stayed there
  for ``complete_hold_steps`` with a safe window, and the original blocker is
  behind the ego by more than ``pass_margin`` metres.
* **ABORT / COMPLETE -> FOLLOW** after their hold counters expire.

The module is pure and read-only, so the same controller is used by the
observation block, the action lock, the reward gate and the tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .sensors import SensorFrame


class OvertakePhase(str, Enum):
    """The phases of the high-level overtake manoeuvre."""

    FOLLOW = "follow"
    PREPARE_LEFT = "prepare_left"
    PREPARE_RIGHT = "prepare_right"
    ABORT = "abort"
    COMPLETE = "complete"

    @property
    def is_prepare(self) -> bool:
        return self in (OvertakePhase.PREPARE_LEFT, OvertakePhase.PREPARE_RIGHT)

    @property
    def direction(self) -> int:
        if self is OvertakePhase.PREPARE_LEFT:
            return 1
        if self is OvertakePhase.PREPARE_RIGHT:
            return -1
        return 0


#: Alias kept because the design notes call the enum "OvertakeState".
OvertakeState = OvertakePhase


DEFAULT_CONFIG = {    # --- opportunity (ego lane) --------------------------------------------
    "opportunity_gap": 18.0,       # [m] closer lead == blocked
    "min_closing": 0.5,            # [m/s] ego must be faster by this to want a pass
    "release_gap": 30.0,           # [m] the lead is no longer worth passing
    # --- target lane entry gate (what turns an opportunity into PREPARE) ----
    "entry_min_front": 12.0,       # [m] centre-to-centre room ahead in the target lane
    "entry_min_rear": 8.0,         # [m] centre-to-centre room behind
    "entry_min_window": 2.0,       # [s] target lane must stay safe this long
    "entry_min_front_ttc": 1.5,    # [s] time to contact with the target-lane lead
    "entry_min_rear_ttc": 2.0,     # [s] time to contact with the target-lane follower
    # --- abort gate (lower than entry: a committed manoeuvre may be tighter) -
    "abort_min_window": 1.0,       # [s]
    "abort_min_front_ttc": 0.8,    # [s]
    "abort_min_rear_ttc": 1.0,     # [s]
    # --- completion gate ---------------------------------------------------
    "complete_min_window": 1.5,    # [s] the target lane must still be safe
    "complete_min_front_ttc": 1.0,  # [s]
    "complete_min_rear_ttc": 1.0,  # [s]
    "complete_hold_steps": 3,      # control steps in the target lane before COMPLETE
    "pass_margin": 3.0,            # [m] blocker must be this far behind the ego
    # --- timers / preference ----------------------------------------------
    "prepare_timeout_steps": 60,
    "abort_hold_steps": 20,
    "complete_hold_after": 3,      # control steps reported as COMPLETE
    "preferred_side": "auto",      # auto | left | right
}


@dataclass(frozen=True)
class OvertakeCommand:
    """One controller decision for the current sensor frame."""

    phase: OvertakePhase = OvertakePhase.FOLLOW
    target_lane: int | None = None
    direction: int = 0
    available: bool = False       # a target lane is viable right now
    lock: bool = False            # the tactical lock may force the steer action
    blocker_id: int | None = None
    blocker_passed: bool = False
    target_window: float = 0.0    # [s] safe window of the chosen target lane
    safe: bool = True             # the target lane is inside its abort gate
    reason: str = "no_opportunity"
    event: str | None = None      # "prepare" | "abort" | "complete" | None

    @property
    def state(self) -> str:
        return self.phase.value

    @property
    def active(self) -> bool:
        return self.phase.is_prepare


class OvertakeController:
    """Sensor-driven, side-effecting state machine for one ego vehicle."""

    def __init__(self, cfg=None):
        c = dict(DEFAULT_CONFIG)
        for k, v in (cfg or {}).items():
            if k not in c:
                raise ValueError(f"unknown overtake config key '{k}'")
            c[k] = v
        side = str(c["preferred_side"]).lower()
        if side not in ("auto", "left", "right"):
            raise ValueError(f"unknown preferred_side '{c['preferred_side']}' "
                             "(auto | left | right)")
        c["preferred_side"] = side
        self.cfg = c
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        """Return to FOLLOW and forget the manoeuvre (new episode)."""
        self.phase = OvertakePhase.FOLLOW
        self.target_lane: int | None = None
        self.start_lane: int | None = None
        self.blocker_id: int | None = None
        self.aborts = 0
        self.completions = 0
        self.last_command: OvertakeCommand | None = None
        self._last_frame: SensorFrame | None = None
        self._hold = 0
        self._prepare_steps = 0
        self._abort_left = 0
        self._complete_left = 0

    # ------------------------------------------------------------------- step
    def step(self, frame: SensorFrame) -> OvertakeCommand:
        """Advance the machine with one sensor frame and return the decision."""
        ego_lane = int(frame.ego_lane)
        event: str | None = None
        reason = ""

        # timers first: ABORT / COMPLETE automatically fall back to FOLLOW
        if self.phase is OvertakePhase.ABORT:
            reason = "abort_hold"
            self._abort_left -= 1
            if self._abort_left <= 0:
                self.phase = OvertakePhase.FOLLOW
                reason = "abort_released"
        elif self.phase is OvertakePhase.COMPLETE:
            reason = "complete_hold"
            self._complete_left -= 1
            if self._complete_left <= 0:
                self.phase = OvertakePhase.FOLLOW
                reason = "complete_released"

        if self.phase is OvertakePhase.FOLLOW:
            candidate = self._pick_target(frame)
            if candidate is None:
                reason = "blocked_no_safe_target" if self._opportunity(frame) else "no_opportunity"
            else:
                lane, direction = candidate
                self._enter_prepare(ego_lane, int(lane), int(direction), frame)
                event = "prepare"
                reason = "opportunity"
        elif self.phase.is_prepare:
            event, reason = self._advance_prepare(frame, ego_lane)

        command = self._command(frame, reason=reason, event=event)
        self.last_command = command
        self._last_frame = frame
        return command

    def abort(self, reason: str = "external") -> OvertakeCommand:
        """Abort a prepared manoeuvre (e.g. the safety shield vetoed the steer)."""
        if self.phase.is_prepare:
            self._enter_abort()
            command = self._command(self._last_frame or self._frame_stub(),
                                    reason=reason, event="abort")
            self.last_command = command
            return command
        return self.last_command or OvertakeCommand(reason=reason)

    # ------------------------------------------------------------ transitions
    def _enter_prepare(self, ego_lane: int, lane: int, direction: int,
                       frame: SensorFrame) -> None:
        self.phase = (OvertakePhase.PREPARE_LEFT if direction > 0
                      else OvertakePhase.PREPARE_RIGHT)
        self.target_lane = int(lane)
        self.start_lane = int(ego_lane)
        idx = int(frame.front_ids[ego_lane]) if int(frame.front_ids[ego_lane]) >= 0 else None
        self.blocker_id = idx
        self._hold = 0
        self._prepare_steps = 0

    def _advance_prepare(self, frame: SensorFrame, ego_lane: int):
        """Return ``(event, reason)`` and update the phase for one sensor frame."""
        self._prepare_steps += 1
        target = frame.reading(int(self.target_lane)) if self.target_lane is not None else None
        if target is None or not target.exists:
            self._enter_abort()
            return "abort", "target_lane_missing"

        if ego_lane == self.target_lane:
            # inside the target lane: only the window and the actual pass matter
            if not self._window_ok(target, abort=True):
                self._enter_abort()
                return "abort", "window"
            self._hold += 1
            if self._hold >= int(self.cfg["complete_hold_steps"]) and self._blocker_passed(frame):
                self.phase = OvertakePhase.COMPLETE
                self._complete_left = int(self.cfg["complete_hold_after"])
                self.completions += 1
                return "complete", "blocker_passed"
            if self._prepare_steps > int(self.cfg["prepare_timeout_steps"]):
                self._enter_abort()
                return "abort", "timeout"
            return None, "holding"

        if not self._window_ok(target, abort=True):
            self._enter_abort()
            return "abort", "window"
        if self._lead_cleared(frame):
            self._enter_abort()
            return "abort", "cleared"
        if self._prepare_steps > int(self.cfg["prepare_timeout_steps"]):
            self._enter_abort()
            return "abort", "timeout"
        return None, "committed"

    def _enter_abort(self) -> None:
        self.phase = OvertakePhase.ABORT
        self.target_lane = None
        self._abort_left = int(self.cfg["abort_hold_steps"])
        self.aborts += 1

    # ------------------------------------------------------------- sensor read
    def _opportunity(self, frame: SensorFrame) -> bool:
        ego = frame.reading(frame.ego_lane)
        if not ego.exists:
            return False
        return bool(ego.front_gap <= float(self.cfg["opportunity_gap"])
                    and ego.front_rel_speed >= float(self.cfg["min_closing"]))

    def _lead_cleared(self, frame: SensorFrame) -> bool:
        """The car that blocked the *start* lane is no longer worth passing.

        The ego may have drifted into another lane while the lock was correcting;
        in that case the original blocker is still there and the manoeuvre is
        still valid, so the check reads the start lane, not the momentary ego
        lane.
        """
        lane = self.start_lane if self.start_lane is not None else frame.ego_lane
        return bool(frame.reading(int(lane)).front_gap > float(self.cfg["release_gap"]))

    def _pick_target(self, frame: SensorFrame) -> tuple[int, int] | None:
        """The best adjacent lane that passes the entry gate, or ``None``."""
        if not self._opportunity(frame):
            return None
        ego_lane = int(frame.ego_lane)
        choices = []
        for lane, direction in ((ego_lane + 1, 1), (ego_lane - 1, -1)):
            reading = frame.reading(lane)
            if not reading.exists or not reading.safe_to_enter:
                continue
            if not self._entry_ok(reading):
                continue
            choices.append((lane, direction, reading))
        if not choices:
            return None

        side = str(self.cfg["preferred_side"])
        if side != "auto":
            wanted = 1 if side == "left" else -1
            preferred = [c for c in choices if c[1] == wanted]
            if preferred:
                choices = preferred
        lane, direction, _ = max(
            choices, key=lambda c: (c[2].safe_window, c[2].front_gap, c[1]))
        return int(lane), int(direction)

    def _entry_ok(self, r) -> bool:
        c = self.cfg
        return bool(
            r.front_gap >= float(c["entry_min_front"])
            and r.rear_gap >= float(c["entry_min_rear"])
            and r.safe_window >= float(c["entry_min_window"])
            and r.front_ttc >= float(c["entry_min_front_ttc"])
            and r.rear_ttc >= float(c["entry_min_rear_ttc"])
        )

    def _window_ok(self, r, abort: bool) -> bool:
        c = self.cfg
        if abort:
            window = float(c["abort_min_window"])
            front_ttc = float(c["abort_min_front_ttc"])
            rear_ttc = float(c["abort_min_rear_ttc"])
        else:
            window = float(c["complete_min_window"])
            front_ttc = float(c["complete_min_front_ttc"])
            rear_ttc = float(c["complete_min_rear_ttc"])
        return bool(r.safe_window >= window
                    and r.front_ttc >= front_ttc
                    and r.rear_ttc >= rear_ttc)

    def _blocker_passed(self, frame: SensorFrame) -> bool:
        if self.blocker_id is None:
            return False
        offset = frame.npc_offsets.get(int(self.blocker_id))
        if offset is None:
            return True                      # the car is gone from the simulation
        return bool(float(offset) < -float(self.cfg["pass_margin"]))

    # ---------------------------------------------------------------- command
    def _command(self, frame: SensorFrame, reason: str,
                 event: str | None) -> OvertakeCommand:
        target = frame.reading(int(self.target_lane)) if self.target_lane is not None else None
        window = float(target.safe_window) if target is not None and target.exists else 0.0
        safe = bool(target is not None and target.exists
                    and self._window_ok(target, abort=True))
        lock = bool(self.phase.is_prepare and target is not None and target.exists
                    and safe)
        return OvertakeCommand(
            phase=self.phase,
            target_lane=self.target_lane,
            direction=self.phase.direction if self.phase.is_prepare else 0,
            available=self.phase.is_prepare,
            lock=lock,
            blocker_id=self.blocker_id,
            blocker_passed=self._blocker_passed(frame),
            target_window=window,
            safe=safe,
            reason=reason,
            event=event,
        )

    def _frame_stub(self) -> SensorFrame:
        """An empty frame for commands raised without a sensor reading."""
        return SensorFrame(ego_lane=-1, ego_speed=0.0, rows=(),
                           front_gaps=(), rear_gaps=(), side_gaps=(), front_ids=())
