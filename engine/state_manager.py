"""State persistence for cloud-compatible draft state management."""

import json
import base64
import numpy as np
from engine.personas import OpponentModel


class DraftState:
    """Complete draft state that can be serialized/restored."""

    def __init__(self):
        self.my_slot: int = 1
        self.num_families: int = 9
        self.current_pick_idx: int = 0
        self.my_picks: list[int] = []
        self.taken_games: dict[int, int] = {}  # game_id -> family_slot
        self.opponent_models: dict[int, OpponentModel] = {}
        self.w_att: float = 0.70
        self.w_profit: float = 0.30
        self.hassle_discount_on: bool = True
        self.must_have_ids: list[int] = []
        self.preference_overrides: dict[int, int] = {}
        self.availability_overrides: dict[int, bool] = {}
        self.undo_stack: list[dict] = []
        self.mode: str = "predraft"

    @property
    def hassle_discount(self) -> float:
        return 0.85 if self.hassle_discount_on else 1.0

    def init_opponents(self, family_names: list = None):
        if family_names is None:
            family_names = [f"Family {i}" for i in range(1, self.num_families + 1)]
        for i in range(1, self.num_families + 1):
            if i == self.my_slot:
                continue
            name = family_names[i - 1] if i - 1 < len(family_names) else f"Family {i}"
            if i not in self.opponent_models:
                self.opponent_models[i] = OpponentModel(i, name)

    def record_opponent_pick(self, game_id: int, family_slot: int,
                             game_features, feature_prevalence, game_categories):
        snapshot = self._snapshot()
        self.undo_stack.append(snapshot)
        if len(self.undo_stack) > 5:
            self.undo_stack.pop(0)

        self.taken_games[game_id] = family_slot
        self.current_pick_idx += 1

        if family_slot in self.opponent_models:
            self.opponent_models[family_slot].record_pick(
                game_id, game_features, feature_prevalence, game_categories
            )

    def record_my_pick(self, game_id: int):
        snapshot = self._snapshot()
        self.undo_stack.append(snapshot)
        if len(self.undo_stack) > 5:
            self.undo_stack.pop(0)

        self.my_picks.append(game_id)
        self.taken_games[game_id] = self.my_slot
        self.current_pick_idx += 1

    def undo(self) -> bool:
        if not self.undo_stack:
            return False
        snapshot = self.undo_stack.pop()
        self._restore_snapshot(snapshot)
        return True

    def get_all_taken(self) -> set[int]:
        return set(self.taken_games.keys())

    def get_current_picker(self, snake_order) -> int:
        if self.current_pick_idx < len(snake_order):
            return snake_order[self.current_pick_idx][0]
        return -1

    def get_current_round(self, snake_order) -> int:
        if self.current_pick_idx < len(snake_order):
            return snake_order[self.current_pick_idx][1]
        return -1

    def picks_until_my_turn(self, snake_order) -> int:
        count = 0
        for i in range(self.current_pick_idx, len(snake_order)):
            slot, _ = snake_order[i]
            if slot == self.my_slot:
                return count
            count += 1
        return count

    def _snapshot(self) -> dict:
        return {
            "current_pick_idx": self.current_pick_idx,
            "my_picks": list(self.my_picks),
            "taken_games": dict(self.taken_games),
            "opponent_models": {
                k: v.to_dict() for k, v in self.opponent_models.items()
            },
        }

    def _restore_snapshot(self, snap: dict):
        self.current_pick_idx = snap["current_pick_idx"]
        self.my_picks = snap["my_picks"]
        self.taken_games = snap["taken_games"]
        self.opponent_models = {
            int(k): OpponentModel.from_dict(v)
            for k, v in snap["opponent_models"].items()
        }

    def to_dict(self) -> dict:
        return {
            "my_slot": self.my_slot,
            "num_families": self.num_families,
            "current_pick_idx": self.current_pick_idx,
            "my_picks": self.my_picks,
            "taken_games": {str(k): v for k, v in self.taken_games.items()},
            "opponent_models": {
                str(k): v.to_dict() for k, v in self.opponent_models.items()
            },
            "w_att": self.w_att,
            "w_profit": self.w_profit,
            "hassle_discount_on": self.hassle_discount_on,
            "must_have_ids": self.must_have_ids,
            "preference_overrides": {str(k): v for k, v in self.preference_overrides.items()},
            "availability_overrides": {str(k): v for k, v in self.availability_overrides.items()},
            "mode": self.mode,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DraftState":
        s = cls()
        s.my_slot = d["my_slot"]
        s.num_families = d.get("num_families", 9)
        s.current_pick_idx = d["current_pick_idx"]
        s.my_picks = d["my_picks"]
        s.taken_games = {int(k): v for k, v in d["taken_games"].items()}
        s.opponent_models = {
            int(k): OpponentModel.from_dict(v)
            for k, v in d["opponent_models"].items()
        }
        s.w_att = d.get("w_att", 0.70)
        s.w_profit = d.get("w_profit", 0.30)
        s.hassle_discount_on = d.get("hassle_discount_on", True)
        s.must_have_ids = d.get("must_have_ids", [])
        s.preference_overrides = {int(k): v for k, v in d.get("preference_overrides", {}).items()}
        s.availability_overrides = {int(k): v for k, v in d.get("availability_overrides", {}).items()}
        s.mode = d.get("mode", "predraft")
        return s

    def to_restore_code(self) -> str:
        data = json.dumps(self.to_dict(), separators=(",", ":"))
        return base64.b64encode(data.encode()).decode()

    @classmethod
    def from_restore_code(cls, code: str) -> "DraftState":
        data = json.loads(base64.b64decode(code.encode()).decode())
        return cls.from_dict(data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "DraftState":
        return cls.from_dict(json.loads(json_str))
