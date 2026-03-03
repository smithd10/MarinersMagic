"""Opponent persona system with Dirichlet-multinomial refinement."""

import numpy as np
from engine.game_data import FEATURE_NAMES

NUM_FEATURES = len(FEATURE_NAMES)

PERSONAS = {
    "Fan": {
        "sigma": 0.15,
        "sigma_std": 0.15,
        "description": "Picks games to attend. Weekend warriors. Blind to resale.",
        "icon": "Fn",
        "weights": {
            "weekend": 2.5, "premium_opp": 2.0, "mid_opp": 1.0, "promo": 1.5,
            "summer": 1.3, "september": 1.0, "resale_high": 1.0, "resale_mid": 1.0,
            "jewish_holiday": 1.0,
        },
        "temperature_mult": 1.0,
    },
    "Casual": {
        "sigma": 0.45,
        "sigma_std": 0.15,
        "description": "Picks intuitively good games. Vaguely aware some are valuable.",
        "icon": "Cs",
        "weights": {
            "weekend": 2.0, "premium_opp": 1.5, "mid_opp": 1.0, "promo": 1.2,
            "summer": 1.2, "september": 1.0, "resale_high": 1.2, "resale_mid": 1.0,
            "jewish_holiday": 1.0,
        },
        "temperature_mult": 1.0,
    },
    "Calculator": {
        "sigma": 0.85,
        "sigma_std": 0.15,
        "description": "Actively optimizing. Checks SeatGeek. Targets profit.",
        "icon": "Ca",
        "weights": {
            "weekend": 1.3, "premium_opp": 1.3, "mid_opp": 1.0, "promo": 1.0,
            "summer": 1.0, "september": 1.0, "resale_high": 2.5, "resale_mid": 1.5,
            "jewish_holiday": 1.0,
        },
        "temperature_mult": 1.0,
    },
    "Wildcard": {
        "sigma": 0.50,
        "sigma_std": 0.20,
        "description": "Unpredictable. Random-seeming picks.",
        "icon": "Wc",
        "weights": {
            "weekend": 1.0, "premium_opp": 1.0, "mid_opp": 1.0, "promo": 1.0,
            "summer": 1.0, "september": 1.0, "resale_high": 1.0, "resale_mid": 1.0,
            "jewish_holiday": 1.0,
        },
        "temperature_mult": 1.5,
    },
}

PERSONA_NAMES = list(PERSONAS.keys())


def get_persona_weights_array(persona_name: str) -> np.ndarray:
    """Get the feature weight vector for a persona as numpy array."""
    w = PERSONAS[persona_name]["weights"]
    return np.array([w[f] for f in FEATURE_NAMES], dtype=np.float32)


def init_dirichlet_alphas(persona_name: str) -> np.ndarray:
    """Initialize Dirichlet concentration params from persona prior."""
    return get_persona_weights_array(persona_name).copy()


def update_dirichlet(alphas: np.ndarray, game_features: np.ndarray,
                     feature_prevalence: np.ndarray) -> np.ndarray:
    """Update Dirichlet params after observing a pick.

    alphas: (n_features,) current concentration params
    game_features: (n_features,) binary features of the picked game
    feature_prevalence: (n_features,) fraction of remaining games with each feature
    """
    safe_prevalence = np.maximum(feature_prevalence, 0.05)
    update = game_features / safe_prevalence
    return alphas + update


def compute_feature_prevalence(feature_matrix: np.ndarray, available_mask: np.ndarray) -> np.ndarray:
    """Compute fraction of available games that have each feature."""
    available_features = feature_matrix[available_mask]
    if len(available_features) == 0:
        return np.ones(feature_matrix.shape[1], dtype=np.float32) * 0.5
    return available_features.mean(axis=0)


class OpponentModel:
    """Tracks an opponent's persona and Dirichlet-refined preferences."""

    def __init__(self, family_id: int, name: str, persona_name: str = "Casual",
                 is_observant: bool = False):
        self.family_id = family_id
        self.name = name
        self.persona_name = persona_name
        self.is_observant = is_observant
        self.alphas = init_dirichlet_alphas(persona_name)
        self.picks: list[int] = []
        self.dmu_counts = {}

    @property
    def sigma(self) -> float:
        return PERSONAS[self.persona_name]["sigma"]

    @property
    def sigma_std(self) -> float:
        return PERSONAS[self.persona_name]["sigma_std"]

    @property
    def temperature_mult(self) -> float:
        return PERSONAS[self.persona_name]["temperature_mult"]

    def set_persona(self, persona_name: str):
        self.persona_name = persona_name
        self.alphas = init_dirichlet_alphas(persona_name)
        for pick_features, prevalence in getattr(self, '_pick_history', []):
            self.alphas = update_dirichlet(self.alphas, pick_features, prevalence)

    def record_pick(self, game_id: int, game_features: np.ndarray,
                    feature_prevalence: np.ndarray, game_categories: dict):
        self.picks.append(game_id)
        if not hasattr(self, '_pick_history'):
            self._pick_history = []
        self._pick_history.append((game_features.copy(), feature_prevalence.copy()))
        self.alphas = update_dirichlet(self.alphas, game_features, feature_prevalence)
        for cat, val in game_categories.items():
            key = f"{cat}:{val}"
            self.dmu_counts[key] = self.dmu_counts.get(key, 0) + 1

    def get_dmu_decay(self, game_categories: dict, base: float = 0.85) -> float:
        total = 0
        for cat, val in game_categories.items():
            key = f"{cat}:{val}"
            total += self.dmu_counts.get(key, 0)
        return base ** total

    def compute_v_att(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Compute attendance value for all games. Returns (n_games,) array in 0-1."""
        raw = feature_matrix @ self.alphas
        alpha_sum = self.alphas.sum()
        if alpha_sum > 0:
            raw = raw / alpha_sum
        rmax = raw.max()
        if rmax > 0:
            raw = raw / rmax
        return raw

    def to_dict(self) -> dict:
        return {
            "family_id": self.family_id,
            "name": self.name,
            "persona_name": self.persona_name,
            "is_observant": self.is_observant,
            "alphas": self.alphas.tolist(),
            "picks": self.picks,
            "dmu_counts": self.dmu_counts,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OpponentModel":
        m = cls(d["family_id"], d["name"], d["persona_name"], d.get("is_observant", False))
        m.alphas = np.array(d["alphas"], dtype=np.float32)
        m.picks = d["picks"]
        m.dmu_counts = d["dmu_counts"]
        return m
