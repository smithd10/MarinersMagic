"""Snake draft order generation for 9-family season ticket draft."""

import numpy as np


def generate_snake_order(num_families: int = 9, num_games: int = 80) -> list[tuple[int, int]]:
    """Generate the full snake draft pick order.

    Returns list of (slot_number, round_number) tuples.
    Slot numbers are 1-indexed. Round 1: 1->9, Round 2: 9->1, etc.
    """
    order = []
    round_num = 0
    while len(order) < num_games:
        round_num += 1
        if round_num % 2 == 1:
            slots = range(1, num_families + 1)
        else:
            slots = range(num_families, 0, -1)
        for slot in slots:
            if len(order) < num_games:
                order.append((slot, round_num))
    return order


def get_my_pick_indices(snake_order: list[tuple[int, int]], my_slot: int) -> list[int]:
    """Get the indices in snake_order where it's my turn to pick."""
    return [i for i, (slot, _) in enumerate(snake_order) if slot == my_slot]


def is_pair_pick(snake_order: list[tuple[int, int]], pick_index: int, my_slot: int) -> bool:
    """Check if this pick is the first of a consecutive pair at a snake turn."""
    if pick_index >= len(snake_order) - 1:
        return False
    current_slot, _ = snake_order[pick_index]
    next_slot, _ = snake_order[pick_index + 1]
    return current_slot == my_slot and next_slot == my_slot


def get_picks_until_next_turn(snake_order: list[tuple[int, int]], current_index: int, my_slot: int) -> list[tuple[int, int, int]]:
    """Get list of (pick_index, slot, round) for opponents picking between now and my next turn."""
    opponents = []
    for i in range(current_index + 1, len(snake_order)):
        slot, rnd = snake_order[i]
        if slot == my_slot:
            break
        opponents.append((i, slot, rnd))
    return opponents
