import random


def roll_dice(*dice):
    if len(dice) == 1 and isinstance(dice[0], list):
        dice = dice[0]
    score = 0
    for die in dice:
        d = die.index("d")
        num_dice = int(die[:d])
        num_sides = int(die[d+1:])
        score += sum(random.randint(1, num_sides) for _ in range(num_dice))
    return score


def example():
    """Example in ipython:

    >>> d = example()
    >>> roll_dice(d['rumia_s<TAB>'])
    """
    return {"youmu_w_dex": ["1d20", "1d16"],
            "youmu_w_int": ["1d20", "1d10"],
            "youmu_w_str": ["1d20", "1d10"],
            "rumia_w_dex": ["1d20", "1d10"],
            "rumia_w_str": ["1d20", "1d8"],
            "rumia_spell": ["1d20", "2d4"],
            "gen": ["1d20", "1d13"]
            }
