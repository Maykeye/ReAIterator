import random
import math
from dataclasses import dataclass
import dataclasses


def roll_dice(*dice):
    if len(dice) == 1 and isinstance(dice[0], list):
        dice = dice[0]
    score = 0
    for die in dice:
        if die == "": die = "0"
        d = die.find("d")
        if d < 0:
            assert ("0"+die[1:]).isdigit() and die[0].isdigit() or die[0] in "+-", f"invalid die-str: {die}"
            score += int(die)
            continue
        num_dice = int(die[:d])
        num_sides = int(die[d+1:])
        if num_sides == 0:
            continue
        delta = sum(random.randint(1, num_sides) for _ in range(abs(num_dice)))
        if num_dice < 0:
            delta = -delta
        score += delta
    return score

def roll_seq(dice):
    assert isinstance(dice, str)
    n_die, n_side = map(int, dice.split('d'))
    return [roll_dice(f'1d{n_side}') for _ in range(n_die)]

def roll_min(dice):
    seq = roll_seq(dice)
    value = sum(seq) - max(seq)
    return value

def roll_cen(dice):
    seq = roll_seq(dice)
    value = sum(seq) - min(seq) - max(seq)
    return value

def roll_max(dice):
    seq = roll_seq(dice)
    value = sum(seq) - min(seq)
    return value

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



class A:
    STR = 0
    INT = 1
    DEX = 2
    CON = 3
    CHA = 4
    WIS = 5
    LUC = 6
    COUNT = 7


@dataclass
class Attributes:
    strength: int
    intellect: int
    dexterity: int
    charisma: int
    constitution: int
    wisdom: int
    luck: int

    def __getitem__(self, idx:int)->int:
        return dataclasses.astuple(self)[idx]

    @property
    def total(self):
        return sum(self[i] for i in range(A.COUNT))

    def simple_factor(self, attr_index: int, div: int):
        # Quartize/Qunitize value
        return max(1, self[attr_index] // div)


def roll_attr(all="3d6", *, strength="", intellect="", dexterity="", constitution="", charisma="", wisdom="", luck=""):
    def make_roll(s):
        if not s:
            return all
        return [all, s] if s[0:1] in "+-" else s

    strength = make_roll(strength)
    intellect = make_roll(intellect)
    dexterity = make_roll(dexterity)
    constitution = make_roll(constitution)
    charisma = make_roll(charisma)
    wisdom = make_roll(wisdom)
    luck = make_roll(luck)

    return Attributes(
            strength=roll_dice(strength),
            intellect=roll_dice(intellect),
            dexterity=roll_dice(dexterity),
            constitution=roll_dice(constitution),
            luck=roll_dice(luck),
            wisdom=roll_dice(wisdom),
            charisma=roll_dice(charisma))
    

def roll_down(n_dice, current_value, max_value=18):
    for dice in range(n_dice, 0, -1):
        sides = math.ceil(max_value / dice)
        if sides == 1:
            continue
        candidate_score = roll_dice(f'{dice}d{sides}')
        if candidate_score <= current_value:
            return candidate_score
    return max(0, roll_dice(f'1d{current_value}'))




def sword_strike(wielder: Attributes, n_rolls:int, bonus: str|list=""):
    return max(0, roll_down(n_rolls, wielder.strength) + roll_dice(bonus))

def melee_strike_save(target: Attributes, bonus: str|list=""):
    return max(0, roll_down(2, target.constitution) + roll_dice(bonus))

# Short Sword can strike for 2rd(STR) + 1d2. 
short_sword_strike = lambda wielder: sword_strike(wielder, 2, "1d2")
long_sword_strike = lambda wielder: sword_strike(wielder, 3, ["1","1d2"])



youmu = Attributes(
        strength=14,
        dexterity=22,
        intellect=11,
        constitution=16,
        charisma=12,
        wisdom=10,
        luck=0)

giant_spider = Attributes(
        strength=19,
        dexterity=14,
        intellect=5,
        constitution=21,
        charisma=2,
        wisdom=3,
        luck=0)


def simple_aim(self: Attributes, other: Attributes):
    """ 5/rdDEX vs 5/rdDEX"""
    aim =  roll_down(self.simple_factor(A.DEX, 5), self[A.DEX])
    dodge =  roll_down(other.simple_factor(A.DEX, 5), other[A.DEX])
    return aim -  dodge


def simple_attack(self: Attributes, other: Attributes, weapon_attack) -> int: # return amount of HP to subtract
    aim = simple_aim(self, other)
    if aim > 0: # DC?
        damage = weapon_attack(self)
        damage_reduction = melee_strike_save(other)
        bonus_damage = 0
        if aim > 10:
            bonus_damage = roll_dice(f"1d{damage}")
        damage = damage - damage_reduction + bonus_damage
        return max(0, damage)
    return 0

def roll_by_dict(d: dict, n=1): # not named roll_dict to prevent name collision from roll_dice during autocomplete
    keys = list(d.keys())
    weights = list(d.values())
    return random.choices(keys, weights=weights, k=n)

NORTHERN_REGION = {
    "Elf": 30,
    "Human": 45,
    "Orcs": 3,
    "Elf-orc": 4,
    "Human-orc": 8,
    "Human-elves": 10
}


