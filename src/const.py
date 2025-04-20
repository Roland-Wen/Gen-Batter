CACHE_DIR = "dataset"

# before preprocessing
BASE_COLS = ['release_pos_x', 'release_pos_z', 'effective_speed', 'spin_axis', 'release_spin_rate', 'pfx_x', 'pfx_z',
             'batter', 'pitcher', 'stand', 'p_throws',
             'balls', 'strikes', 
             'on_3b', 'on_2b', 'on_1b', 'outs_when_up', 'bat_score', 'fld_score',
             'inning', 'inning_topbot', 
             'description', 'launch_speed', 'launch_angle', 
             'game_pk', 'at_bat_number', 'pitch_number']
VALID_DESCRIPTIONS = [
    'called_strike', 'ball', 'swinging_strike',
    'swinging_strike_blocked', 'foul', 'foul_tip', 'blocked_ball',
    'hit_into_play'
]
# Numerical mapping: 0:Take, 1:S&M, 2:Foul, 3:BIP
ACTION_MAP = {
    'ball': 0, 'called_strike': 0, 'blocked_ball': 0,
    'swinging_strike': 1, 'swinging_strike_blocked': 1,
    'foul': 2, 'foul_tip': 2,
    'hit_into_play': 3
}
BIP_ACTION_IDX = 3


# after preprocessing
IDENTIFIER_COLS = ['game_pk', 'at_bat_number', 'pitch_number']
TARGET_ACTION_COL = 'target_action'
TARGET_EV_LA_COL = ['target_ev', 'target_la']
NUMERICAL_FEATS_BASE = [
    'release_pos_x', 'release_pos_z', 'pfx_x', 'pfx_z', 'release_spin_rate', 'spin_axis', 'effective_speed',
    'score_diff'
]
RUNNER_FEATS = ['on_1b', 'on_2b', 'on_3b']
CATEGORICAL_FEATS_BASE = [
    'batter', 'pitcher',
    'stand', 'p_throws',
    'balls', 'strikes', 'outs_when_up', 'inning', 'inning_topbot'
]