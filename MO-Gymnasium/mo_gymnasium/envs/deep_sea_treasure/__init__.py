from gymnasium.envs.registration import register

from mo_gymnasium.envs.deep_sea_treasure.deep_sea_treasure import CONCAVE_MAP

from mo_gymnasium.envs.deep_sea_treasure.custom_deep_sea_treasure import BST_BEST_PATHS_LENGTHS, MBST_BEST_PATHS_LENGTHS
from mo_gymnasium.envs.deep_sea_treasure.custom_deep_sea_treasure import BST_TREASURES, MBST_TREASURES
from mo_gymnasium.envs.deep_sea_treasure.custom_deep_sea_treasure import BST_MAP, MBST_MAP

register(
    id='deep-sea-treasure-v0',
    entry_point='mo_gymnasium.envs.deep_sea_treasure.deep_sea_treasure:DeepSeaTreasure',
    max_episode_steps=100,
)

register(
    id='deep-sea-treasure-concave-v0',
    entry_point='mo_gymnasium.envs.deep_sea_treasure.deep_sea_treasure:DeepSeaTreasure',
    max_episode_steps=100,
    kwargs={'dst_map': CONCAVE_MAP}
)

register(
    id='deep-sea-treasure-v1',
    entry_point='mo_gymnasium.envs.deep_sea_treasure.custom_deep_sea_treasure:DeepSeaTreasure',
    max_episode_steps=1000
)

register(
    id='bountiful-sea-treasure-v1',
    entry_point='mo_gymnasium.envs.deep_sea_treasure.custom_deep_sea_treasure:DeepSeaTreasure',
    max_episode_steps=1000,
    kwargs={
        'dst_map': BST_MAP, 
        'best_paths_lengths': BST_BEST_PATHS_LENGTHS,
        'treasures': BST_TREASURES,
        'name': 'BST'
    }
)

register(
    id='modified-bountiful-sea-treasure-v1',
    entry_point='mo_gymnasium.envs.deep_sea_treasure.custom_deep_sea_treasure:DeepSeaTreasure',
    max_episode_steps=1000,
    kwargs={
        'dst_map': MBST_MAP, 
        'best_paths_lengths': MBST_BEST_PATHS_LENGTHS,
        'treasures': MBST_TREASURES,
        'name': 'MBST'
    }
)