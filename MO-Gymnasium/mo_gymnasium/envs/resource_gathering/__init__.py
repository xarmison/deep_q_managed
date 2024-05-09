from gymnasium.envs.registration import register

register(
    id="resource-gathering-v0",
    entry_point="mo_gymnasium.envs.resource_gathering.resource_gathering:ResourceGathering",
    max_episode_steps=100,
)

register(
    id="modified-resource-gathering-v0",
    entry_point="mo_gymnasium.envs.resource_gathering.custom_resource_gathering:ModifiedResourceGathering",
    max_episode_steps=100,
)
