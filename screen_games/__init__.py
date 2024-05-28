from gymnasium.envs.registration import register

register(
    id="screen_games/ScreenEnv-v0",
    entry_point="screen_games.envs:ScreenEnv"
)