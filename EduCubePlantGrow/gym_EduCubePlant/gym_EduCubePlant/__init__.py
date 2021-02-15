from gym.envs.registration import register

register(
    id='EduCubePlant-v0',
    entry_point='gym_EduCubePlant.envs:EduCubePlantEnv',
)
#register(id='EduCubePlant-extrahard-v0',entry_point='gym_EduCubePlant.envs:EduCubePlantExtraHardEnv',)