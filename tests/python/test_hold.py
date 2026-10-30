import numpy as np
import pytest
from robot_sim.precision import HoldTracker, rollout, summarize


def test_hold_requires_consecutive_simulation_time():
    h = HoldTracker(.005, .5)
    assert not h.update(1., .004)
    assert not h.update(1.49, .005)
    assert not h.update(1.5, .006)  # Crossing the limit resets the timer.
    assert not h.update(2., .004)
    assert h.update(2.5, .004)
    assert h.metrics()['post_reach_max_distance'] == .006
    assert h.metrics()['longest_hold_seconds'] == .5


def test_hold_zero_and_invalid_configuration():
    assert HoldTracker(.005, 0).update(.1, .005)
    for seconds in (-1, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            HoldTracker(.005, seconds)


def test_summary_does_not_count_first_entry_as_hold_success():
    result = summarize([dict(final_distance=.001, success=False, hold_success=False,
                             reached=True, initial_distance=.1, reward=0, length=10)])
    assert result['success_rate'] == result['hold_success_rate'] == 0
    assert result['reach_success_rate'] == 1


@pytest.mark.parametrize('seconds,expected_calls,success', [(.5,4,False),(.05,3,True),(0.,1,True)])
@pytest.mark.parametrize('robot', ['so101', 'panda', 'ur5e'])
def test_hold_rollout_uses_current_policy_until_timeout(robot,seconds,expected_calls,success):
    from robot_sim.config import TaskConfig
    from dataclasses import asdict
    class ZeroPolicy:
        calls = 0
        def predict(self, obs, deterministic):
            self.calls += 1
            return np.zeros((len(obs['observation'])-3)//2, np.float32), None
    config = dict(robot=robot, task_config=asdict(TaskConfig(success_threshold=1., max_episode_steps=4)),
                  max_delta=.05, distance_scale=.1, rate_limit=.1, control_frequency=30.)
    model = ZeroPolicy()
    result = rollout(model, config, episodes=1, hold_seconds=seconds)
    assert model.calls == expected_calls
    assert result['reach_success_rate'] == 1
    assert result['hold_success_rate'] == float(success)
    assert result['outcomes'][0]['length'] == expected_calls
