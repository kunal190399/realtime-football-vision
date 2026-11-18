from src.tracking.observations import Track
from src.tracking.possession import PossessionEstimator


PLAYER = Track("player", 7, (0, 0, 20, 40), 0.9)
BALL = Track("ball", 12, (8, 38, 12, 42), 0.8)


def test_confirmation_loss_and_reacquisition():
    estimator = PossessionEstimator()
    frames = [[PLAYER, BALL]] * 3 + [[PLAYER]] + [[PLAYER, BALL]]
    rows = [estimator.update(frame) for frame in frames]
    assert [row["possession_track_id"] for row in rows] == [None, None, 7, None, None]
    assert rows[2]["ball_x"] == 10
    assert rows[3]["ball_x"] is None


def test_ball_continuity_and_track_switch():
    estimator = PossessionEstimator(2)
    other = Track("ball", 99, BALL.xyxy, 0.95)
    estimator.update([PLAYER, BALL])
    assert estimator.update([PLAYER, BALL, other])["ball_track_id"] == 12
    assert estimator.update([PLAYER, other])["possession_track_id"] is None


def test_distant_players_and_untracked_balls_remain_unknown():
    estimator = PossessionEstimator(1)
    for ball in [Track("ball", 12, (100, 100, 102, 102), 0.9),
                 Track("ball", None, BALL.xyxy, 0.9)]:
        assert estimator.update([PLAYER, ball])["possession_track_id"] is None
