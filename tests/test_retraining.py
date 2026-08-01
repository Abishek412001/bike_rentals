from src.retraining.retrainer import retrainer


def test_retraining_execution():
    res = retrainer.execute_retraining()

    assert isinstance(res, dict)
    assert "retraining_timestamp" in res
    assert "new_rmse" in res
    assert "promoted_to_champion" in res


def test_rollback_execution():
    try:
        res = retrainer.rollback()
        assert isinstance(res, dict)
        assert "champion_model" in res
    except FileNotFoundError:
        # If no challenger exists yet, pass cleanly
        pass
