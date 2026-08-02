import pytest

from kreyol_asr.train import check_att_context


def test_accepts_supported_latencies_and_normalizes_to_list_of_pairs():
    for right in (0, 3, 6, 13):
        assert check_att_context([56, right]) == [[56, right]]


def test_preserves_the_checkpoints_multi_latency_list():
    # The released checkpoint trains on all four contexts at once; narrowing to a
    # single pair would specialise the model and regress the other latencies.
    contexts = [[56, 3], [56, 0], [56, 6], [56, 13]]
    assert check_att_context(contexts) == contexts


def test_rejects_stock_yaml_left_context():
    # The shipped NeMo YAML says [70, 6]; this checkpoint needs 56. Catching this
    # is the whole point of the guard.
    with pytest.raises(ValueError, match="left context is 70"):
        check_att_context([70, 6])


def test_rejects_unsupported_right_context():
    with pytest.raises(ValueError, match="right context 4 unsupported"):
        check_att_context([56, 4])
