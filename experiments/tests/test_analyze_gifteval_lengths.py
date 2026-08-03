from experiments.analyze_gifteval_lengths import summarize_lengths


def test_summarize_lengths():
    result = summarize_lengths(
        [100, 200, 300, 400], ge_name="x", term="short", display="X")
    assert result["n_instances"] == 4
    assert result["min"] == 100
    assert result["median"] == 250
    assert result["mean"] == 250
    assert result["max"] == 400
