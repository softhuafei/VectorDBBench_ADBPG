from vectordb_bench.backend.clients.adbpg import hybrid_synth


def test_unified_percentile_is_stable_and_bounded():
    values = [hybrid_synth.filter_percentile_for(i) for i in range(100)]
    assert values == [hybrid_synth.filter_percentile_for(i) for i in range(100)]
    assert all(0 <= value < 10_000 for value in values)


def test_scalar_array_json_truth_sets_are_identical():
    rate = 0.2
    threshold = hybrid_synth.unified_threshold_for_rate(rate)
    marker = hybrid_synth.unified_rate_marker_for_rate(rate)

    for row_id in range(1_000):
        scalar_hit = hybrid_synth.filter_percentile_for(row_id) < threshold
        array_hit = marker in hybrid_synth.unified_user_array_for(row_id, filler_len=0)
        json_hit = marker in hybrid_synth.unified_payload_for(row_id)["rates"]
        assert scalar_hit == array_hit == json_hit


def test_unified_truth_sets_are_nested():
    marker_20 = hybrid_synth.unified_rate_marker_for_rate(0.2)
    marker_30 = hybrid_synth.unified_rate_marker_for_rate(0.3)
    for row_id in range(1_000):
        markers = hybrid_synth.unified_rate_markers_for_id(row_id)
        if marker_20 in markers:
            assert marker_30 in markers


def test_unified_filters_share_groundtruth_file():
    from vectordb_bench.backend.filter import (
        ArrayContainsFilter,
        JsonContainsFilter,
        PercentileLTFilter,
    )

    filters = [
        PercentileLTFilter(filter_rate=0.2, threshold=2_000),
        ArrayContainsFilter(filter_rate=0.2, values=["r_2000bp"]),
        JsonContainsFilter(filter_rate=0.2, marker="r_2000bp"),
    ]
    assert {item.groundtruth_file for item in filters} == {"neighbors_hybrid_20p.parquet"}


def test_join_marker_and_groundtruth_remain_independent():
    from vectordb_bench.backend.filter import JoinArrayOverlapFilter

    marker = hybrid_synth.rate_marker_for_rate(0.1)
    join_filter = JoinArrayOverlapFilter(filter_rate=0.1, values=[marker])

    assert marker == "r_10"
    assert join_filter.groundtruth_file == "neighbors_join_10p.parquet"
