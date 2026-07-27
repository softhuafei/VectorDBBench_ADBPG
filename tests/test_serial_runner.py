from contextlib import nullcontext

from vectordb_bench.backend.runner.serial_runner import SerialSearchRunner


class FakeVectorDB:
    def __init__(self, results):
        self._results = iter(results)

    def init(self):
        return nullcontext()

    def prepare_filter(self, _filters):
        return None

    def search_embedding(self, _embedding, _k):
        return next(self._results)


def test_serial_search_reports_result_shortage(caplog):
    runner = SerialSearchRunner(
        db=FakeVectorDB([[1, 2], [1]]),
        test_data=[[0.0], [1.0]],
        ground_truth=[[1, 2], [1, 2]],
        k=2,
        query_count=2,
    )

    result = runner.search((runner.test_data, runner.ground_truth))

    assert result[5:] == (1, 1, 0.5, 2)
    assert "SERIAL_QUERY_RESULT" in caplog.text


def test_serial_search_respects_query_count():
    runner = SerialSearchRunner(
        db=FakeVectorDB([[1, 2]]),
        test_data=[[0.0], [1.0]],
        ground_truth=[[1, 2], [1, 2]],
        k=2,
        query_count=1,
    )

    assert len(runner.test_data) == 1
    assert len(runner.ground_truth) == 1
