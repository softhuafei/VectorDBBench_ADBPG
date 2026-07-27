from enum import StrEnum

from pydantic import Field

from ..base import BaseModel


class FilterOp(StrEnum):
    NumGE = "NumGE"  # test ">="
    StrEqual = "Label"  # test "=="
    NonFilter = "NonFilter"
    ArrayContains = "ArrayContains"  # GIN @> on a single-table array column
    JoinArrayOverlap = "JoinArrayOverlap"  # subquery IN (SELECT ... WHERE tags && ARRAY[...])
    PercentileLT = "PercentileLT"
    JsonContains = "JsonContains"


class Filter(BaseModel):
    type: FilterOp
    filter_rate: float = 0.0

    @property
    def groundtruth_file(self) -> str:
        raise NotImplementedError


class NonFilter(Filter):
    type: FilterOp = FilterOp.NonFilter
    filter_rate: float = 0.0
    gt_file_name: str = "neighbors.parquet"

    @property
    def groundtruth_file(self) -> str:
        return self.gt_file_name


non_filter = NonFilter()


class IntFilter(Filter):
    """
    compatible with older int-filter cases
    filter expr: int_field >= int_value (dataset_size * filter_rate)
    """

    type: FilterOp = FilterOp.NumGE
    int_field: str = "id"
    int_value: int

    @property
    def groundtruth_file(self) -> str:
        if self.filter_rate == 0.01:
            return "neighbors_head_1p.parquet"
        if self.filter_rate == 0.99:
            return "neighbors_tail_1p.parquet"
        msg = f"Not Support Int Filter - {self.filter_rate}"
        raise RuntimeError(msg)


class NewIntFilter(Filter):
    type: FilterOp = FilterOp.NumGE
    int_field: str = "id"
    int_value: int

    @property
    def int_rate(self) -> str:
        r = self.filter_rate * 100
        if 1 <= r <= 99:
            return f"int_{int(r)}p"
        return f"int_{r:.1f}p"

    @property
    def groundtruth_file(self) -> str:
        return f"neighbors_{self.int_rate}.parquet"


class LabelFilter(Filter):
    """
    filter expr: label_field == label_value, like `color == "red"`
    """

    type: FilterOp = FilterOp.StrEqual
    label_field: str = "labels"
    label_percentage: float

    @property
    def label_value(self) -> str:
        p = self.label_percentage * 100
        if p >= 1:
            return f"label_{int(p)}p"  # such as 5p, 20p, 1p, ...
        return f"label_{p:.1f}p"  # such as 0.1p, 0.5p, ...

    def __init__(self, label_percentage: float, **kwargs):
        filter_rate = 1.0 - label_percentage
        super().__init__(filter_rate=filter_rate, label_percentage=label_percentage, **kwargs)

    @property
    def groundtruth_file(self) -> str:
        return f"neighbors_{self.label_field}_{self.label_value}.parquet"


class ArrayContainsFilter(Filter):
    """Single-table GIN array containment predicate.

    Filter expression: <array_field> @> ARRAY[<values>]
    """

    type: FilterOp = FilterOp.ArrayContains
    array_field: str = "user_array"
    values: list[str] = Field(default_factory=list)

    @property
    def groundtruth_file(self) -> str:
        return f"neighbors_hybrid_{hybrid_rate_label(self.filter_rate)}.parquet"


class JoinArrayOverlapFilter(Filter):
    """Two-table JOIN predicate: chunk.<join_field> IN (SELECT <join_field> FROM doc WHERE <tags_field> && ARRAY[<values>])."""

    type: FilterOp = FilterOp.JoinArrayOverlap
    join_field: str = "pipeline_doc_id"
    tags_field: str = "tags"
    doc_table: str = "doc_table"
    values: list[str] = Field(default_factory=list)

    @property
    def rate_label(self) -> str:
        r = self.filter_rate * 100
        if r >= 1:
            return f"{int(r)}p"
        return f"{r:.2f}p"

    @property
    def groundtruth_file(self) -> str:
        return f"neighbors_join_{self.rate_label}.parquet"


def hybrid_rate_label(rate: float) -> str:
    percent = rate * 100
    if abs(percent - round(percent)) < 1e-9:
        return f"{int(round(percent))}p"
    return f"{percent:.2f}".rstrip("0").rstrip(".").replace(".", "_") + "p"


class UnifiedHybridFilter(Filter):
    """Base for predicates sharing one percentile-derived truth set."""

    @property
    def groundtruth_file(self) -> str:
        return f"neighbors_hybrid_{hybrid_rate_label(self.filter_rate)}.parquet"


class PercentileLTFilter(UnifiedHybridFilter):
    type: FilterOp = FilterOp.PercentileLT
    percentile_field: str = "percentile"
    threshold: int


class JsonContainsFilter(UnifiedHybridFilter):
    type: FilterOp = FilterOp.JsonContains
    json_field: str = "payload"
    marker: str
