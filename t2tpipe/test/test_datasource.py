from t2tpipe.datasource import IterableDataSource, ParquetDataSource
from t2tpipe.test.util import dummy_env


class TestIterableDatasource:
    def test_output(self):
        source = IterableDataSource([1, 2, 3])
        source.setup(dummy_env)
        assert list(iter(source)) == [1, 2, 3]

    def test_len(self):
        source = IterableDataSource([1, 2, 3])
        source.setup(dummy_env)
        assert len(source) == 3


class TestParquetDatasource:
    def test_output(self):
        source = ParquetDataSource("t2tpipe/test/data/test.parquet")
        source.setup(dummy_env)
        assert list(iter(source)) == [{"a": 1, "b": 2}] * 10

    def test_len(self):
        source = ParquetDataSource("t2tpipe/test/data/test.parquet")
        source.setup(dummy_env)
        assert len(source) == 10
