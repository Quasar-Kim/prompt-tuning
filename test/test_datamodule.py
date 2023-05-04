from data.datamodule import ParquetFileLoader

def test_parquet_file_loader():
    loader = ParquetFileLoader('data/khs/train.parquet')
    sample = list(loader)[0]
    assert sample.text == '(현재 호텔주인 심정) 아18 난 마른하늘에 날벼락맞고 호텔망하게생겼는데 누군 계속 추모받네....'
    assert sample.label == 'hate'