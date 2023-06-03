from t2tpipe.postprocessor import ClassificationPostProcessor


class TestClassificationPostProcessor:
    def test_output(self):
        postprocessor = ClassificationPostProcessor({"부정": 0, "긍정": 1})
        y = "부정"
        y_pred = "긍정"
        assert postprocessor(y) == 0
        assert postprocessor(y_pred) == 1

    def test_unk(self):
        postprocessor = ClassificationPostProcessor({"부정": 0, "긍정": 1}, unk_id=-100)
        assert postprocessor("중립") == -100
