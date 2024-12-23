import numpy as np

import data
import word2vec


class TestModel:

    model = word2vec.SGNS(2, 2)
    model.embeddings = np.array([[0.8, -0.8], [0.2, 0.4]])
    model.context_embeddings = np.array([[0.3, 0.3], [-0.3, 0.7]])

    def test_forward(self):
        np.testing.assert_allclose(
            TestModel.model.forward((0, 0))["probability"], np.array(0.5)
        )
        np.testing.assert_allclose(
            TestModel.model.forward((0, 1))["probability"],
            np.array(0.31002),
            rtol=0,
            atol=1e-5,
        )


class TestGradients:

    positive_result = {
        "target_word_embedding": np.array([0.8, -0.8]),
        "context_word_embedding": np.array([0.3, 0.3]),
        "probability": 0.5,
    }

    negative_results = [
        {
            "target_word_embedding": np.array([0.8, -0.8]),
            "context_word_embedding": np.array([-0.3, 0.7]),
            "probability": 0.31003,
        },
        {
            "target_word_embedding": np.array([0.8, -0.8]),
            "context_word_embedding": np.array([-0.2, 0.4]),
            "probability": 0.38225,
        },
    ]

    def test_positive_context_gradient(self):
        np.testing.assert_allclose(
            word2vec.get_positive_context_gradient(
                TestGradients.positive_result, TestGradients.negative_results
            ),
            np.array([-0.4, 0.4]),
        )

    def test_negative_context_gradients(self):
        negative_gradients = word2vec.get_negative_context_gradients(
            TestGradients.positive_result, TestGradients.negative_results
        )
        np.testing.assert_allclose(
            negative_gradients[0], np.array([0.24802, -0.24802]), rtol=0, atol=1e-5
        )
        np.testing.assert_allclose(negative_gradients[1], np.array([0.3058, -0.3058]), rtol=0, atol=1e-4)

    def test_target_word_gradient(self):
        target_gradient = word2vec.get_target_word_gradient(
            TestGradients.positive_result, TestGradients.negative_results
        )
        np.testing.assert_allclose(
            target_gradient, np.array([-0.31945, 0.21992]), rtol=0, atol=1e-5
        )


class TestData:

    text = "the cat sat on the mat".split(" ")

    def test_positive_samples(self):
        positive_samples = data.get_positive_samples(TestData.text, 2, TestData.text)
        print(positive_samples) 
        assert set(positive_samples) == set(
            [
                ("the", "cat"),
                ("the", "sat"),
                ("cat", "the"),
                ("cat", "sat"),
                ("cat", "on"),
                ("sat", "the"),
                ("sat", "cat"),
                ("sat", "on"),
                ("sat", "the"),
                ("on", "cat"),
                ("on", "sat"),
                ("on", "the"),
                ("on", "mat"),
                ("the", "sat"),
                ("the", "on"),
                ("the", "mat"),
                ("mat", "on"),
                ("mat", "the"),
            ]
        )

    def test_negative_samples(self):
        # NB: not a full test
        weights = [1] * len(TestData.text)
        avoid = "the"
        negative_samples = []
        # repeat sampling a few times
        for _ in range(5):
            negative_samples.extend(
                data.negative_samples(
                    TestData.text, weights=weights, num_samples=10, avoid=avoid
                )
            )
        assert "the" not in negative_samples
