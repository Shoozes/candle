"""Negative controls for typed receipt validation; zero model forwards."""
import copy
import unittest

from verify_run import strict_json, verify_answer


class Answers(unittest.TestCase):
    def setUp(self):
        self.choice = {"type": "choice", "criteria": {"second": "Second", "first": "First"}}
        self.chosen = {"type": "choice", "choice": "second", "confidence": 0.5,
                       "probabilities": {"second": 0.5, "first": 0.5}}
        self.score = {"type": "score", "criteria": ["low", "middle", "high"]}
        self.scored = {"type": "score", "score": 1.5, "confidence": 0.625,
                       "probabilities": {"0": 0.125, "1": 0.25, "2": 0.625},
                       "legend": {"0": "low", "1": "middle", "2": "high"}}

    def test_valid_typed_outputs_and_first_tie(self):
        self.assertEqual(verify_answer(self.choice, self.chosen, [.5, .5]), 0.)
        self.assertEqual(verify_answer(self.score, self.scored, [.125, .25, .625]), 0.)
        self.assertEqual(verify_answer({"type": "noul"}, {"type": "noul", "noul": .25}, [.25, .75]), 0.)

    def test_duplicate_keys_and_nonfinite_json_are_rejected(self):
        for text in ['{"noul":NaN}', '{"noul":Infinity}', '{"noul":-Infinity}',
                     '{"answers":{"x":1,"x":2}}']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                strict_json(text)

    def test_nonfinite_and_out_of_range_probabilities_fail(self):
        for value in [float("nan"), float("inf"), -0.1, 1.1, True, None]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                verify_answer({"type": "noul"}, {"type": "noul", "noul": value}, [.25, .75])
        with self.assertRaises(ValueError):
            verify_answer(self.choice, self.chosen, [float("nan"), .5])

    def test_wrong_selection_and_confidence_fail_with_correct_probabilities(self):
        for field, value in [("choice", "first"), ("confidence", .9), ("confidence", float("nan"))]:
            answer = copy.deepcopy(self.chosen)
            answer[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify_answer(self.choice, answer, [.5, .5])

    def test_wrong_score_and_legend_fail_with_correct_probabilities(self):
        for field, value in [("score", 1.), ("score", float("inf")),
                             ("legend", {"0": "low", "1": "high", "2": "middle"})]:
            answer = copy.deepcopy(self.scored)
            answer[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify_answer(self.score, answer, [.125, .25, .625])

    def test_missing_extra_or_reordered_fields_fail(self):
        cases = []
        answer = copy.deepcopy(self.chosen); del answer["confidence"]; cases.append(answer)
        answer = copy.deepcopy(self.chosen); answer["extra"] = "unknown"; cases.append(answer)
        answer = copy.deepcopy(self.chosen); answer["probabilities"] = {"first": .5, "second": .5}; cases.append(answer)
        for answer in cases:
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                verify_answer(self.choice, answer, [.5, .5])
        with self.assertRaises(ValueError):
            verify_answer(self.choice, self.chosen, [.5])


if __name__ == "__main__":
    unittest.main()
