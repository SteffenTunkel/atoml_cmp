import unittest
from atoml_cmp.run_tool import main


class TestRunThrough(unittest.TestCase):
    def test_atoml_cmp_usecase(self):
        """Tests a possible run through of the overall tool (including docker build, docker run and evaluation).

        Assert how many csv files the evaluation unit read.
        In order to make sure that the test fails if the pipeline runs through empty."""
        num_datasets = 3
        num_algorithms = 1
        num_frameworks = 4
        expected_num_csv_files = num_frameworks * num_algorithms * num_datasets

        csv_files_evaluated = main("tests/system_tests/test_resources/test_dockerlist.json",
                                   yaml_folder="tests/system_tests/test_resources/test_algorithm_descriptions")
        self.assertEqual(csv_files_evaluated, expected_num_csv_files)
