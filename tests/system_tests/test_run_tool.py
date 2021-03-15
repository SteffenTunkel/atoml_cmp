import unittest
from atoml_cmp.run_tool import main
from atoml_cmp.external_data_utils import create_wine_arff, create_breast_cancer_arff


class TestRunThrough(unittest.TestCase):
    def test_atoml_cmp_usecase(self):
        num_logreg_csv = 4 * 8
        create_wine_arff()
        create_breast_cancer_arff()
        csv_files_evaluated = main("tests/system_tests/resources/test_dockerlist.json",
                                   yaml_folder="tests/system_tests/resources/test_algorithm_descriptions")
        self.assertEqual(csv_files_evaluated, num_logreg_csv)
