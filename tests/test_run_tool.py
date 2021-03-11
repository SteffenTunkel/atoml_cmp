import unittest
from atoml_cmp.run_tool import main


class TestRunThrough(unittest.TestCase):
	def test_atoml_cmp_usecase(self):
		num_logreg_csv = 4*8
		num_knn_csv = 3*8
		expected_value = 32
		csv_files_evaluated = main("dockerlist.json")
		#self.assertEqual(csv_files_evaluated, expected_value)

