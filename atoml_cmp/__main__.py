import atoml_cmp.run_tool
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dockerlist', default="dockerlist.json", metavar='',
                    help='json file with docker descriptions ["dockerlist.json"]')
parser.add_argument('-y', '--yaml_desc', default="algorithm-descriptions", metavar='',
                    help='directory of the yaml files with the algorithm definitions ["algorithm-descriptions"]')
parser.add_argument('-t', '--testcases', default="generated-tests", metavar='',
                    help='directory for the generated test cases ["generated-tests"]')
parser.add_argument('-p', '--predictions', default="predictions", metavar='',
                    help='directory for the prediction csv files ["predictions"]')
parser.add_argument('-a', '--archive', default="archive", metavar='',
                    help='directory, where to save the archive ["archive"]')
parser.add_argument('-m', '--manual', action='store_true',
                    help='flag for manual decisions in program run [False]')
args = parser.parse_args()

atoml_cmp.run_tool.main(dockerlist_file=args.dockerlist, gen_tests_folder=args.testcases, pred_folder=args.predictions,
                        yaml_folder=args.yaml_desc, archive_folder=args.archive, manual_flag=args.manual)
