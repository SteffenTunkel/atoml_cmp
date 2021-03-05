import os
import sys

print("\nRun Sklearn Test Suite")
path="generated-tests/sklearn"
test_scripts = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
print(test_scripts)
sys.stdout.flush() # to get the print commands before the system call
os.chdir(path)
for file in test_scripts:
	os.system('python ' + file)

