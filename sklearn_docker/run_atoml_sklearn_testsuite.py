import os

path="generated-tests/sklearn"
test_scripts = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
print(test_scripts)
os.chdir(path)
for file in test_scripts:
	os.system('python ' + file)

