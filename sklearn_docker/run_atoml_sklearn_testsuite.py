import os

path="/sklearn"
testScripts = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
print(testScripts)
print(os.system("dir"))
os.chdir(path)
print(os.system("dir"))
for file in testScripts:
	os.system('python ' + file)

