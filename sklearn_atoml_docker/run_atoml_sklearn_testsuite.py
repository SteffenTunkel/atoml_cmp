import os

path="/sklearn"
testScripts = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
print(testScripts)
print(os.system("dir"))
os.chdir(path)
print(os.system("dir"))
#os.system('cd ' + path)
for file in testScripts:
	#os.system('python ' + path + '/' + file)
	os.system('python ' + file)

