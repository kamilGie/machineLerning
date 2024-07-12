import subprocess

def run_python_script(script_name):
    try:
        # if different python version is used, change the version number
        result = subprocess.run(['python3.10', script_name], check=True, stdout=subprocess.PIPE)
        print(result.stdout.decode())
    except subprocess.CalledProcessError as e:
        print("Error executing command:", e)

if __name__ == "__main__":
    run_python_script('logisiticRegression.py')
    run_python_script('k-NearestNeighbors.py')
    run_python_script('linearSvm.py')
    run_python_script('kernelSvm.py')
    run_python_script('NaiveBayes.py')
    run_python_script('DecisionTree.py')
    run_python_script('randomForest.py')