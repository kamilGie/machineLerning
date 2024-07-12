import subprocess

def run_python_script(script_name):
    try:
        # if different python version is used, change the version number
        result = subprocess.run(['python3.10', script_name], check=True, stdout=subprocess.PIPE)
        print(result.stdout.decode())
    except subprocess.CalledProcessError as e:
        print("Error executing command:", e)

if __name__ == "__main__":
    run_python_script('MultipleLinearRegression.py')
    run_python_script('PolynomialLinearRegression.py')
    run_python_script('SupportVectorRegression.py')
    run_python_script('DecisionTreeRegression.py')
    run_python_script('RandomForestRegression.py')
    