import subprocess
import os
datasets = ["churn", "telco_churn", "santander_sub", "covtype_bin_sub", "spambase", "shoppers", "diabetes", "breast_cancer", "ionosphere", "philippine"]
current_dir = os.getcwd()
processes = []
for dataset in datasets:
    f = open("run_" + dataset + ".txt", "w")
    os.chdir("../" + str(dataset))
    p = subprocess.Popen(["python3 read_and_collate.py"], stdout = f, stderr = f, shell=True)
    processes.append((p, f))
    os.chdir(current_dir)


for p, f in processes:
    p.wait()
