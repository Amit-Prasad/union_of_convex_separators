import pandas as pd
import subprocess
import ast
datasets = ["ionosphere"]
options = 'all'

for data_name in datasets:
    if options == 'rf':
        f = open("temp_" + "rf_" + data_name + ".txt", "w")
        subprocess.run(["python3 rf.py " + data_name], stderr=f, stdout=f, shell=True)
        f.close()
        with open("temp_" + "rf_" + data_name + ".txt") as f:
            data = f.read()
        output_dict = ast.literal_eval(data)
        df = pd.DataFrame(output_dict)
        df.to_csv('rf_' + str(data_name) + '.csv', index=False)

    elif options == 'xg':
        f = open("temp_" + "xg_" + data_name + ".txt", "w")
        subprocess.run(["python3 xg.py " + data_name], stderr=f, stdout=f, shell=True)
        f.close()
        with open("temp_" + "xg_" + data_name + ".txt") as f:
            data = f.read()
        output_dict = ast.literal_eval(data)
        df = pd.DataFrame(output_dict)
        df.to_csv('xg_' + str(data_name) + '.csv', index=False)

    elif options == 'slp':
        f = open("temp_" + "slp_" + data_name + ".txt", "w")
        subprocess.run(["python3 slp.py " + data_name], stderr=f, stdout=f, shell=True)
        f.close()
        with open("temp_" + "slp_" + data_name + ".txt") as f:
            data = f.read()
        output_dict = ast.literal_eval(data)
        df = pd.DataFrame(output_dict)
        df.to_csv('slp_' + str(data_name) + '.csv', index=False)

    elif options == 'mlp2':
        f = open("temp_" + "mlp2_" + data_name + ".txt", "w")
        subprocess.run(["python3 mlp2.py " + data_name], stderr=f, stdout=f, shell=True)
        f.close()
        with open("temp_" + "mlp2_" + data_name + ".txt") as f:
            data = f.read()
        output_dict = ast.literal_eval(data)
        df = pd.DataFrame(output_dict)
        df.to_csv('mpl2_' + str(data_name) + '.csv', index=False)

    elif options == 'mlp3':
        f = open("temp_" + "mlp3_" + data_name + ".txt", "w")
        subprocess.run(["python3 mlp3.py " + data_name], stderr=f, stdout=f, shell=True)
        f.close()
        with open("temp_" + "mlp3_" + data_name + ".txt") as f:
            data = f.read()
        output_dict = ast.literal_eval(data)
        df = pd.DataFrame(output_dict)
        df.to_csv('mlp3_' + str(data_name) + '.csv', index=False)

    elif options == 'all':
        f = open("temp_" + "all_rf_" + data_name + ".txt", "w")
        subprocess.run(["python3 rf.py " + data_name], stderr=f, stdout=f, shell=True)
        f.close()


        #f = open("temp_" + "all_xg_" + data_name + ".txt", "w")
        #subprocess.run(["python3 xg.py " + data_name], stderr=f, stdout=f, shell=True)
        #f.close()


        #f = open("temp_" + "all_slp_" + data_name + ".txt", "w")
        #subprocess.run(["python3 slp.py " + data_name], stderr=f, stdout=f, shell=True)
        #f.close()


        #f = open("temp_" + "all_mlp2_" + data_name + ".txt", "w")
        #subprocess.run(["python3 mlp2.py " + data_name], stderr=f, stdout=f, shell=True)
        #f.close()


        #f = open("temp_" + "all_mlp3_" + data_name + ".txt", "w")
        #subprocess.run(["python3 mlp3.py "+ data_name], stderr=f, stdout=f, shell=True)
        #f.close()

        ##########################################################
        output = []

        with open("temp_" + "all_rf_" + data_name + ".txt") as f:
            data = f.read()
        output_dict = ast.literal_eval(data)
        output.append(output_dict)
        df = pd.DataFrame(output_dict)
        df.to_csv('all_rf_' + str(data_name) + '.csv', index=False)

        with open("temp_" + "all_xg_" + data_name + ".txt") as f:
            data = f.read()
        output_dict = ast.literal_eval(data)
        output.append(output_dict)
        df = pd.DataFrame(output_dict)
        df.to_csv('all_xg_' + str(data_name) + '.csv', index=False)

        with open("temp_" + "all_slp_" + data_name + ".txt") as f:
            data = f.read()
        output_dict = ast.literal_eval(data)
        output.append(output_dict)
        df = pd.DataFrame(output_dict)
        df.to_csv('all_slp_' + str(data_name) + '.csv', index=False)

        with open("temp_" + "all_mlp2_" + data_name + ".txt") as f:
            data = f.read()
        output_dict = ast.literal_eval(data)
        output.append(output_dict)
        df = pd.DataFrame(output_dict)
        df.to_csv('all_mlp2_' + str(data_name) + '.csv', index=False)

        with open("temp_" + "all_mlp3_" + data_name + ".txt") as f:
            data = f.read()
        output_dict = ast.literal_eval(data)
        output.append(output_dict)

        df = pd.DataFrame(output_dict)
        df.to_csv('all_mlp3_' + str(data_name) + '.csv', index=False)

        df = pd.DataFrame.from_records(output)
        df.to_csv('all_' + str(data_name) + '.csv', index=False)

# f = open("temp.txt", "w")
# subprocess.run(["python3 rf.py churn"], stdout=f, shell=True)
# f.close()
# with open("temp.txt") as f:
#     data = f.read()
# output_dict = ast.literal_eval(data)
# print(output_dict)
