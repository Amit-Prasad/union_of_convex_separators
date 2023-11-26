import pandas as pd
import ast

data_name = "philippine"
output = []

with open("temp_" + "all_rf_" + data_name + ".txt") as f:
    data = f.read()
output_dict = ast.literal_eval(data)
df = pd.DataFrame(output_dict)
df.to_csv('all_rf_' + str(data_name) + '.csv', index=False)
output.append(output_dict)

with open("temp_" + "all_xg_" + data_name + ".txt") as f:
    data = f.read()
output_dict = ast.literal_eval(data)
df = pd.DataFrame(output_dict)
df.to_csv('all_xg_' + str(data_name) + '.csv', index=False)
output.append(output_dict)

with open("temp_" + "all_slp_" + data_name + ".txt") as f:
    data = f.read()
output_dict = ast.literal_eval(data)
df = pd.DataFrame(output_dict)
df.to_csv('all_slp_' + str(data_name) + '.csv', index=False)
output.append(output_dict)

with open("temp_" + "all_mlp2_" + data_name + ".txt") as f:
    data = f.read()
output_dict = ast.literal_eval(data)
df = pd.DataFrame(output_dict)
df.to_csv('all_mlp2_' + str(data_name) + '.csv', index=False)
output.append(output_dict)

with open("temp_" + "all_mlp3_" + data_name + ".txt") as f:
    data = f.read()
output_dict = ast.literal_eval(data)
df = pd.DataFrame(output_dict)
df.to_csv('all_mlp3_' + str(data_name) + '.csv', index=False)
output.append(output_dict)


df = pd.DataFrame.from_records(output)
df.to_csv('all_' + str(data_name) + '.csv', index=False)
