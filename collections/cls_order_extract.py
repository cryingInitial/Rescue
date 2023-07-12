import json
import os

dir = "cifar10"
file_list = os.listdir(dir)

for dir_file in file_list:    
    file_path = dir + "/" + dir_file
    print(file_path)
    if "val" in file_path:
        continue
    
    with open(file_path, 'r') as file:
        data = json.load(file)

    cls_order = []
    future_train_dict_k5 = {}
    future_train_dict_k10 = {}
    future_train_dict_k20 = {}
    future_train_dict_k100 = {}

    for sample in data["stream"]:
        if sample["klass"] not in cls_order:
            cls_order.append(sample["klass"])
            future_train_dict_k5[sample["klass"]] = [sample]
            future_train_dict_k10[sample["klass"]] = [sample]
            future_train_dict_k20[sample["klass"]] = [sample]
            future_train_dict_k100[sample["klass"]] = [sample]
        else:
            if len(future_train_dict_k5[sample["klass"]]) < 5:
                future_train_dict_k5[sample["klass"]].append(sample)
            if len(future_train_dict_k10[sample["klass"]]) < 10:
                future_train_dict_k10[sample["klass"]].append(sample)
            if len(future_train_dict_k20[sample["klass"]]) < 20:
                future_train_dict_k20[sample["klass"]].append(sample)
            if len(future_train_dict_k100[sample["klass"]]) < 100:
                future_train_dict_k100[sample["klass"]].append(sample)
            
    data["cls_order"] = cls_order
    data["future_train_dict_k5"] = future_train_dict_k5
    data["future_train_dict_k10"] = future_train_dict_k10
    data["future_train_dict_k20"] = future_train_dict_k20
    data["future_train_dict_k100"] = future_train_dict_k100
    
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file)