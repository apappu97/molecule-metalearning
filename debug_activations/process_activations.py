import pickle
import os 

percentage_dict = {}
THRESHOLD = .03

for directory in os.listdir('histogram_output'):
    curr_path = os.path.join('histogram_output', directory)
    percentages = []
    print(curr_path)
    for i, file in enumerate(os.listdir(curr_path)):
        if i == len(os.listdir(curr_path)) - 1: continue # skip last pickle file as was corrupted during write
        print('Processing i: {}'.format(i))
        with open(os.path.join(curr_path, file), 'rb') as handle:
            curr_acts = pickle.load(handle)
            for sub_list in curr_acts:
                num_left = [x for x in sub_list if x <= THRESHOLD]
                ratio = len(num_left) * 1.0/len(sub_list)
                percentages.append(ratio)
        
    percentage_dict[curr_path] = percentages

with open('percentage_dict.pickle', 'wb') as handle:
    pickle.dump(percentage_dict, handle)