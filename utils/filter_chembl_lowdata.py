"""
Filter Chembl dataset CSV to tasks with either <= or >=  number instances provided
Uses/expects chembl.csv data provided in Chemprop repo
@Author: Aneesh Pappu
"""

import argparse
import os
import csv
import operator
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='Location of chembl.csv',
            required=True)
    parser.add_argument('--n_instances', type=int, help="""n_instance
            thresholding to filter by""", required = True)
    parser.add_argument('--thresholding', type=str, help="""Whether to filter
    tasks based on instances => n_instances or instances <= n_instances.
    Options are \'less\' and \'more\'""", required = True)
    parser.add_argument('--save_dir', type=str, help="""Location to save filtered
    dataset to""", required = 'True')
    args = parser.parse_args()
    
    assert args.thresholding in ['less', 'more']
    
    # Make savedir/ensure it exists
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    if args.thresholding == 'less':
        op = operator.le
    else:
        op = operator.ge

    chembl_df = pd.read_csv(args.data_dir)
    n_instances = args.n_instances
    # To filter, we just need to a) delete columns that have < or > than
    # desired instances and b) delete any empty rows as those compounds are no
    # longer relevant
    cols_to_drop = []
    for c in chembl_df.columns[1:]:
        # print('c: {}'.format(c))
        # print(chembl_df[c].count())
        instance_counts = chembl_df[c].count()
        if not op(instance_counts, n_instances):
            cols_to_drop.append(c)
    print("""Planning to drop {} cols out of {} total, leaving {}
            left""".format(len(cols_to_drop), len(chembl_df.columns),
                len(chembl_df.columns) - len(cols_to_drop)))
    chembl_df.drop(cols_to_drop, axis = 1, inplace=True)
    # print("""sanity check, number of columns remaining:
    #         {}""".format(len(chembl_df.columns)))
    # Drop empty rows now
    tasks_rem = chembl_df.columns.tolist()[1:]
    chembl_df.dropna(subset=tasks_rem, how='all', inplace=True)

    csv_name = 'chembl_{}_{}_instances.csv'.format(args.thresholding, n_instances)
    if args.save_dir[-1] == '/':
        path_char = ""
    else:
        path_char = "/"
    save_path = args.save_dir + path_char + csv_name
    print("writing dataset to {}".format(save_path))
    chembl_df.to_csv(save_path, index=False)
    print("Written filtered chembl to {}".format(save_path))

if __name__ == '__main__':
    main()
