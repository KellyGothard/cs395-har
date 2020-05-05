description = """
Load UCI HAR

Example

python sepdat.py --col=-1 --out-dir=data score1.csv score2.csv
"""
import argparse
import os

import pandas as pd
import numpy as np
import glob


# def init_parser():
#     parser = argparse.ArgumentParser(
#         description=description, formatter_class=argparse.RawTextHelpFormatter)
#     parser.add_argument('files', type=str, nargs='+',
#                         help='Files to separate')
#     parser.add_argument('--delimiter', type=str, default=',',
#                         help='Delimiter for the CSV.')
#     parser.add_argument('--out-dir', type=str, default=None,
#                         help='The directory to output the data. Default is current directory.')
#     parser.add_argument('--col', type=int, default=-1,
#                         help='The column to base the split on. Default is last column.')
#     parser.add_argument('--npy', action='store_true',
#                         help='Flag to save as npy.')
#
#     return parser

# load a single file as a numpy array
def load_file(filepath):
    # dataframe = pd.read_csv(filepath, header=None, delim_whitespace=True)
    dataframe = pd.read_csv(filepath, header=None)
    return dataframe

def combine_train_test():

    training_features_path = "../uci_har/train/Inertial Signals/"
    testing_features_path = "../uci_har/test/Inertial Signals/"

    training_labels_path = "../uci_har/train/y_train.txt"
    testing_labels_path = "../uci_har/test/y_test.txt"

    training_subject_path = "../uci_har/train/subject_train.txt"
    testing_subject_path = "../uci_har/test/subject_test.txt"

    output_dir = "../uci_har/total/"

    train_files = glob.glob(training_features_path + 'total_acc*') + glob.glob(
        training_features_path + '*gyro*')
    test_files = glob.glob(testing_features_path + 'total_acc*') + glob.glob(
        testing_features_path + '*gyro*')


    train_files.sort()
    test_files.sort()

    for i in range(len(train_files)):
        print('Files to combine:')
        print(train_files[i])
        print(test_files[i])

        train = load_file(train_files[i])
        print('\nTrain shape: ')
        print(train.shape)

        test = load_file(test_files[i])
        print('\nTest shape: ')
        print(test.shape)

        total = train.append(test)
        print('\nNew shape: ')
        print(total.shape)

        new_name = train_files[i].split('/')[-1].split('.')[-2][:11] + '.csv'
        print(new_name)
        total.to_csv(output_dir + new_name, index = False, header = False)

    training_labels = load_file(training_labels_path)
    testing_labels = load_file(testing_labels_path)
    total = training_labels.append(testing_labels)
    total.to_csv('../uci_har/total/labels.csv', index = False, header = False)

    training_subjects = load_file(training_subject_path)
    testing_subjects = load_file(testing_subject_path)
    total = training_subjects.append(testing_subjects)
    total[0].to_csv('../uci_har/total/subjects.csv', index = False, header = False)

def combine_features():
    subjects = load_file('../uci_har/total/subjects.csv')
    print(subjects.shape)

    for file in glob.glob('../uci_har/total/*'):
        if file != '../uci_har/total/subjects.csv':
            print(file)
            feature = load_file(file)

            name = file.split('/')[-1].split('.')[-2]
            subjects[name] = feature[0]
            print(subjects.shape)

    subjects.to_csv('../uci_har/combined_dataset.csv',index=False)

def load_feature_add_subject(path,subjects, labels):
    acc_x = load_file(path)
    acc_x['subject'] = subjects
    acc_x['label'] = labels
    acc_x = acc_x#.transpose()
    return acc_x

def load_all_features():
    subjects = load_file("../uci_har/total/subjects.csv")#.transpose()
    labels = load_file("../uci_har/total/labels.csv")#.transpose()

    acc_x = load_feature_add_subject("../uci_har/total/total_acc_x.csv", subjects, labels)
    acc_y = load_feature_add_subject("../uci_har/total/total_acc_y.csv", subjects, labels)
    acc_z = load_feature_add_subject("../uci_har/total/total_acc_z.csv", subjects, labels)

    gyro_x = load_feature_add_subject("../uci_har/total/body_gyro_x.csv", subjects, labels)
    gyro_y = load_feature_add_subject("../uci_har/total/body_gyro_y.csv", subjects, labels)
    gyro_z = load_feature_add_subject("../uci_har/total/body_gyro_z.csv", subjects, labels)

    return subjects, labels, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z

def main():

    # Combined training and test set
    # combine_train_test()

    subjects, labels, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z = load_all_features()

    # print(subjects.shape)
    # print(labels.shape)
    # print(acc_x.shape)

    # d = load_file("../files16hz/Dyer_John_2-1-2020 2_RASS_L.txt_labelled.csv")
    # s = load_file("../sepdat_out/Dyer_John_2-1-2020 2_RASS_L.txt_norm_labelled_-1.csv")
    # r = load_file("../data_rolling/Dyer_John_2-1-2020 2_RASS_R.txt_norm_labelled_4_0140.csv")
    # print(d.shape)
    # print(s.shape)
    # print(r.shape)


    pairs = {}
    # for i in range(acc_x.shape[1]):
    for subject in range(1,25):
        print('SUBJECT '+str(subject))
        acc_x_subject = acc_x[(acc_x['subject'] == subject)]
        acc_y_subject = acc_y[(acc_y['subject'] == subject)]
        acc_z_subject = acc_z[(acc_z['subject'] == subject)]

        gyro_x_subject = gyro_x[(gyro_x['subject'] == subject)]
        gyro_y_subject = gyro_y[(gyro_y['subject'] == subject)]
        gyro_z_subject = gyro_z[(gyro_z['subject'] == subject)]

        for label in range(1,7):
            acc_x_label = acc_x_subject[(acc_x_subject['label'] == label)]
            acc_y_label = acc_y_subject[(acc_y_subject['label'] == label)]
            acc_z_label = acc_z_subject[(acc_z_subject['label'] == label)]

            gyro_x_label = gyro_x_subject[(gyro_x_subject['label'] == label)]
            gyro_y_label = gyro_y_subject[(gyro_y_subject['label'] == label)]
            gyro_z_label = gyro_z_subject[(gyro_z_subject['label'] == label)]

            for i in range(acc_x_label.shape[0]):
                print(acc_x_label.shape[0])
                o = np.column_stack((acc_x_label.iloc[i, :], acc_y_label.iloc[i, :],
                                     acc_z_label.iloc[i, :], gyro_x_label.iloc[i, :],
                                     gyro_y_label.iloc[i, :], gyro_z_label.iloc[i, :]))
                obs = pd.DataFrame(o)

                obs = obs.drop(index=[128,129])

                subject_label_pair = subject+label
                try:
                    pairs[subject_label_pair] = pairs[subject_label_pair] + 1
                except:
                    pairs[subject_label_pair] = 0

                num = '{:04d}'.format(pairs[subject_label_pair])

                out = "../uci_har/sample/subject"+str(subject)+'_label'+str(label)+'_'+str(num)+'.npy'
                np.save(out, obs)  
              # obs.to_csv(out, index=False, header=False)
            # break


        # obs = pd.DataFrame()
        # obs['acc_x'] = acc_x.iloc[:,i]
        # obs['acc_y'] = acc_y.iloc[:,i]
        # obs['acc_z'] = acc_z.iloc[:,i]
        #
        # obs['gyro_x'] = gyro_x.iloc[:,i]
        # obs['gyro_y'] = gyro_y.iloc[:,i]
        # obs['gyro_z'] = gyro_z.iloc[:,i]
        #
        # # obs['subject'] = list(subjects.iloc[:,i].values) * 128
        # obs['label'] = list(labels.iloc[:,i].values) * 128
        #
        # subject = str(subjects.iloc[:,i].values[0])
        # label = str(labels.iloc[:,i].values[0])
        # subject_label_pair = subject+label
        #
        # try:
        #     pairs[subject_label_pair] = pairs[subject_label_pair] + 1
        # except:
        #     pairs[subject_label_pair] = 0
        #
        # out = "../uci_har/sample/subject"+subject+'_label'+label+'_'+str(pairs[subject_label_pair])+'.csv'
        # obs.to_csv(out, index=False, header=False)
        # print(out)


    # for file in glob.glob("../uci_har/sample/*"):
    #     print(file)
    #     subject = file[25]
    #     label = file[32]
    #     print(subject)
    #     print(label)

    # acc_x = load_file("../uci_har/total/total_acc_x.csv")
    # subjects = load_file("../uci_har/total/subjects.csv")
    # labels = load_file("../uci_har/total/labels.csv")
    #
    # print(acc_x.shape)
    #
    # counter=0
    # for i in range(len(acc_x)-1):
    #     row1 = list(acc_x.iloc[i,64:])
    #     row2 = list(acc_x.iloc[i+1,:64])
    #
    #     difference = []
    #     zip_object = zip(row1, row2)
    #     for list1_i, list2_i in zip_object:
    #         difference.append(list1_i - list2_i)
    #     if len(set(difference)) > 1:
    #         counter += 1
    #         # print(set(difference))
    #         # print(subjects.iloc[i,0])
    #         # print(labels.iloc[i,0])
    #         # print(subjects.iloc[i+1,0])
    #         # print(labels.iloc[i+1,0])
    #
    # print(counter)



if __name__ == '__main__':
    main()
