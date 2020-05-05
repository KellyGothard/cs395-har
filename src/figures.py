# plot all vars for one subject
from numpy import array
from numpy import dstack
from numpy import unique
from pandas import read_csv
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import numpy as np
import glob
import seaborn as sns


# load a single file as a numpy array
def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values


# load a list of files, such as x, y, z data for a given variable
def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = dstack(loaded)
    return loaded


# load a dataset group, such as train or test
def load_dataset(group, prefix=''):
    filepath = prefix + group + '/Inertial Signals/'
    # load all 9 files as a single array
    filenames = list()
    # total acceleration
    filenames += ['total_acc_x_' + group + '.txt', 'total_acc_y_' + group + '.txt', 'total_acc_z_' + group + '.txt']
    # body acceleration
    filenames += ['body_acc_x_' + group + '.txt', 'body_acc_y_' + group + '.txt', 'body_acc_z_' + group + '.txt']
    # body gyroscope
    filenames += ['body_gyro_x_' + group + '.txt', 'body_gyro_y_' + group + '.txt', 'body_gyro_z_' + group + '.txt']
    # load input data
    X = load_group(filenames, filepath)
    # load class output
    y = load_file(prefix + group + '/y_' + group + '.txt')
    return X, y


# get all data for one subject
def data_for_subject(X, y, sub_map, sub_id):
    # get row indexes for the subject id
    ix = [i for i in range(len(sub_map)) if sub_map[i] == sub_id]
    # return the selected samples
    return X[ix, :, :], y[ix]


# convert a series of windows to a 1D list
def to_series(windows):
    series = list()
    for window in windows:
        # remove the overlap from the window
        half = int(len(window) / 2) - 1
        for value in window[-half:]:
            series.append(value)
    return series


# plot the data for one subject
def plot_subject(X, y):
    plt.figure(figsize=(7,8)).subplots_adjust(hspace=0.6)

    rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 11})
    rc('text', usetex=True)
    # determine the total number of plots
    n, off = X.shape[2] + 1 - 4, 0
    # plot total acc
    for i in range(3):
        plt.subplot(n, 2, off + 1)
        plt.plot(to_series(X[:, :, off]), lw = 0.7)
        plt.title('total acc ' + str(i), y=0,  x = 0.01, loc='left')
        off += 1
    # # plot body acc
    # for i in range(3):
    #     plt.subplot(n, 1, off + 1)
    #     plt.plot(to_series(X[:, :, off]))
    #     plt.title('body acc ' + str(i), y=0, loc='left')
    #     off += 1
    # plot body gyro
    for i in range(3):
        plt.subplot(n, 2, off + 1)
        plt.plot(to_series(X[:, :, off]), lw = 0.7)
        plt.title('body gyro ' + str(i), y=0,  x = 0.01, loc='left')
        off += 1
    # plot activities
    # plt.subplot(n, 1, n)
    # plt.plot(y, lw = 0.7)
    # plt.title('activity', y=0, x = 0.01, loc='left')
    plt.savefig('../images/uci_example.png', bbox_inches = 'tight')
    plt.show()

# group data by activity
def data_by_activity(X, y, activities):
    # group windows by activity
    return {a: X[y[:, 0] == a, :, :] for a in activities}


def get_activity_labels():
    labels = {1:'Walking',2:'Walking Upstairs',3:'Walking Downstairs',
              4:'Sitting', 5:'Standing',6:'Laying'}
    return labels

# plot histograms for each activity for a subject
def plot_acc_activity_histograms(X, y):
    # get a list of unique activities for the subject
    activity_ids = unique(y[:, 0])
    # group windows by activity
    grouped = data_by_activity(X, y, activity_ids)
    # plot per activity, histograms for each axis

    fig = plt.figure(figsize=(7,6))
    fig.subplots_adjust(hspace=0.3, top=0.995)

    rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 18})
    rc('text', usetex=True)

    params = {'legend.handlelength': 0.4}
    plt.rcParams.update(params)

    xaxis = None
    for k in range(len(activity_ids)):
        act_id = activity_ids[k]
        # total acceleration
        for i in range(3):
            ax = fig.add_subplot(len(activity_ids), 1, k + 1, sharex=xaxis)
            ax.set_xlim(-1, 1)
            # ax.set_xticks([])
            if k == 0:
                xaxis = ax
            plt.hist(to_series(grouped[act_id][:, :, i]), bins=100, alpha=0.7, label='acc '+str(i))
            plt.title(get_activity_labels()[act_id], x=0.01, y=0.53, loc='left')

    for ax in fig.get_axes():
        ax.label_outer()
    plt.legend(frameon=False, ncol = 3, bbox_to_anchor = (0.83, 8.3), fontsize=18)
    plt.savefig('../images/uci_activity_example.png', bbox_inches = 'tight')
    plt.show()


# plot histograms for each activity for a subject - gyroscope
def plot_gyro_activity_histograms(X, y):
    # get a list of unique activities for the subject
    activity_ids = unique(y[:, 0])
    # group windows by activity
    grouped = data_by_activity(X, y, activity_ids)
    # plot per activity, histograms for each axis

    fig = plt.figure(figsize=(7,6))
    fig.subplots_adjust(hspace=0.3, top=0.995)

    rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 18})
    rc('text', usetex=True)
    params = {'legend.handlelength': 0.4}
    plt.rcParams.update(params)

    xaxis = None
    for k in range(len(activity_ids)):
        act_id = activity_ids[k]
        # total acceleration
        for i in range(3):
            ax = fig.add_subplot(len(activity_ids), 1, k + 1, sharex=xaxis)
            ax.set_xlim(-1, 1)
            # ax.set_xticks([])
            if k == 0:
                xaxis = ax
            plt.hist(to_series(grouped[act_id][:,:,6+i]), bins=100, alpha=0.6, label='gyro '+str(i))
            # plt.title(get_activity_labels()[act_id][:,:,6+i], x=0.01, y=0.55, loc='left')

    for ax in fig.get_axes():
        ax.label_outer()
    plt.legend(frameon=False, ncol = 3, bbox_to_anchor = (0.92, 8.3), fontsize=18)
    plt.savefig('../images/uci_gyro_activity_example.png', bbox_inches = 'tight')
    plt.show()


def loss(df):

    rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 18})
    rc('text', usetex=True)
    params = {'legend.handlelength': 0.4}
    plt.rcParams.update(params)

    plt.plot(df['epoch'], df['accuracy'], label='Accuracy')
    plt.plot(df['epoch'], df['val_accuracy'], label='Val Accuracy')
    plt.legend(loc='best', frameon=False, fontsize=22)
    plt.savefig('../images/uci_tuned_class_acc.png', bbox_inches='tight')
    plt.show()

    plt.plot(df['epoch'], df['loss'], label='Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
    plt.legend(loc='best', frameon=False, fontsize=22)
    plt.savefig('../images/uci_tuned_class_loss.png', bbox_inches='tight')
    plt.show()


# plot histograms for multiple subjects
def plot_subject_histograms(X, y, sub_map, n=10):
    plt.figure()
    # get unique subjects
    subject_ids = unique(sub_map[:,0])
    # enumerate subjects
    xaxis = None
    for k in range(n):
        sub_id = subject_ids[k]
        # get data for one subject
        subX, _ = data_for_subject(X, y, sub_map, sub_id)
        # total acc
        for i in range(3):
            ax = plt.subplot(n, 1, k+1, sharex=xaxis)
            ax.set_xlim(-1,1)
            if k == 0:
                xaxis = ax
            plt.hist(to_series(subX[:,:,i]), bins=100)
    plt.show()

def main():

    # # load data
    # trainX, trainy = load_dataset('train', '../uci_har/')
    #
    # # load mapping of rows to subjects
    # sub_map = load_file('../uci_har/train/subject_train.txt')
    # train_subjects = unique(sub_map)
    #
    # # get the data for one subject
    # sub_id = train_subjects[10]
    # subX, suby = data_for_subject(trainX, trainy, sub_map, sub_id)
    #
    # # plot data for subject
    # plot_subject(subX, suby)

    # plot data for activity
    # plot_acc_activity_histograms(subX, suby)

    # plot_gyro_activity_histograms(subX, suby)

    # # load training history
    # uci_untuned = pd.read_csv('../output/uci_class_20200502-142411.csv')
    # uci_untuned_reg = pd.read_csv('../output/uci_reg_20200504-085716.csv')
    # uci_tuned_reg = pd.read_csv('../output/uci_ft_reg_20200504-093242.csv')
    #
    # uci_tuned = pd.read_csv('../output/uci_ft_20200503-234901.csv')
    #
    # uci_tuned_class = pd.read_csv("../output/uci_ft_class_20200504-085812.csv")
    #
    # print(uci_tuned.head())
    # loss(uci_tuned_class)

    #
    # s = pd.read_csv(subjects[0], header=None)
    # rass = subjects[0].split('.')[-2][-2:]
    # s['rass'] = [rass] * len(s)
    #
    # for subject in subjects[1:]:
    #     print(subject)
    #     s1 = pd.read_csv(subject, header=None)
    #     print(s1.head())
    #     rass = subject.split('.')[-2][-2:]
    #     s1['rass'] = [rass] * len(s1)
    #     s = s.append(s1)
    #     print(len(s))
    #
    # print(s.head())
    # print(s.tail())
    #
    # for rass in s.rass.unique():
    #     s_rass = s[s['rass']==rass]
    #
    #     fig = plt.figure(figsize=(6, 9))
    #     fig.subplots_adjust(hspace=0.5, top=0.995)
    #     rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 12})
    #     rc('text', usetex=True)
    #     params = {'legend.handlelength': 0.4}
    #     plt.rcParams.update(params)
    #
    #     for i in range(6):
    #     # for col in s_rass.columns:
    #
    #         ax = fig.add_subplot(6, 1, i + 1)
    #         ax.hist(s_rass.iloc[:,i], bins=100)
    #     plt.savefig('../images/dists/'+str(rass)+'.png')
    #     # plt.show()

    # cm = np.array([[638,214,26,218,297,6,57,63],
    #     [275,463,29,286,375,57,6,10],
    #     [152,148,361,282,470,81,59,77],
    #     [242,29,57,316,365,89,127,153],
    #     [131,33,106,199,869,76,46,170],
    #     [105,47,61,112,279,610,111,169],
    #     [119,69,52,238,233,179,356,277],
    #     [68,75,53,161,302,89,43,694]])
    #
    # cm = cm / np.sum(cm, axis=1)
    # plt.figure(figsize=(7, 7))
    # sns.heatmap(cm, annot=True, cmap='viridis', cbar=False, square=True)
    # plt.show()
    # plt.savefig('images/normed_ft_cm.png')

if __name__ == '__main__':
    main()
