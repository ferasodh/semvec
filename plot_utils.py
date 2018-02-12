import matplotlib.pyplot as plt
import numpy
from collections import OrderedDict


def plot_fold_results(y, title, ylabel,outfile_path ):
    fig1 = plt.figure(figsize=(8, 6), dpi=120)

    N = len(y)
    x = range(N)

    rect=plt.bar(x, y, width=0.5, color="blue",label=ylabel)

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Fold')

    plt.legend(loc='upper left', prop={'size': 7})

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2., 1.01 * height,
                     '%d' % int(height),
                     ha='center', va='bottom')

    autolabel(rect)

    fig1.savefig(outfile_path)
    plt.close()

def plot(x_data,y_data_list, title, xlabel, ylabel,epochs_num,outfile_path, lw_lst,alpha_lst,mean_lst,ylim=[-0.5, 101] ):
    fig1 = plt.figure(figsize=(8, 6), dpi=120)
    for i in range(len(y_data_list)):
        plt.plot(x_data, y_data_list[i], lw_lst[i], alpha_lst[i],
                 label='Fold %d (%s = %0.2f%%)' % (i,ylabel, mean_lst[i]))#

    mean_list = numpy.mean(y_data_list, axis=0)
    lw_lst.append(2)
    alpha_lst.append(0.8)
    std = numpy.std(mean_list)

    plt.plot(x_data, mean_list, color='b',
             label='Mean (%s = %0.2f%%)' % (ylabel,numpy.mean(mean_list)),
             lw=2, alpha=.8)

    upper = numpy.minimum(mean_list + std, 100)
    lower = numpy.maximum(mean_list - std, 0)

    plt.fill_between(x_data, lower, upper, color='grey', alpha=.2,
                     label=r'$\pm$ %0.2f std. dev.'%(std))

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xlim([-0.5, epochs_num + 1])
    plt.ylim(ylim)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),loc='lower right', prop={'size': 7})

    fig1.savefig(outfile_path)
    plt.close()