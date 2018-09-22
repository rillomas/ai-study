#!/usr/bin/env python
import click
import math
import matplotlib
import matplotlib.pyplot as plt
from sklearn import svm, cluster
from sklearn.grid_search import GridSearchCV
import sklearn.metrics as skm
import random
import numpy as np
import pdb

# We multiply this scale value because
# the input value seems to be too small?
SCALE = 1.0

class PopulationInfo:
    def __init__(self, line):
        self.total = int(line[3])
        self.total_japanese = int(line[6])
        self.total_foreign = int(line[21])
        self.birth = int(line[22])
        self.death = int(line[23])
        self.moved_in = int(line[24])
        self.moved_out = int(line[25])
        if self.total > 0:
            self.foreign_rate = self.total_foreign / self.total
            self.birth_rate = self.birth / self.total
        else:
            self.foreign_rate = math.nan
            self.birth_rate = math.nan
        growth = self.birth + self.moved_in
        decrease = self.death + self.moved_out
        if growth > 0:
            self.growth_rate = growth / decrease
        else:
            self.growth_rate = math.nan

    def __str__(self):
        return ("total:{:>10} "
                "total foreign: {:>8} "
                "foreign rate: {:.3f} "
                "birth rate: {:.3f} "
                "growth rate: {:.3f} "
                .format(self.total,
                        self.total_foreign,
                        self.foreign_rate,
                        self.birth_rate,
                        self.growth_rate))

class LocationInfo:
    def __init__(self, line):
        self.prefecture = line[1]
        self.municipality = line[2]
        self.population = PopulationInfo(line)

    def __str__(self):
        return "{:<6} {:<8} population: {}".format(self.prefecture,
                                                   self.municipality,
                                                   self.population)

def load_input(input):
    info_list = []
    with open(input) as f:
        for _ in range(2):
            # skip first two lines
            next(f)

        for l in f:
            cols = l.split(",")
            inf = LocationInfo(cols)
            info_list.append(inf)
    return info_list

def plot_chart(coord, tag, colors):
    plt.figure(figsize=(9,6))
    plt.title("Foreigner to Birth rate")
    plt.xlabel("Foreigner Rate")
    plt.ylabel("Birth Rate")
    plt.scatter(*zip(*coord),
                c=tag,
                cmap=matplotlib.colors.ListedColormap(colors),
                edgecolors='k')
    #plt.ylim(ymin=0)

def plot_contours(model, coord, tag, colors):
    x, y = zip(*coord)
    step = 0.02
    offset = 0.02
    x_min = np.min(x) - offset
    x_max = np.max(x) + offset
    y_min = np.min(y) - offset
    y_max = np.max(y) + offset
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))
    points = np.c_[xx.ravel(), yy.ravel()]
    z = model.predict(points)
    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap=matplotlib.colors.ListedColormap(colors))


def extract_coord_and_tags(data):
    coord = []
    tag = []
    for d in data:
        dp = d.population
        if (dp.foreign_rate is math.nan or
            dp.birth_rate is math.nan or
            dp.growth_rate is math.nan):
            print("birth_rate, foreign_rate, or growth_rate is NaN. Skipping: {} {}"
                  .format(d.prefecture, d.municipality))
            continue
        if dp.birth_rate > 0.05:
            print("birth_rate is very high {}: {} {}"
                  .format(dp.birth_rate, d.prefecture, d.municipality))
            continue
        #if dp.foreign_rate > 0.1:
        #    print("foreign_rate is very high {}: {} {}"
        #          .format(dp.foreign_rate, d.prefecture, d.municipality))
        #    continue
        coord.append((dp.foreign_rate*SCALE, dp.birth_rate*SCALE)) # coordinates
        tag.append(0 if dp.growth_rate < 1.0 else 1)
    return coord, tag

@click.command()
@click.option('--input','-i',default=None)
def analyze(input):
    data = load_input(input)
    seed = 12345
    random.seed(seed)
    random.shuffle(data)
    trainNum = 600
    train = data[:trainNum]
    validate = data[trainNum:]
    print("whole: ", len(data))
    print("training: ", len(train))
    print("validating: ", len(validate))
    train_coord, train_tag = extract_coord_and_tags(train)
    validate_coord, validate_tag = extract_coord_and_tags(validate)

    #plot_chart(train_coord, train_tag, ["blue","red"])
    #param = [
    #    {
    #        'C': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    #        'kernel': ['rbf'],
    #        'gamma': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    #    }
    #]
    #clf = GridSearchCV(svm.SVC(), param)
    #clf.fit(train_coord, train_tag)
    #model = clf.best_estimator_
    #print(model)
    C = 1.0 # SVM regularization parameter
    model = svm.SVC(kernel='rbf',
                    C=C,
                    class_weight='balanced',
                    random_state=seed)
    model.fit(train_coord, train_tag)
    predict_result = model.predict(validate_coord)
    mat = skm.confusion_matrix(validate_tag, predict_result)
    accuracy = skm.accuracy_score(validate_tag, predict_result)
    precision = skm.precision_score(validate_tag, predict_result)
    recall = skm.recall_score(validate_tag, predict_result)
    f1 = skm.f1_score(validate_tag, predict_result)
    print("confusion matrix: ", mat)
    print("accuary: ", accuracy)
    print("precision: ", precision)
    print("recall: ", recall)
    print("F1: ", f1)

    colors = ["blue","red"]
    plot_contours(model, validate_coord, validate_tag, colors)
    plot_chart(validate_coord, validate_tag, colors)
    plt.show()
    ##w = model.coef_[0]
    ##i = model.intercept_
    ##a = -w[0]/w[1]
    ##b = i[0]/w[1]
    ##print("y = {}x - {}".format(a, b))
    ##start = (0, -b)
    ##val = 0.15
    ##ev = val*a - b
    ##end = (val, ev)
    ##print(start, end)

if __name__ == "__main__":
    analyze()
