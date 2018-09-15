#!/usr/bin/env python
import click
import math
import matplotlib
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import pdb

class PopulationInfo:
    def __init__(self, line):
        self.total = int(line[3])
        self.total_japanese = int(line[6])
        self.total_foreign = int(line[21])
        self.birth = int(line[22])
        if self.total > 0:
            self.foreign_rate = self.total_foreign / self.total
            self.birth_rate = self.birth / self.total
        else:
            self.foreign_rate = math.nan
            self.birth_rate = math.nan

    def __str__(self):
        return ("total:{:>10} "
                "total foreign: {:>8} "
                "foreign rate: {:.3f} "
                "birth rate: {:.3f} "
                .format(self.total,
                        self.total_foreign,
                        self.foreign_rate,
                        self.birth_rate))

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

def display_chart(coord, tag, colors):
    plt.figure()
    plt.title("Foreigner to Birth rate")
    plt.xlabel("Foreigner Rate")
    plt.ylabel("Birth Rate")
    plt.scatter(*zip(*coord), c=tag, cmap=matplotlib.colors.ListedColormap(colors))
    plt.show()

@click.command()
@click.option('--input','-i',default=None)
def analyze(input):
    data = load_input(input)
    trainNum = 600
    train = data[:trainNum]
    validate = data[trainNum:]
    train_coord = []
    for d in train:
        dp = d.population
        #if dp.birth_rate > 0.2:
        #    print("birth_rate seems too high and considered an anomaly. Skipping: {}".format(dp))
        #    continue
        if dp.foreign_rate is math.nan or dp.birth_rate is math.nan:
            print("birth_rate or foreign_rate is NaN. Skipping: {} {}"
                  .format(d.prefecture, d.municipality))
            continue
        train_coord.append((dp.foreign_rate, dp.birth_rate)) # coordinates
    km = cluster.KMeans(n_clusters=2, random_state=1234)
    cluster_result = km.fit(train_coord)
    train_tag = cluster_result.labels_
    #pdb.set_trace()

    C = 1.0 # SVM regularization parameter
    #model = svm.SVC(kernel='linear', C=C)
    #model.fit(coord, tag)
    display_chart(train_coord, train_tag, ["red","blue"])

if __name__ == "__main__":
    analyze()
