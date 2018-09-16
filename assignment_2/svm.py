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

def display_chart(coord, tag, colors):
    plt.figure(figsize=(9,6))
    plt.title("Foreigner to Birth rate")
    plt.xlabel("Foreigner Rate")
    plt.ylabel("Birth Rate")
    plt.scatter(*zip(*coord), c=tag, cmap=matplotlib.colors.ListedColormap(colors))
    #plt.ylim(ymin=0)
    plt.show()

@click.command()
@click.option('--input','-i',default=None)
def analyze(input):
    data = load_input(input)
    trainNum = 600
    train = data[:trainNum]
    validate = data[trainNum:]
    train_coord = []
    train_tag = []
    for d in train:
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
        if dp.foreign_rate > 0.1:
            print("foreign_rate is very high {}: {} {}"
                  .format(dp.foreign_rate, d.prefecture, d.municipality))
            continue
        train_coord.append((dp.foreign_rate, dp.birth_rate)) # coordinates
        train_tag.append(0 if dp.growth_rate < 1.0 else 1)
    #km = cluster.KMeans(n_clusters=2, random_state=1234)
    #cluster_result = km.fit(train_coord)
    #train_tag = cluster_result.labels_
    #pdb.set_trace()

    C = 1.0 # SVM regularization parameter
    #model = svm.SVC(kernel='linear', C=C)
    #model.fit(coord, tag)
    display_chart(train_coord, train_tag, ["blue","red"])

if __name__ == "__main__":
    analyze()
