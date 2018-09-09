#!/usr/bin/env python
import click
import math
import matplotlib.pyplot as plt

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

@click.command()
@click.option('--input','-i',default=None)
def analyze(input):
    data = load_input(input)
    x = []
    y = []
    for d in data:
        x.append(d.population.foreign_rate)
        y.append(d.population.birth_rate)
    plt.figure()
    plt.title("Foreigner to Birth rate")
    plt.xlabel("Foreigner Rate")
    plt.ylabel("Birth Rate")
    plt.scatter(x, y)
    plt.show()

if __name__ == "__main__":
    analyze()
