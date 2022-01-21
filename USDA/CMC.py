import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

default_color = ['r', 'g', 'b', 'c', 'm']
default_marker = ['*', 'o', 's', 'v', 'X']


class CMC:
    def __init__(self, cmc_dict, color=default_color,marker=default_marker):
        self.color = color
        self.marker = marker
        self.cmc_dict = cmc_dict

    def plot(self, title, rank=20, xlabel='Rank', ylabel='Matching Rates (%)', show_grid=True):
        fig, ax = plt.subplots()
        fig.suptitle(title)
        x = list(range(0, rank + 1, 5))
        plt.ylim(0, 1.0)
        plt.xlim(1, rank)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(x)
        plt.grid(show_grid)

        method_name = []
        i = 0
        for name in self.cmc_dict.keys():
            if rank < len(self.cmc_dict[name]):
                temp_cmc = self.cmc_dict[name][:rank]
                r = list(range(1, rank + 1))
            else:
                temp_cmc = self.cmc_dict[name]
                r = list(range(1, len(temp_cmc) + 1))

            if name == list(self.cmc_dict.keys())[-1]:
                globals()[name] = mlines.Line2D(r, temp_cmc, color=self.color[0], marker=self.marker[0],
                                                label='{:.1f}% {}'.format(self.cmc_dict[name][0] * 100, name))
            else:
                globals()[name] = mlines.Line2D(r, temp_cmc, color=self.color[i + 1], marker=self.marker[i + 1],
                                                label='{:.1f}% {}'.format(self.cmc_dict[name][0] * 100, name))
                i = i + 1
            ax.add_line(globals()[name])
            method_name.append(globals()[name])

        plt.legend(handles=method_name)
        plt.show()

    def save(self, title, filename,
             rank=20, xlabel='Rank',
             ylabel='Matching Rates (%)', show_grid=True,
             save_path=os.getcwd(), format='png', **kwargs):
        fig, ax = plt.subplots()
        fig.suptitle(title)
        x = list(range(0, rank + 1, 5))
        plt.ylim(0, 1.0)
        plt.xlim(1, rank)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(x)
        plt.grid(show_grid)

        method_name = []
        i = 0
        for name in self.cmc_dict.keys():
            if rank < len(self.cmc_dict[name]):
                temp_cmc = self.cmc_dict[name][:rank]
                r = list(range(1, rank + 1))
            else:
                temp_cmc = self.cmc_dict[name]
                r = list(range(1, len(temp_cmc) + 1))

            if name == list(self.cmc_dict.keys())[-1]:
                globals()[name] = mlines.Line2D(r, temp_cmc, color='r', marker='*',
                                                label='{:.1f}% {}'.format(self.cmc_dict[name][0] * 100, name))
            else:
                globals()[name] = mlines.Line2D(r, temp_cmc, color=self.color[i], marker=self.marker[i],
                                                label='{:.1f}% {}'.format(self.cmc_dict[name][0] * 100, name))
                i = i + 1
            ax.add_line(globals()[name])
            method_name.append(globals()[name])

        plt.legend(handles=method_name)
        fig.savefig(os.path.join(save_path, filename + '.' + format),
                    format=format,
                    bbox_inches='tight',
                    pad_inches=0, **kwargs)


cmc_dict ={
    'PRAI-1581': [0.649, 0.75, 0.81, 0.82, 0.84, 0.86, 0.87, 0.91, 0.93, 0.94, 0.94, 0.94, 0.95, 0.95, 0.95, 0.96, 0.96, 0.96, 0.97, 0.97],
    'VeRi-776': [0.81, 0.86, 0.91, 0.94, 0.96, 0.963, 0.972, 0.978, 0.983, 0.99, 0.99, 0.99,  0.99,  0.99, 0.99,  0.99, 0.99,  0.99,   0.99,   0.99],
    'VRAI': [0.833, 0.87, 0.89, 0.91, 0.92, 0.943, 0.956, 0.969, 0.979, 0.987, 0.987, 0.987,  0.987,  0.987, 0.987,  0.987, 0.987,  0.987,   0.988,   0.988],
    'DUKE': [0.896, 0.898, 0.92, 0.95, 0.96, 0.963, 0.976, 0.979, 0.986, 0.987, 0.988, 0.988,  0.988,  0.988, 0.988,  0.988, 0.988,  0.988,   0.99,   0.99],
    'Market-1501': [0.947, 0.97, 0.972, 0.975, 0.976, 0.98, 0.982, 0.989, 0.99, 0.99, 0.99, 0.99,  0.99,  0.99, 0.99,  0.99, 0.99,  0.99,   0.99,   0.99],
}
from CMC import CMC
cmc = CMC(cmc_dict)

#custimised color and marker
new_color = ['r','g','b','c','m']
new_marker = ['*','o','s','v','X']
cmc = CMC(cmc_dict,color=new_color,marker=new_marker)
cmc.plot(title = 'CMC curve')
cmc.save(title = 'CMC curve', filename='cmc_result_1')