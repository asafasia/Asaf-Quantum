import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class Plotter:
    meta_data: dict = None
    data: dict = None

    def plot(self, *args, **kwargs):
        if self.meta_data['type'] == '1d':
            self._plot_1d()
        elif self.meta_data['type'] == '2d':
            self._plot_2d()
        elif self.meta_data['type'] == 'IQ':
            self._plot_IQ(*args, **kwargs)
        else:
            print('no type in meta_data')

    def _plot_1d(self):
        plt.figure()
        plt.xlabel(self.meta_data['plot_properties']['x_label'])
        plt.ylabel(self.meta_data['plot_properties']['y_label'])
        plt.title(self.meta_data['plot_properties']['title'])
        plt.plot(self.data['x_data'], self.data['y_data'])

    def _plot_2d(self):
        pass

    def _plot_IQ(self, *args, **kwargs):
        plt.figure()
        if kwargs.get('type') == 'clouds':
            plt.plot(self.data['I_g'], self.data['Q_g'], '.', label='ground')
            plt.plot(self.data['I_e'], self.data['Q_e'], '.', label='excited')

            plt.title(self.meta_data['plot_properties']['title'])
            plt.xlabel(self.meta_data['plot_properties']['x_label'])
            plt.ylabel(self.meta_data['plot_properties']['y_label'])
            plt.legend()

        elif kwargs.get('type') == 'histogram':
            plt.hist(self.data['I_g'], bins=80, edgecolor='black', label='ground state', alpha=0.6)
            plt.hist(self.data['I_e'], bins=80, edgecolor='black', label='excited state', alpha=0.5)

            plt.xlabel('I [a.u.]')
            plt.ylabel('number')
            plt.title('Separation Histogram')
        else:
            print('no type in kwargs')
