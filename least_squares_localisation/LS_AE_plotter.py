import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vallenae as vae
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Plotter:
    def __init__(self, label: str, sensors=0):

        self.size = [0.140, 0.140]
        self.label = label

        self.sensors = sensors

    def read_csv(self):
        path = "source_locations_backup" + self.label + ".csv"
        df = pd.read_csv(path)
        # convert to numpy array
        data = df.iloc[:, 1::].to_numpy()
        self.cluster_1 = data[np.where(data[:, -1] == 0)]
        self.cluster_2 = data[np.where(data[:, -1] == 1)]
        self.cluster_3 = data[np.where(data[:, -1] == 2)]
        return data



    def draw(self, clustering=True):
        self.map = plt.figure()
        self.ax = self.map.add_subplot(111)
        self.map.canvas.mpl_connect('button_press_event', self.press)
        self.ax.axis(xmin=0, xmax=self.size[0], ymin=0, ymax=self.size[1])
        #self.ax.title.set_text("Damage map, " + self.label)
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.damages = self.read_csv()
        self.X_coordinates = self.damages[:, 0]
        self.Y_coordinates = self.damages[:, 1]
        self.cluster_labels = self.damages[:, -1]
        print(self.X_coordinates)
        print(self.Y_coordinates)
        print(np.shape(self.X_coordinates))
        self.X_1 = self.cluster_1[:, 0]
        self.Y_1 = self.cluster_1[:, 1]
        self.X_2 = self.cluster_2[:, 0]
        self.Y_2 = self.cluster_2[:, 1]
        self.X_3 = self.cluster_3[:, 0]
        self.Y_3 = self.cluster_3[:, 0]
        if clustering:
            self.ax.scatter(self.X_1, self.Y_1, label='cluster 0', marker='.')
            self.ax.scatter(self.X_2, self.Y_2, label='cluster 1', marker='.')
            self.ax.scatter(self.X_3, self.Y_3, label='cluster 2', marker='.')
        else:
            self.ax.scatter(self.X_coordinates, self.Y_coordinates, marker='.')
        """
        for i in range(len(self.X_coordinates)):
            self.ax.scatter(self.X_coordinates[i], self.Y_coordinates[i], label=str(self.cluster_labels[i]), marker='.')
        """
        self.ax.legend()
        self.ax.grid()
        self.ax.set_aspect('equal')
        fig = plt.gcf()
        name_eps = 'AE_source_plot_' + self.label + '.eps'
        name_pdf = 'AE_source_plot_' + self.label + '.pdf'
        fig.savefig(name_eps)
        fig.savefig(name_pdf)
        plt.show()


    def press(self, event):
        x = event.xdata
        y = event.ydata

        for i in range(len(self.damages)):
            if ((self.X_coordinates[i] - x) ** 2 + (self.Y_coordinates[i] - y) ** 2) < 0.004:
                try:
                    self.lucy([self.X_coordinates[i], self.Y_coordinates[i]], self.label, self.damages[i, 2])
                    return True
                except:
                    print("No LUCY given!")

    def lucy(self, point, label, uncertainty):
        self.lucymap = plt.figure()
        self.lucyax = self.lucymap.add_subplot(111)
        self.lucymap.canvas.mpl_connect('button_press_event', self.lucypress)
        self.lucyax.axis(xmin=0, xmax=self.size[0], ymin=0, ymax=self.size[1])
        self.lucyax.title.set_text(f"LUCY for point [{point[0]}, {point[1]}]")
        self.lucyax.set_xlabel("x [mm]")
        self.lucyax.set_ylabel("y [mm]")
        self.lucyax.scatter(point[0], point[1], label=label)
        circle = plt.Circle(point, uncertainty, edgecolor='r')
        self.lucyax.add_patch(circle)
        self.lucyax.legend()
        self.lucyax.grid()
        self.lucyax.set_aspect('equal')
        fig = plt.gcf()
        name_eps = f'LU_for_point_[{point[0]}, {point[1]}]_' + self.label + '.eps'
        name_pdf = f'LU_for_point_[{point[0]}, {point[1]}]_' + self.label + '.pdf'
        fig.savefig(name_eps)
        fig.savefig(name_pdf)
        plt.show()

    def lucypress(self, event):
        plt.close(self.lucymap)


if __name__ == "__main__":
    map = Plotter("PD_PCLSR_QI090LU5")
    map.draw()