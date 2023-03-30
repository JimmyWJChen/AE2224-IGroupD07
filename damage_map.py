# LUCY data missing for plot -> will add when LUCY algorithm output is set
import matplotlib.pyplot as plt

class DamageMap:
    def __init__(self, label, size, damages, sensors=0):
        self.size = size
        self.label = label
        self.damages = damages
        self.sensors = sensors
    def draw(self):
        self.map = plt.figure()
        self.ax = self.map.add_subplot(111)
        self.map.canvas.mpl_connect('button_press_event', self.press)
        self.ax.axis(xmin=0, xmax=self.size[0], ymin=0, ymax=self.size[1])
        self.ax.title.set_text("Damage map, " + self.label)
        self.ax.set_xlabel("x [mm]")
        self.ax.set_ylabel("y [mm]")
        for key, value in self.damages.items():
            self.ax.scatter([value[i][0] for i in range(len(value))], [value[i][1] for i in range(len(value))], label=key)
        self.ax.legend()
        self.ax.grid()
        self.ax.set_aspect('equal')
        plt.show()
    def press(self, event):
        x = event.xdata
        y = event.ydata
        for key, value in self.damages.items():
            for i in range(len(value)):
                if ((value[i][0]-x)**2 + (value[i][1]-y)**2) < 4:
                    try:
                        self.lucy([value[i][0], value[i][1]], key, value[i][2])
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
        plt.show()
    
    def lucypress(self, event):
        plt.close(self.lucymap)

        


if __name__=="__main__":
    map = DamageMap("PCLO test 3", [176, 176], {'Delamination' : [[30, 70, 8], [40, 80, 20], [160, 20, 12]], 'Matrix cracking': [[40, 20], [10, 170], [60, 60]]})
    map.draw()