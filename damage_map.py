# LUCY data missing for plot -> will add when LUCY algorithm output is set

class DamageMap:
    def __init__(self, label, size, damages, sensors=0):
        self.size = size
        self.label = label
        self.damages = damages
        self.sensors = sensors
    def draw(self):
        import matplotlib.pyplot as plt
        plt.xlim([0, self.size[0]])
        plt.ylim([0, self.size[1]])
        plt.title("Damage map, " + self.label)
        plt.xlabel("x [mm]")
        plt.ylabel("y [mm]")
        for key, value in self.damages.items():
            plt.scatter([value[i][0] for i in range(len(value))], [value[i][1] for i in range(len(value))], label=key)
        plt.legend()
        plt.grid()
        plt.gca().set_aspect('equal')
        plt.show()

if __name__=="__main__":
    map = DamageMap("PCLO test 3", [176, 176], {'Delamination' : [[30, 70], [40, 80], [160, 20]], 'Matrix cracking': [[40, 20], [10, 170], [60, 60]]})
    map.draw()