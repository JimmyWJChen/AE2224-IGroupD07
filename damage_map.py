import matplotlib.pyplot as plt

# LUCY data missing for plot -> will add when LUCY algorithm output is set

def drawDamageMap(label, size, damages, sensor=0):
    # plt.figure(figsize=(6,6))
    plt.xlim([0, size[0]])
    plt.ylim([0, size[1]])
    plt.title("Damage map, " + label)
    plt.xlabel("x [mm]")
    plt.ylabel("y [mm]")
    for key, value in damages.items():
        plt.scatter([value[i][0] for i in range(len(value))], [value[i][1] for i in range(len(value))], label=key)
    plt.legend()
    plt.grid()
    plt.gca().set_aspect('equal')
    plt.show()

if __name__=="__main__":
    drawDamageMap("PCLO test 3", [176, 176], {'Delamination' : [[30, 70], [40, 80], [160, 20]], 'Matrix cracking': [[40, 20], [10, 170], [60, 60]]})