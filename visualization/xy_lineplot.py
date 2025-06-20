import matplotlib.pyplot as plt

def basic_lineplot(x, y, title="Line Plot", xlabel="X-axis", ylabel="Y-axis"):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker='o', linestyle='-', color='blue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    


def multi_lineplot(data_dict, title="Line Plot", xlabel="X-axis", ylabel="Y-axis"):
    plt.figure(figsize=(8, 5))
    plt.title(title)

    for label, (x, y, texts) in data_dict.items():
        plt.plot(x, y, marker='o', linestyle='-', label=label)
        for xi, yi, txt in zip(x, y, texts):
            plt.text(xi, yi, str(txt), fontsize=9, ha='right', va='bottom')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
