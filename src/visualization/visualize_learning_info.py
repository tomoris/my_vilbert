import argparse

import numpy as np
import matplotlib as mpl

mpl.use("Agg")  # AGG(Anti-Grain Geometry engine)
import matplotlib.pyplot as plt

# loss_info_list = [
#     "loss",
#     "masked_lm_loss",
#     "next_sent_loss",
#     "non_text_loss",
#     "corresponding_loss",
# ]

loss_info_list = [
    "loss",
]

total_loss_info_list = [
    "total_loss",
    "total_masked_lm_loss",
    "total_next_sent_loss",
    "total_non_text_loss",
    "total_corresponding_loss",
]

accu_info_list = [
    "accu_mask_token",
    "accu_next_sent",
    "accu_corresponding",
]


def main():
    parser = argparse.ArgumentParser(description="my_vilbert")
    parser.add_argument("--log_file", type=str, default="results/log/log.txt")
    parser.add_argument("--save_file", type=str, default="results/figures/figure.png")
    args = parser.parse_args()

    loss_info_array = [[] for _ in loss_info_list]
    total_loss_info_array = [[] for _ in total_loss_info_list]
    accu_info_array = [[] for _ in accu_info_list]
    for line in open(args.log_file, "r"):
        line = line.rstrip()
        for i, learning_info in enumerate(loss_info_list):
            if (
                line.find(learning_info + ":") >= 0
                and line.find("total_" + learning_info + ":") == -1
            ):
                num = float(line.split(learning_info + ":")[1])
                loss_info_array[i].append(num)
        for i, learning_info in enumerate(total_loss_info_list):
            if line.find(learning_info + ":") >= 0:
                num = float(line.split(learning_info + ":")[1])
                epoch = int(line.split(" \t ")[1].split(" ")[0].split("epoch:")[1])
                total_loss_info_array[i].append(num)
                assert epoch + 1 == len(total_loss_info_array[i])
        for i, learning_info in enumerate(accu_info_list):
            if line.find(learning_info + ":") >= 0:
                num = float(line.split(learning_info + ":")[1])
                epoch = int(line.split(" \t ")[1].split(" ")[0].split("epoch:")[1])
                accu_info_array[i].append(num)
                assert epoch + 1 == len(accu_info_array[i])

    loss_info_array = np.array(loss_info_array).T
    total_loss_info_array = np.array(total_loss_info_array).T
    accu_info_array = np.array(accu_info_array).T

    fig, (axL, axC, axR) = plt.subplots(ncols=3)

    print(
        np.sort(loss_info_array.T)[0][-10:],
        loss_info_array.max(),
        loss_info_array.argmax(),
    )
    axL.plot(loss_info_array)
    axL.legend(loss_info_list)
    axL.set_ylim(
        loss_info_array.min(), loss_info_array[0][0] + loss_info_array[0][0] * 0.1,
    )

    axC.plot(total_loss_info_array)
    axC.legend(total_loss_info_list)

    axR.plot(accu_info_array)
    axR.legend(accu_info_list)

    fig.savefig(args.save_file)


if __name__ == "__main__":
    main()
