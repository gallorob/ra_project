import matplotlib.pyplot as plt
from typing import Union
import pandas as pd
import json
import os


def plot(data: dict, title: str, labels: Union[str, list] = None, savedst: str = None) -> None:
    colors = ['tab:green', 'tab:red', 'tab:blue', 'tab:orange']
    data.plot(color=colors, linewidth=1, figsize=(12,6))
    plt.legend(labels=labels, fontsize=14)
    # modify ticks size
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # title and labels
    plt.title(title, fontsize=20)
    plt.xlabel('Episode', fontsize=16)
    plt.ylabel('Reward', fontsize=16)
    if savedst is not None:
        title = title.replace(" ", "").replace(":", "_")
        plt.savefig(f'{savedst}{title}.png', transparent=True)
    plt.show()


def plot_all(plot1: list, plot2: list, title: str, window_size: int, dst: str) -> None:
    # All rewards
    merged = pd.DataFrame.from_dict({'Without Reward Shaping': plot1, 'With Reward Shaping': plot2})
    # SMA Without Reward Shaping
    merged[f'Without Reward Shaping SMA {window_size}'] = merged['Without Reward Shaping'].rolling(window_size, min_periods=1).mean()
    # SMA With Reward Shaping
    merged[f'With Reward Shaping SMA {window_size}'] = merged['With Reward Shaping'].rolling(window_size, min_periods=1).mean()
    # Plot all
    plot(merged[['Without Reward Shaping', f'Without Reward Shaping SMA {window_size}']], f'{title} Without reward shaping', ['Reward', f'{window_size}-episodes SMA'], savedst=dst)
    plot(merged[['With Reward Shaping', f'With Reward Shaping SMA {window_size}']], f'{title} With reward shaping', ['Reward', f'{window_size}-episodes SMA'], savedst=dst)
    plot(merged[[f'Without Reward Shaping SMA {window_size}', f'With Reward Shaping SMA {window_size}']], f'{title} {window_size}-Episodes Moving Average Rewards', ['Without reward shaping', 'With reward shaping'], savedst=dst)


def main(hist1_path: str, hist2_path: str, dst: str, train_window_size: int, test_window_size: int) -> None:
    # Open both histories with and without reward shaping
    with open(hist1_path, 'r') as h1, open(hist2_path, 'r') as h2:
        no_rs_hist = json.load(h1)
        rs_hist = json.load(h2)
        h1.close()
        h2.close()
    # Handle plots save path
    if dst is not None:
        dst = dst + 'plots/'
        if not os.path.isdir(dst):
            os.makedirs(dst)
    # Plot
    plot_all(no_rs_hist['train_rewards'], rs_hist['train_rewards'], 'Training:', train_window_size, dst)
    plot_all(no_rs_hist['test_rewards'], rs_hist['test_rewards'], 'Test:', test_window_size, dst)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Taxi v3 history plotter')
    parser.add_argument('--p1', type=str, dest='hist1_path',
                        help='Path to history .json without reward shaping')
    parser.add_argument('--p2', type=str, dest='hist2_path',
                        help='Path to history .json with reward shaping')
    parser.add_argument('--dst', type=str, dest='dst', default=None,
                        help='Optional; Path to save plots')
    parser.add_argument('--w1', type=int, dest='train_window_size', default=10,
                        help='Optional; Window size to compute the moving average for training')
    parser.add_argument('--w2', type=int, dest='test_window_size', default=5,
                        help='Optional; Window size to compute the moving average for test')
    args = parser.parse_args()

    main(hist1_path=args.hist1_path,
         hist2_path=args.hist2_path,
         dst=args.dst,
         train_window_size=args.train_window_size,
         test_window_size=args.test_window_size)
