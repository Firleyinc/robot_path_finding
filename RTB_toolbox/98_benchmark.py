import time
import subprocess
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", required=True, help="path to script beeing benchmarked")
    parser.add_argument("-i", type=int, help="optional: number of measurment runs")
    args = parser.parse_args()

    path = args.f
    iters = 1 if args.i is None else args.i
    times = []
    for i in range(0, iters):
        start = time.perf_counter()
        subprocess.run(["python", path])
        end = time.perf_counter()
        times.append(end-start)

    if iters == 1:
        print(f"Execution time: {times[0]:.3f} sec")
    else:
        avg = np.mean(np.array(times))
        med = np.median(np.array(times))
        run = 1
        for tim in times:
            print(f"Measurment run {run} time: {tim:.3f} sec")
            run += 1

        print(f"----\nAverage execution time: {avg:.4f} sec, median: {med:.4f} sec")

if __name__ == '__main__':
    main()
