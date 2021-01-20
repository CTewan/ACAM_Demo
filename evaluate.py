import os
import glob

import pandas as pd

PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))
CLF_OUTPUT_PATH = os.path.join(PROJECT_FOLDER, "clf_output")
CLF_OUTPUT_PATTERN = os.path.join(CLF_OUTPUT_PATH, "*.csv")

clf_files = glob.glob(CLF_OUTPUT_PATTERN)

def _evaluate_tracker(csv_path):
	clf_output = pd.read_csv(csv_path)
	time = clf_output.loc[:, "Time"].astype(int).tolist()
	duration = max(time) # rough estimation of max_duration. Alternative is hardcode

	time = set(time)
	predicted_time = len(time)

	percent_tracked = round((float(predicted_time) / float(duration)) * 100, 1)

	return percent_tracked

def evaluate_tracker(clf_files):
	results = {}

	for clf_file in clf_files:
		#filename = clf_file.split(r"\\")[-1].split(".")[0]
		filename = os.path.basename(clf_file).split(".")[0]

		percent_tracked = _evaluate_tracker(csv_path=clf_file)
		results[filename] = percent_tracked

	return results

if __name__ == "__main__":
	print(evaluate_tracker(clf_files=clf_files))