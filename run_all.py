import glob
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--video_folder', type=str, required=True, default="")
    parser.add_argument('-d', '--display', type=str, required=False, default="True")

    args = parser.parse_args()

    video_file_pattern = os.path.join(args.video_folder, ".mp4")
    video_files = glob.glob(video_file_pattern)

    if len(video_files) == 0:
        print("No videos found.")

        return

    for video_file in enumerate(video_files):
        os.system("python detect_actions.py --display {} --video_path {}".format(args.display, video_file))

if __name__ == "__main__":
    main()