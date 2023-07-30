import os
import glob
import subprocess


def run_command(command):
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        stdout, stderr = process.communicate()
        returncode = process.returncode
        
        if returncode == 0:
            print("Command executed successfully!")
            print("Output:")
            print(stdout)
        else:
            print("Command execution failed.")
            print("Error:")
            print(stderr)
    except Exception as e:
        print("An error occurred:", str(e))

def print_mp4_files(directory_path):
    # Use glob to find all files with .mp4 extension in the specified directory
    mp4_files = glob.glob(os.path.join(directory_path, '*.csv'))

    if not mp4_files:
        print("No .mp4 files found in the directory.")
    else:
        print("MP4 files in the directory:")
        for file_path in mp4_files:
            run_command(["python", "check.py", "%s"%file_path])
            run_command(["ffmpeg", "-i", "%s"%file_path])
            # print(file_path)

if __name__ == "__main__":
    # Replace 'directory_path' with the path of the directory containing .mp4 files
    directory_path = "./RESULTS/BelleroseSchoolEastEntrance/Am_peak_hours/"
    print_mp4_files(directory_path)

