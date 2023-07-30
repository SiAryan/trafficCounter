import sys

def print_command_line_args():
    # sys.argv[0] contains the script name, so we skip it and start from index 1
    arguments = sys.argv[1:]

    # Check if any arguments were provided
    if not arguments:
        print("No command-line arguments provided.")
    else:
        print("Command-line arguments:")
        for index, arg in enumerate(arguments, start=1):
            print("Argument {}: {}".format(index, arg))

if __name__ == "__main__":
    print_command_line_args()