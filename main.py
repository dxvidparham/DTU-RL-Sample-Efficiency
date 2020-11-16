import argparse

parser = argparse.ArgumentParser(description="Run the SAC-RL Agent.")

# Add Arguments depending on the hyperparameter variables.
parser.add_argument('-n',
                    type=int,
                    nargs='+',
                    help='an integer for the accumulator')

args = parser.parse_args()
print(args)


# TODO Forward to the RL Agent
