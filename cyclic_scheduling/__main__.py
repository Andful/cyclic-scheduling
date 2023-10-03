import yaml
import argparse
from . import CyclicSchedulingProblem

parser = argparse.ArgumentParser(
                    prog='cyclic-scheduling',
                    description='Cyclic Scheduling problem definition and solver')

command = parser.add_subparsers(dest='command', required=True)

display = command.add_parser('display')
solve = command.add_parser('solve')
solve_relaxation = command.add_parser('solve-relaxation')

display_commands = display.add_subparsers(dest='subcommand', required=True)
display_commands.add_parser('system-of-inequalities')
display_commands.add_parser('mip')
display_commands.add_parser('matrix')


parser.add_argument('problem-file', type=str, help='problem yaml file of the problem definition', nargs=1)

def main():
    args = parser.parse_args()
    [problem_file] =  getattr(args, 'problem-file')
    data = CyclicSchedulingProblem.from_file(problem_file)
    if args.command == "display":
        if args.subcommand == "matrix":
            print(data.matrix())
        elif args.subcommand == "mip":
            print(data.mip_formulation())
        elif args.subcommand == "system-of-inequalities":
            print(data.systems_of_inequalities())
    elif args.command == "solve":
        data.solve()

if __name__ == "__main__":
    main()

