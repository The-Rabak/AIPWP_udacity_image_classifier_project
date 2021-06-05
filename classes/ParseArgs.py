import argparse
from classes.CommandArgs import CommandArgs

class ParseArgs():
    '''
    args dict should be an array of CommandArgs objects
    '''

    def __init__(self, args=[]) -> None:
        self.args = args
        self.parser = argparse.ArgumentParser()

    def set_args(self):
        for arg in self.args:
            if not isinstance(arg, CommandArgs):
                continue
            #the default first argument for both the predict and train files
            #As requested by the venerable Udacity team
            if len(arg.flags) == 0:
                self.parser.add_argument("images_dir",
                                         help=arg.get_help(),
                                         default=arg.get_default(),
                                         nargs='?', const=''
                                         )
                continue

            if len(arg.choices) > 0:
                self.parser.add_argument(*arg.flags,
                                    choices=arg.choices,
                                    help=arg.get_help(),
                                    default=arg.get_default(),
                                    nargs='?', const=''
                                    )
            else:
                self.parser.add_argument(*arg.flags,
                                    help=arg.get_help(),
                                    default=arg.get_default(),
                                    nargs='?', const=''
                                    )

    def get_args(self):
        return self.parser.parse_args()

    def get_parser_help(self):
        return self.parser.print_help()