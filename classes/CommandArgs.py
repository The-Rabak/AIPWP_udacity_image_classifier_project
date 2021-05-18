class CommandArgs():
    def __init__(self, flags, type, choices = {}, help = '', default = 0):
        self.flags = flags
        self.type = type
        self.choices = choices
        self.help = help
        self.default = default

    def get_flags(self):
        return self.flags
    def get_type(self):
        return self.type
    def get_choices(self):
        return self.choices
    def get_help(self):
        return self.help
    def get_default(self):
        return self.default