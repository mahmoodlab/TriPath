import os
import subprocess
import sys

class Conditional():
    def __init__(self):
        self.success = True
        self.err_msgs = []
    def check(self, statement, err_msg=None):
        if not statement:
            self.success = False
        if err_msg:
            self.err_msgs.append(err_msg)
    def file_check(self, path):
        if not os.path.exists(path):
            self.success = False
            self.err_msgs.append("Path does not exist: "+path)
    def eval(self):
        return self.success
    def print_errs(self):
        for err_msg in self.err_msgs:
            print(err_msg)
    def reset(self):
        self.success = True

class Colorcodes(object):
    """
    Provides ANSI terminal color codes which are gathered via the ``tput``
    utility. That way, they are portable. If there occurs any error with
    ``tput``, all codes are initialized as an empty string.
    The provides fields are listed below.
    Control:
    - bold
    - reset
    Colors:
    - blue
    - green
    - orange
    - red
    :license: MIT
    """
    def __init__(self):
        try:
            self.bold = subprocess.check_output("tput bold".split()).decode()
            self.reset = subprocess.check_output("tput sgr0".split()).decode()

            self.blue = subprocess.check_output("tput setab 4".split()).decode()
            self.green = subprocess.check_output("tput setab 2".split()).decode()
            self.orange = subprocess.check_output("tput setab 3".split()).decode()
            self.red = subprocess.check_output("tput setab 1".split()).decode()
        except subprocess.CalledProcessError as e:
            self.bold = ""
            self.reset = ""

            self.blue = ""
            self.green = ""
            self.orange = ""
            self.red = ""

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout