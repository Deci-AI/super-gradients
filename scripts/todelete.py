import sys
import io



class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
        print("ENNDDD STD_OUT")

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()


class Tee2(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stderr = sys.stderr
        sys.stderr = self

    def __del__(self):
        sys.stderr = self.stderr
        if sys.exc_info()[0] is not None:
            # only entered if there's an *unhandled* exception, e.g. NOT a HandleThis exception
            print('funky code raised')
        self.file.close()
        print("ENNDDD STD_ERR")

    def write(self, data):
        self.file.write(data)
        self.stderr.write(data)
        if sys.exc_info()[0] is not None:
            # only entered if there's an *unhandled* exception, e.g. NOT a HandleThis exception
            print('funky code raised 2')

    def flush(self):
        self.file.flush()


Tee("/home/louis.dupont/PycharmProjects/super-gradients/local_tee_out.log", "a")
Tee2("/home/louis.dupont/PycharmProjects/super-gradients/local_tee_err.log", "a")

print("=================-------------")

raise ValueError("0 is not valid")
