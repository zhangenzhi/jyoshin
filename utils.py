import os
import ruamel.yaml as yaml
from colorama import Fore

ERROR_INFO = 'ERROR: '
NORMAL_INFO = 'INFO: '
WARNING_INFO = 'WARNING: '


def write_to_file(path, filename, s):
    check_mkdir(path=path)
    file = os.path.join(path, filename)
    with open(file=file, mode='a') as f:
        f.writelines(s+'\n')


def check_file(file):
    if not os.path.exists(path=file):
        print_error("no such file: {}".format(file))


def check_mkdir(path):
    if not os.path.exists(path=path):
        print_warning("no such path: {}, but we made.".format(path))
        os.makedirs(path)


def get_yml_content(file_path):
    '''Load yaml file content'''
    try:
        with open(file_path, 'r') as file:
            return yaml.load(file, Loader=yaml.Loader)
    except yaml.scanner.ScannerError as err:
        print_error('yaml file format error!')
        print_error(err)
        exit(1)
    except Exception as exception:
        print_error(exception)
        exit(1)


def print_error(*content):
    '''Print error information to screen'''
    print(Fore.RED + ERROR_INFO + ' '.join([str(c)
                                            for c in content]) + Fore.RESET)


def print_green(*content):
    '''Print information to screen in green'''
    print(Fore.GREEN + ' '.join([str(c) for c in content]) + Fore.RESET)


def print_normal(*content):
    '''Print error information to screen'''
    print(NORMAL_INFO, *content)


def print_warning(*content):
    '''Print warning information to screen'''
    print(Fore.YELLOW + WARNING_INFO + ' '.join([str(c) for c in content]) +
          Fore.RESET)


def print_dict(d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            print_green('\t' * (indent) + key + ": ")
            print_dict(value, indent + 1)
        else:
            try:
                print('\t' * (indent) + str(key) + ": " + str(value))
            except:
                print_error("value can not print: ", type(value))
                raise


if __name__ == '__main__':
    print_error("wang ru qin")
    print("wang ru qin")
