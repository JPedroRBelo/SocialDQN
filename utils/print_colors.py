class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def cyan(text):
    format(bcolors.OKCYAN,text)

def format(color,text):
    print(color+text + bcolors.ENDC)

def attention(text):
    format(bcolors.WARNING ,"ATTENTION: "+text)

def warning(text):
    format(bcolors.WARNING , "WARNING: "+text)

def error(text):
    format(bcolors.FAIL , "ERROR: "+text)

def header(text):
    format(bcolors.HEADER,text)

def blue(text):
    format(bcolors.OKBLUE,text)

def bold(text):
    format(bcolors.BOLD,text)
