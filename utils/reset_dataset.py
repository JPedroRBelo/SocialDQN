import os

os.system("rm -rf dataset/RGB/ep*")
os.system("rm -rf dataset/RGB/ep*")
os.system("rm -rf results/*")
os.system("python makefiles.py")
os.system("python init_model.py")
#os.system("mkdir results/ep0")
#os.system("cp -RT ep13/ results/ep0")
os.system("python makefiles.py")
#os.system("python mdqn.py")