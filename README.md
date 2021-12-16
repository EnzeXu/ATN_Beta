
ATN_Beta
===========================
This document is the description of work for ATN_Beta

****
 
| Project Name | Authors |
| ---- | ---- |
| ATN_Auto | Enze Xu (xue20@wfu.edu) & Jingwen Zhang (zhanj318@wfu.edu) |

| Version | Date       | Comment |
|---------|------------|---------|
| v1.0    | 11/17/2021 | Auto    |
| v2.0    | 12/13/2021 | Alpha   |
| v3.0    | 12/15/2021 | Beta    |

| Python Version | Platform | GPU or CPU |
| ---- | ---- | ---- |
| python3.5 / 3.6 / 3.7 / 3.8 | Linux / Windows / MacOS | Both OK |

****
# Catalog

* [1 Purpose](#1-purpose)
* [2 Build Virtual Environment](#2-build-virtual-environment-done)
* [3 Cloning & Executing Instructions](#3-cloning--executing-instructions)

****

# 1 Purpose

1. Auto test for DPS model on ATN datasets.

****

# 2 Build Virtual Environment (Done)
```shell
$ sudo pip3 install virtualenv
$ cd ~
$ virtualenv atn_env
```

# 2 Cloning & Executing Instructions
```shell
$ source /deac/csc/chenGrp/software/tensorflow/bin/activate # activate virtual environment
(tensorflow) $ cd ~/workspace
(tensorflow) $ git clone https://github.com/EnzeXu/ATN_Beta.git
(tensorflow) $ cd ATN_Beta
(tensorflow) $ pip install -r requirements.txt
(tensorflow) $ python auto.py --data beta1 --num 10 --comment test
# It is normal to see huge warnings at this step, but don't worry.
# after it finishes (may cost several minutes)
(tensorflow) $ cat record/alpha1/record.csv
(tensorflow) $ deactivate
```
