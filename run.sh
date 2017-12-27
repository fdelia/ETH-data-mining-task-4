# use:
#   in python:
#   import os
#   alpha = float(os.environ['ALPHA'])
# correct path/filename
# make executable chmod +x run.sh
# ./run.sh

for i in `seq 2 0.3 5`
do
    export ALPHA=$i
    echo "${i}  ${j}"
    python2.7 runner.py data/webscope-logs.txt data/webscope-articles.txt subm_try.py
done
