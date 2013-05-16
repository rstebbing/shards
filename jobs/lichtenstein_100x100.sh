#!/bin/sh
export PYTHONPATH=/users/RichardStebbing/Code/Python:$PYTHONPATH
export LD_LIBRARY_PATH=/users/RichardStebbing/lib/InsightToolkit:/users/RichardStebbing/lib:/users/RichardStebbing/lib64
cd /users/RichardStebbing/Code/Projects/transparency
/users/RichardStebbing/bin/python solve_multiple_shards.py examples/lichtenstein_100x100.png lichtenstein_100x100_white_4.0_0.3_1000_10_update-colours white 4.0 0.3 1000 --maxiter 10 --update-colours
