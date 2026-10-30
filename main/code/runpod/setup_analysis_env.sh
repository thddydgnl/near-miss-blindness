#!/bin/bash
# Separate small venv for the E1 analysis (numpy/scipy/scikit-learn/matplotlib), so the cosmos-policy venv used by the
# runner stays unchanged. Usage on the pod: bash /workspace/code/setup_analysis_env.sh  -> /root/b/ana/bin/python
source /workspace/code/env.sh
[ -x $RB/ana/bin/python ] || uv venv -q $RB/ana --python 3.10
uv pip install -q --python $RB/ana/bin/python numpy==2.2.6 scipy==1.15.3 scikit-learn==1.7.2 matplotlib==3.10.8
$RB/ana/bin/python -c "import numpy, scipy, sklearn, matplotlib; print('ana env', numpy.__version__, scipy.__version__, sklearn.__version__, matplotlib.__version__)"
