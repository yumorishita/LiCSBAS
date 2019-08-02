## Path
export LICSBAS_PATH="$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)"
export PATH="$LICSBAS_PATH/bin:$PATH"
export PYTHONPATH="$LICSBAS_PATH/LiCSBAS_lib:$PYTHONPATH"

