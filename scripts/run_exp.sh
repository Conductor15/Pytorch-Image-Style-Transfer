EXP_PATH=$1

if [ -z "$EXP_PATH" ]; then
  echo "Usage: bash scripts/run_exp.sh experiments/exp_01/config.yaml"
  exit 1
fi

python3 src/run_style_transfer.py $EXP_PATH