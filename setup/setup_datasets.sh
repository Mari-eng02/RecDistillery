export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python ./scripts/data_preparation/citeulike.py
python ./scripts/data_preparation/bookcrossing.py
python ./scripts/data_preparation/amazon_cd_2014.py
