#!/bin/bash

VENV=env
MODEL_DIR=data
MODEL_FILE=german.model
NLTK_DATA_ID=punkt

echo -e "\n--> Setting up python virtual environment: $VENV"
virtualenv -p python3 "$VENV"
. "$VENV/bin/activate"
pip install -r requirements.txt

echo -e "\n--> Downloading German Language Model ($MODEL_FILE) to $MODEL_DIR"
mkdir -p "$MODEL_DIR"
wget -c -O "$MODEL_DIR/$MODEL_FILE" "https://tubcloud.tu-berlin.de/s/dc4f9d207bcaf4d4fae99ab3fbb1af16/download"

echo -e "\n--> Downloading NLTK data: $NLTK_DATA_ID"
python3 -c "import nltk; nltk.download('punkt')"

echo -e "\n--> Recreating dependency.json"
echo '{"mdl":"'"$MODEL_DIR/$MODEL_FILE"'"}' > dependency.json

echo -e "\n--> TODO: Get the dataset from Architraver/Media/Playground/ExtractionData/data and place it into the 'data' directory of this project."
echo -e "\n--> When this is done, you should be able to run the project with the scripts in the 'calller' directory."
