# to install dependencies. if you don't have uv, check <https://docs.astral.sh/uv/getting-started/installation/>

uv sync

# setting up playwright

playwright install

# configuring variables

cp example.env .env

add your .env with your COMPARTMENT_ID

# run the script

uv run jarvis.py
