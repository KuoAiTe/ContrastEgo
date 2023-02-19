# MongoDB
sudo mongod --config /etc/mongod.conf 
mongod --dbpath /media/aite/OS/mongodb
# database preproces
# This only needs one run.
python3 db_preprocess.py

# Collect data from the specified time windows.
python3 run.py

# Convert the collected data from run.py to graphs
convert_df_to_graph.py