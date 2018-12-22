import os

persistent_directory = '/storage'
published_directory = '/artifacts'
fast_directory = '/workspace'

cold_data_directory = os.path.join(persistent_directory, 'data')
hot_data_directory = os.path.join(fast_directory, 'data')
openimages_directory = os.path.join(hot_data_directory, "openimages-v4/bb")
log_directory = os.path.join(persistent_directory, 'logs')
