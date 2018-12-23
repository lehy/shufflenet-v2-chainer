import os

persistent_directory = '/home/paperspace'
published_directory = '/home/paperspace/public_html'
fast_directory = '/home/paperspace/'

cold_data_directory = os.path.join(persistent_directory, 'data')
hot_data_directory = os.path.join(fast_directory, 'data')
openimages_directory = os.path.join(hot_data_directory, "openimages-v4/bb")
log_directory = os.path.join(persistent_directory, 'logs')
