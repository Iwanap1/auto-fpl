# The script that we will run to make the transfer each GW

from utils import *
import os

email = os.getenv('FPL_EMAIL')
password = os.getenv('FPL_PASSWORD')
manager_id = os.getenv('FPL_MANAGER_ID')

current_team = get_current_team(email, password, manager_id)
# {
#     "goalkeepers": [int, int],
#     "defenders": [int, int, int, int, int],
#     "midfielders": [int, int, int, int, int],
#     "forwards": [int, int, int]
# }