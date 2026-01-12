import os

_TEST_ROOT = os.path.dirname(__file__) # root of test folder, e.g. 'project_name/tests/'
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT) # root of project, i.e. 'project_name/'
_PATH_DATA = os.path.join(_PROJECT_ROOT, "data") # root of data folder. should include 'raw/' and 'processed/' as subfolders
