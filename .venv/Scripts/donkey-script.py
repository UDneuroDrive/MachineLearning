#!"c:\users\fix\documents\senior design\machinelearning\.venv\scripts\python.exe"
# EASY-INSTALL-ENTRY-SCRIPT: 'donkeycar==2.5.8','console_scripts','donkey'
__requires__ = 'donkeycar==2.5.8'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('donkeycar==2.5.8', 'console_scripts', 'donkey')()
    )
