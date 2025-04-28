"""
Configuration module to store shared variables across modules.
This helps prevent circular imports between engine.py and tester/accuracy.py.
"""

# Command line arguments configuration
CMD_CONFIG = None

def get_cfg():
    global CMD_CONFIG
    return CMD_CONFIG

def set_cfg(cfg):
    global CMD_CONFIG
    if cfg.id != "":
        cfg.id = "_"+cfg.id
    CMD_CONFIG = cfg
