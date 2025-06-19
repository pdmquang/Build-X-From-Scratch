import argparse
import configparser
from datetime import datetime
import grp, pwd 
from fnmatch import fnmatch
import hashlib
from math import ceil
import os
import re
import sys
import zlib

argparser = argparse.ArgumentParser(description="The stupidest content tracker")

# subcommand (aka. subparser) e.g: git [init, commit, etc]
argsubparsers = argparser.add_subparsers(title="Commands", dest="command")
argsubparsers.require = True

def main(argv=sys.argv[1:]):
	args = argparser.parse_args(argv)
	match args.command:
		case "add" : cmd_add(args)
		case "init" : cmd_init(args)
		case _ : print("Bad command.")

	
