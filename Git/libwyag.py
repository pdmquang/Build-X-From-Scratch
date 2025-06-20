import argparse
import configparser
from datetime import datetime
# import grp, pwd 
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
	argsp = argsubparsers.add_parser("init", help="Initialize a new, empty repository.")
	argsp.add_argument("path",
                   metavar="directory",
                   nargs="?",
                   default=".",
                   help="Where to create the repository.")
	args = argparser.parse_args(argv)
	match args.command:
		# case "add" : cmd_add(args)
		case "init" : cmd_init(args)
		case _ : print("Bad command.")

def cmd_init(args):
	repo_create(args.path)

class GitRepository(object):
	"""A git repository"""

	worktree = None
	gitdir = None
	conf = None

	def __init__(self, path, force=False): 
		# Force: disable all checks? -> allow repo_create() create Repo in brute-force way 
		self.worktree = path
		self.gitdir = os.path.join(path, ".git")

		if not (force or os.path.isdir(self.gitdir)):
			# if worktree doesnt contain .git file, raise Error
			# force: bypass this check, create .git from scratch
			raise Exception(f"Not a Git repository {path}")
		
		# Read config file (.git/config)
		self.conf = configparser.ConfigParser()
		cf = repo_file(self, "config")

		if cf and os.path.exists(cf):
			self.conf.read([cf])
		elif not force:
			raise Exception("Configuration file missing")
		
		# Check version of configuration
		if not force:
			vers = int(self.conf.get("core", "repositoryformatversion"))
			if vers != 0:
				raise Exception(f'Unsupported repositoryformatversion: {vers}')

def repo_path(repo, *path): 
	# path as List(), e.g repo_path(repo, "objects", "df", "4ec9fc2ad990cb9da906a95a6eda6627d7b7b0")
	"""Compute path under repo's gitdir."""
	return os.path.join(repo.gitdir, *path)

def repo_file(repo, *path, mkdir=False):
	'''
	Same as repo_path, but create dirname(*path) if absent
	e.g repo_file(repo, \"refs\", \"remotes\", \"origin\", \"HEAD\") will create
	.git/refs/remotes/origin
	'''

	if repo_dir(repo, *path[:-1], mkdir=mkdir): # Check if repo dir exists...
		return repo_path(repo, *path) 
	
def repo_dir(repo, *path, mkdir=False):
	'''Same as repo_path, but mkdir *path if absent if mkdir.'''

	path = repo_path(repo, *path)

	if os.path.exists(path): # Check if repo path exists
		if (os.path.isdir(path)):
			return path
		else:
			raise Exception(f'Not a directory {path}')

	''' Why mkdir is after raise Exception???'''	
	if mkdir:
		os.makedirs(path)
		return 
	else:
		return None
	
def repo_create(path):
	"""Create a new repository at path"""

	repo = GitRepository(path, True)

	# First, we make sure the path either doesn't exist or is an empty dir
	if os.path.exists(repo.worktree):
		if not os.path.isdir(repo.worktree):
			raise Exception(f'{path} is not a directory')
		if os.path.exists(repo.gitdir) and os.listdir(repo.gitdir):
			raise Exception(f'{path} is not empty')
	else:
		os.makedirs(repo.worktree)

	assert repo_dir(repo, "branches", mkdir=True)
	assert repo_dir(repo, "objects", mkdir=True)
	assert repo_dir(repo, "refs", "tags", mkdir=True)
	assert repo_dir(repo, "refs", "heads", mkdir=True)

	# .git/description (rarely used)
	with open(repo_file(repo, "description"), "w") as f:
		f.write("Unnamed repository; edit this file 'description' to name the repository.\n")
	
	# .git/HEAD
	with open(repo_file(repo, "HEAD"), "w") as f:
		f.write("ref: refs/heads/master\n")

	with open(repo_file(repo, "config"), "w") as f:
		config = repo_default_config()
		config.write(f)

	return repo

def repo_default_config():
	'''
	repositoryformatversion: 0-initial, 1-with extension
	filemode = false: disable tracking of file modes (permission)
	bare = false: indicate this repo has a worktree
	'''
	ret = configparser.ConfigParser()

	ret.add_section("core")
	ret.set("core", "repositoryformatversion", "0")
	ret.set("core", "filemode", "false")
	ret.set("core", "bare", "false")

	return ret