from __future__ import print_function

import string

def char2id(char):
	first_letter = ord(string.ascii_lowercase[0])

	if char in string.ascii_lowercase:
		return ord(char) - first_letter + 1
	elif char == ' ':
		return 0
	else:
		print('Unexpected character: %s' % char)
		return 0

def id2char(charid):
	first_letter = ord(string.ascii_lowercase[0])

	if charid > 0:
		return chr(charid + first_letter - 1)
	else:
		return ' '
