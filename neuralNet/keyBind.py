import win32api as wapi
import time

originalKeys = ["\b"]
for character in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.":
	originalKeys.append(character)
def key_check():
	keys = []
	for char in originalKeys:
		if wapi.GetAsyncKeyState(ord(char)):
			keys.append(char)
		return keys

