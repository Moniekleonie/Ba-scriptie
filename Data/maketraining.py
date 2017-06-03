import random


def main():
	with open ('scriptiedata.txt') as tweets:
		
		random.seed(26)
		lines = random.sample(tweets.readlines(),1000)
		for line in lines:
			line = line.rstrip()
			print(line)

main()
