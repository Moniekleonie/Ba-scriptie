
def main():
	with open ("scriptie2016.txt") as tweets:
		for line in tweets:
			line = line.rstrip()
			l = line.split("\t")
			if len(l) == 7:
				nieuwetweets= '\t'.join(l)
				print(nieuwetweets)
main()
