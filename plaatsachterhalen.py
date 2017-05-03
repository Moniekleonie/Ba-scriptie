
def main():
	with open ("scriptie201601.txt") as tweets:
		for line in tweets:
			line = line.rstrip()
			hoi= line.split("\t")
			if len(hoi) == 7:
				nieuwetweets= '\t'.join(hoi)
				print(nieuwetweets)
main()