def gemeentes ():
	gemeentenlist = []

	with open('gemeenten-alfabetisch-2016.txt', 'r') as gemeenten:
		for g in gemeenten:  
			g = g.rstrip()
			g = g.lower()
			gemeentenlist.append(g)
	return(gemeentenlist)


def main():
        # Define Netherland
	ned = ["TheNetherlands", "Nederland", "Niederlande", "Pays-Bas"]

	# Get gemeente names
	gemeenten = gemeentes()
	
	with open ('scriptiedata.txt', encoding='utf-8') as tweets:
		for line in tweets:
			line = line.rstrip()
			(id, user, text, words, hastags, date, place) = line.split("\t")
			plaatsnaamlijst = place.split(",")
                        # Check if place has 2 or 3 elements 

			
			try:
				land = plaatsnaamlijst[2].replace(" ", "")
			except:
				land = plaatsnaamlijst[1].replace(" ", "")

			if land in ned:

				plaatsnaam = plaatsnaamlijst[0].replace("/","")
				plaatsnaam = plaatsnaam.lower()


				if plaatsnaam in gemeenten:

					plaatsnaam = plaatsnaam.replace(" ", "")
					filename = 'gemeenten/' + plaatsnaam + '.txt'

					# Make new file for every gemeente or append on excisting file

					with open(filename, 'a') as file:
							file.writelines(line + '\n')
			
main ()
