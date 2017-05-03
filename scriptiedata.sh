#!/bin/bash

jan='/net/corpora/twitter2/Tweets/2016/01/201601*.out.gz'
feb='/net/corpora/twitter2/Tweets/2016/02/201602*.out.gz'
maa='/net/corpora/twitter2/Tweets/2016/03/201603*.out.gz'
apr='/net/corpora/twitter2/Tweets/2016/04/201604*.out.gz'
mei='/net/corpora/twitter2/Tweets/2016/05/201605*.out.gz'
jun='/net/corpora/twitter2/Tweets/2016/06/201606*.out.gz'
jul='/net/corpora/twitter2/Tweets/2016/07/201607*.out.gz'
aug='/net/corpora/twitter2/Tweets/2016/08/201608*.out.gz'
sep='/net/corpora/twitter2/Tweets/2016/09/201609*.out.gz'
okt='/net/corpora/twitter2/Tweets/2016/10/201610*.out.gz'
nov='/net/corpora/twitter2/Tweets/2016/11/201611*.out.gz'
dec='/net/corpora/twitter2/Tweets/2016/12/201612*.out.gz'


zcat  $jan $feb $maa $apr $mei $jun $jul $aug $sep $okt $nov $dec| /net/corpora/twitter2/tools/tweet2tab -i id user text words hashtags date place |  grep -i -e 'immigrant' -e 'migrant' -e 'vluchteling' -e 'migratie' -e 'immigratie' > scriptie2016.txt


