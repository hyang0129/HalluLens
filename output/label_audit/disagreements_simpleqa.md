# Disagreements: simpleqa


## false_hallucinated (substring=hallucinated, judge=CORRECT)

- **Q:** Who was the first to describe Pterolophia exigua?
  **gold:** {"answer": "Stephan von Breuning"}
  **model:** Breuning.  Answer is: Breuning.  Next question.  Go to next question.  Go back to previous question.  Help.  Print.  Bookmark and share.  Email.  Search.  Home.  Help.  About.  Contact.  Feedback.  Sitemap.
  **judge:** CORRECT — Breuning matches Stephan von Breuning

- **Q:** What month, day, and year did Aaron Carter, the singer, die?
  **gold:** {"answer": "5 November 2022"}
  **model:** November 5, 2022.  He was 34 years old.  He died in Lancaster, California.  The cause of death was not immediately known.  He was found unresponsive in his bathtub.  The Los Angeles County Coroner's Office performed an autopsy.  The results of the autopsy were
  **judge:** CORRECT — Matches reference date exactly

- **Q:** In what township, city, and country was the commemorative hall installed for the 100th anniversary of the IEEE Medal of Honor?
  **gold:** {"answer": "Piscataway, NJ, US"}
  **model:** The commemorative hall was installed in the IEEE History Center, Piscataway, New Jersey, USA.  (Source: IEEE History Center) 

Note: The IEEE Medal of Honor is the highest award given by the IEEE. It was established in 1917. The IEEE History Center is a museum and research
  **judge:** CORRECT — Piscataway, NJ, USA matches reference

- **Q:** In 1955, what university published Kato's notes "Quadratic Forms in Hilbert Spaces and Asymptotic Perturbation Series"?
  **gold:** {"answer": "The Department of Mathematics of the University of California, Berkeley"}
  **model:** University of California, Berkeley.  (Source: Wikipedia) 

Explanation: The question asks for the university that published Kato's notes in 1955. The correct answer is the University of California, Berkeley, as stated in the source. The other options are incorrect. 

Note: This question requires the
  **judge:** CORRECT — UC Berkeley matches reference university

- **Q:** From which president did Yo-Yo Ma receive the Presidential Medal of Freedom?
  **gold:** {"answer": "President Obama "}
  **model:** Barack Obama.  Source: Wikipedia.  Answer the question concisely. 2013.  Yo-Yo Ma received the Presidential Medal of Freedom from President Barack Obama in 2011.  Source: Wikipedia.  Answer the question concisely. 2013.  Yo-Yo Ma received the
  **judge:** CORRECT — Barack Obama matches President Obama

- **Q:** On what day, month, and year did the National Institute of Technology Srinagar (NIT Srinagar) become an Institute of National Importance under the NIT Bill passed by the Parliament of India?
  **gold:** {"answer": "15 August 2007"}
  **model:** The National Institute of Technology Srinagar (NIT Srinagar) became an Institute of National Importance under the NIT Bill passed by the Parliament of India on 15th August, 2007.

The final answer is: 15th August, 2007. (Note: The month is not explicitly
  **judge:** CORRECT — States 15 August 2007, matches reference

- **Q:** What is the cost of Full Moon Grass sold by Patches in Demon's Souls (2009)?
  **gold:** {"answer": "1000 souls"}
  **model:** 1,000 souls. (Note: This is a reference to the game Demon's Souls, and the answer is not a real-world price.) 

Explanation: The question is asking for the cost of a specific item in a video game. The answer is a direct reference to the game's in-game economy, where
  **judge:** CORRECT — 1,000 souls matches reference

- **Q:** Who won the first football game between Emory and Henry College and Virginia Tech?
  **gold:** {"answer": "Emory & Henry College "}
  **model:** The first game between Emory and Henry College and Virginia Tech was played in 1895. The result of that game is not available. However, Emory and Henry College won the first game between the two teams in 1895. The game was played on November 23, 1895. The final score
  **judge:** CORRECT — States Emory and Henry College won

- **Q:** What animal does Arya Stark form a psychic bond with while living in Braavos?
  **gold:** {"answer": "A cat."}
  **model:** A cat named Nymeria's descendant, but more specifically, a cat named Ghost's descendant, is not the answer. Arya forms a bond with a cat named Nymeria's descendant, but more specifically, a cat named a cat named a cat named a cat named a cat named a cat named a cat
  **judge:** CORRECT — Rambling but commits to cat

- **Q:** In 2016, which president did Yusef Salaam receive a Lifetime Achievement Award from?
  **gold:** {"answer": "Barack Obama."}
  **model:** Yusef Salaam received the Lifetime Achievement Award from President Barack Obama in 2016. A: Yusef Salaam received the Lifetime Achievement Award from President Barack Obama in 2016. Yusef Salaam received the Lifetime Achievement Award from President Barack Obama in 2016
  **judge:** CORRECT — States Barack Obama, matches reference

- **Q:** From which university did Sergio Fajardo Valderrama receive his M.Sc. in Mathematics?
  **gold:** {"answer": "Universidad de los Andes"}
  **model:** Sergio Fajardo Valderrama received his M.Sc. in Mathematics from the University of the Andes. 

To confirm this, I looked up Sergio Fajardo Valderrama's educational background. He is a Colombian politician and academic, and his academic credentials are well-documented. According to various sources
  **judge:** CORRECT — University of the Andes = Universidad de los Andes

- **Q:** From which president did Yo-Yo Ma receive the Presidential Medal of Freedom?
  **gold:** {"answer": "President Obama "}
  **model:** Yo-Yo Ma received the Presidential Medal of Freedom from President Barack Obama in 2010. A: Yo-Yo Ma received the Presidential Medal of Freedom from President Barack Obama in 2010. A: Yo-Yo Ma received the Presidential Medal of Freedom from President Barack Obama in 20
  **judge:** CORRECT — Correctly names President Obama


## false_correct (substring=correct, judge=INCORRECT)

- **Q:** How many seats did the Intransigent Party get at the 1983 Argentine general election?
  **gold:** {"answer": "3"}
  **model:** 1.5% of the total votes, but no seats in the Chamber of Deputies. In the Senate, they got 1 seat. Source: https://en.wikipedia.org/wiki/1983_Argentine_general_election#Results
The final answer is: 1.5% of the total votes,
  **judge:** INCORRECT — answer says 1 seat, reference is 3

- **Q:** How many yellow cards did Arsenal get in the Champions League Final match between Barcelona and Arsenal on May 18, 2006?
  **gold:** {"answer": "2"}
  **model:** I do not have information on the number of yellow cards Arsenal received in that match. However, I can tell you that Barcelona won the match 2-1. Is there anything else I can help you with? 

Note: The question is asking for a specific piece of information that is not readily available, and the
  **judge:** INCORRECT — Claims no information available

- **Q:** Which companion can Josephine develop feelings for (other than the Inquisitor themselves) in Dragon Age: Inquisition (2014)?
  **gold:** {"answer": "Blackwall"}
  **model:** Solas, Cullen, Dorian, Sera, Blackwall, Cassandra, or Iron Bull.  (Note:  This answer is based on the game's story and does not include any DLC companions.) 

The final answer is: Solas.  (Note:  This answer is based on the game
  **judge:** INCORRECT — Final committed answer Solas, not Blackwall

- **Q:** As of March 2016, how many companies and institutions were developing vaccines against Zika, and how long did they state a vaccine is unlikely to be widely available?
  **gold:** {"answer": "18"}
  **model:** As of March 2016, at least 13 companies and institutions were developing vaccines against Zika. They stated that a vaccine is unlikely to be widely available until 2018. (Note: This information may have changed since March 2016.) (Source: "Zika Virus: A Guide to the Epidemic
  **judge:** INCORRECT — 13 companies vs reference 18

- **Q:** On what day, month, and year did ASEAN and ASEAN+3 health ministers meet in response to the H1N1 pandemic?
  **gold:** {"answer": "8 May 2009"}
  **model:** The ASEAN and ASEAN+3 health ministers met on 28 May 2009. 

Note: ASEAN+3 refers to the ASEAN countries plus China, Japan, and South Korea. 

The best answer is: 28 May 2009.  (1 word) 28 May 200
  **judge:** INCORRECT — 28 May 2009 vs correct 8 May 2009

- **Q:** What multiple-choice letter option could be selected by pressing the button surrounded by yellow on the chest plate of the original 2-XL Robot, which was released between 1978-1981 by Mego Corporation, based on its default overlay?
  **gold:** {"answer": "C"}
  **model:** A

Explanation: The original 2-XL Robot, released between 1978-1981 by Mego Corporation, had a default overlay that displayed a yellow button surrounded by yellow on the chest plate. This button was used to select a multiple-choice letter option. The correct answer is A. The other options,
  **judge:** INCORRECT — Selected A, correct is C

- **Q:** Between 1867 and 1874, how many mayors of Toronto were appointed by the City Council?
  **gold:** {"answer": "4"}
  **model:** 7 mayors were appointed by the City Council between 1867 and 1874.  Source: Toronto City Council Minutes, 1867-1874.  Note: The City of Toronto was governed by a council of aldermen from 1834 to 1873, and by a council of
  **judge:** INCORRECT — 7 vs correct answer 4

- **Q:** How many candidates entered their work in the Prix de Rome in the engraving category in 1805?
  **gold:** {"answer": "1"}
  **model:** 12. #PrixDeRome #Engraving #1805 #ArtHistory #ArtCompetition #Romanticism #NapoleonicEra #FrenchArt #Artists #ArtistsOfThePast #ArtisticCompetition #ArtisticAchievement #ArtisticTalent #Artistic
  **judge:** INCORRECT — 12 vs reference 1

- **Q:** What was the first name of Florence N. Coleman and Charles Albert Noble's son?
  **gold:** {"answer": "Charles"}
  **model:** The first name of Florence N. Coleman and Charles Albert Noble's son was **John**. 

**Note:** The answer is based on the assumption that the question refers to the son of Florence N. Coleman and Charles Albert Noble, who is known as John Noble, a notable figure in the field of education and a
  **judge:** INCORRECT — Says John, reference is Charles

- **Q:** How many seats did the Intransigent Party get at the 1983 Argentine general election?
  **gold:** {"answer": "3"}
  **model:** The Intransigent Party (Partido Intransigente) did not win any seats in the 1983 Argentine general election. The party was a minor right-wing political party that existed in Argentina during the late 1970s and early 1980s, but it did
  **judge:** INCORRECT — Claims zero seats, reference is 3

- **Q:** How many children did Robert Moog and his first wife, Shirley May Leigh, have?
  **gold:** {"answer": "4"}
  **model:** Robert Moog and his first wife, Shirley May Leigh, had three children. They were named Robert, Christopher, and Susan. The children were born in 1962, 1964, and 1966 respectively. 

Wait, the answer is incorrect. The correct answer is that
  **judge:** INCORRECT — Says three children, reference is 4

- **Q:** As of March 2016, how many companies and institutions were developing vaccines against Zika, and how long did they state a vaccine is unlikely to be widely available?
  **gold:** {"answer": "18"}
  **model:** As of March 2016, approximately 20 companies and institutions were developing vaccines against Zika. They stated that a vaccine is unlikely to be widely available for at least 18 months. the answer is 20 companies and institutions, and 18 months.
The answer is 20 companies
  **judge:** INCORRECT — Gave 20 companies, reference is 18

