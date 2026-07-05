# Disagreements: hotpotqa


## false_hallucinated (substring=hallucinated, judge=CORRECT)

- **Q:** How many members does the Tennessee Volunteers football team's Conference have?
  **gold:** {"answer": "fourteen members"}
  **model:** The Tennessee Volunteers football team's conference is the Southeastern Conference (SEC), which has 14 member institutions. However, the question asks for the number of members in the conference, not the number of institutions. The SEC has 14 member institutions, but the question is asking for the n
  **judge:** CORRECT — States 14 members, matches fourteen

- **Q:** Walt Zeboski photographed which 40th President during his campaign in 1980?
  **gold:** {"answer": "Ronald Wilson Reagan"}
  **model:** Ronald Reagan.  (Answer is 1 word)  (Answer should be a single word) 

The best answer is Reagan.  (Answer is 1 word)  (Answer should be a single word)  (Answer should be a single word)  (Answer should be a single word)  (
  **judge:** CORRECT — Ronald Reagan matches full name

- **Q:** Rimo I and Passu Sar are both part of which mountain range?
  **gold:** {"answer": "Karakoram mountain range"}
  **model:** Karakoram range.  (Answer should be in one line)  Karakoram range.  (Answer should be in one line)  Karakoram range.  (Answer should be in one line)  Karakoram range.  (Answer should be in one line)  Karakoram range
  **judge:** CORRECT — States Karakoram range

- **Q:** Is The Stuff of Legend and Babes in Toyland both entertainment? 
  **gold:** {"answer": "by"}
  **model:** Yes. The Stuff of Legend is a comic book series and Babes in Toyland is a 1961 film. Both are forms of entertainment. 

Note: The question is asking if both are entertainment, not if they are both the same type of entertainment. 

This answer is concise because it directly addresses the question
  **judge:** CORRECT — Answers Yes, semantically matches expected

- **Q:** Who aided the Colorado forces in the Uruguayan war?
  **gold:** {"answer": "Argentina"}
  **model:** The Uruguayan forces were aided by the Argentine forces.  The Colorado party was aided by the Argentine forces.  The Colorado party was aided by the Argentine forces.  The Colorado party was aided by the Argentine forces.  The Colorado party was aided by the Argentine forces.  The Colorado party was
  **judge:** CORRECT — States Argentine forces aided

- **Q:** Which film industry would have films in which Ilya Khrzhanovsky would be the director?
  **gold:** {"answer": "The cinema of Russia"}
  **model:** Russian film industry.  Ilya Khrzhanovsky is a Russian film director. He is known for his work on films such as "The Man with the Golden Arm" and "The Man with the Golden Arm 2". He has also directed several documentaries and short films. Khrzhanovsky's films
  **judge:** CORRECT — Russian film industry = cinema of Russia

- **Q:** Sian Gibson co-wrote which British sitcom, set around a supermarket and car share scheme?
  **gold:** {"answer": "Peter Kay's Car Share"}
  **model:** Car Share.  Sian Gibson co-wrote the British sitcom Car Share with Phil Hanson. The show is set around a supermarket and a car share scheme. It was first broadcast in 2015 and ran for four seasons. The show follows the lives of two colleagues, John Redmond and Kayleigh Kitson,
  **judge:** CORRECT — Car Share = Peter Kay's Car Share

- **Q:** Which Olympic game featured Boules and happened right before medals started to be awarded?
  **gold:** {"answer": "The Summer Olympic Games"}
  **model:** The 1900 Summer Olympics.  Boules was a sport featured in the 1900 Summer Olympics.  The medals were awarded after the sport was completed.  The medals were awarded in the order of the sport being completed.  The medals were awarded in the order of the sport being completed.  The medals
  **judge:** CORRECT — 1900 Summer Olympics is a Summer Olympics

- **Q:** Which is a record producer, Zach Bogosian or Daron Malakian?
  **gold:** {"answer": "Daron Vartan Malakian"}
  **model:** Daron Malakian is a record producer. Zach Bogosian is a professional ice hockey defenseman.  next
The best answer is Daron Malakian. next
Daron Malakian is a record producer. Zach Bogosian is a professional ice hockey defenseman.  next
The best answer is
  **judge:** CORRECT — Selects Daron Malakian as producer

- **Q:** Philadelphia, Here I come! is a 1964 play by an Irish dramatist who has been likened to an Irish Checkhov?
  **gold:** {"answer": "Brian Patrick Friel"}
  **model:** Brian Friel.  Next question, please.  (Note:  The question is asking for the name of the Irish dramatist, not the title of the play.) (Source:  Wikipedia) Next question, please.  (Note:  The question is asking for the name of the Irish dramat
  **judge:** CORRECT — Brian Friel matches Brian Patrick Friel

- **Q:** What do Susan Stafford and Vanna White have in common?
  **gold:** {"answer": "\"Wheel of Fortune\""}
  **model:** They are both letter turners on the TV game show Wheel of Fortune.  Susan Stafford was the original letter turner and Vanna White has been the letter turner since 1982.  They are both women.  They are both American.  They are both blonde.  They are both famous.
  **judge:** CORRECT — Both letter turners on Wheel of Fortune

- **Q:** The seventh largest inhabited island in Hawaii has what remote wild beach?
  **gold:** {"answer": "Polihale State Park"}
  **model:** Polihale Beach.  It is located on the western coast of Kauai.  It is a remote beach that is accessible only by a dirt road.  It is a popular destination for beachgoers and surfers.  It is known for its white sand and turquoise waters.  It is a protected
  **judge:** CORRECT — Polihale Beach matches Polihale State Park


## false_correct (substring=correct, judge=INCORRECT)

- **Q:** Which band My Darkest Days or Human Drama had only one constant member?
  **gold:** {"answer": "Human Drama"}
  **model:** My Darkest Days.  The only constant member of My Darkest Days was Matt Walst.  The only constant member of Human Drama was John Corabi.  Matt Walst was the lead vocalist of My Darkest Days.  John Corabi was the lead vocalist of Human Drama.  Matt Walst was
  **judge:** INCORRECT — Says My Darkest Days, not Human Drama

- **Q:** Who is the fictional head of a British Secret Service division and for which a one-time missionary was the inspiration for?
  **gold:** {"answer": "Q"}
  **model:** M (James Bond's boss) was inspired by a one-time missionary named Dusan "Dusko" Popov.  Answer is: M.  Next question.  Go to next question.  Go to next question.  Go to next question.  Go to next question.  Go to next question
  **judge:** INCORRECT — Answers M, reference is Q

- **Q:** Temagami-Lorrain Mine and Burwash Mine were both mines created during the early 1900s; which one yielded a 16-ton bulk sample of gold ore in 1935?
  **gold:** {"answer": "Burwash Mine"}
  **model:** Temagami-Lorrain Mine. 

Explanation: The Temagami-Lorrain Mine was a mine in the Canadian Shield that produced a 16-ton bulk sample of gold ore in 1935. The Burwash Mine, on the other hand, was a mine in the Yukon that produced a 16
  **judge:** INCORRECT — Says Temagami-Lorrain, not Burwash Mine

- **Q:** What is older, Downfall or Dark Tower?
  **gold:** {"answer": "Downfall"}
  **model:** Dark Tower is older. The first book in the series, The Gunslinger, was published in 1982. Downfall, the last book in the series, was published in 2011. The series spans 8 books and 3 companion books. The series was written by Stephen King. The series is a
  **judge:** INCORRECT — Claims Dark Tower older, contradicts reference

- **Q:** What profession does Nora Roberts and Elizabeth Hardwick have in common?
  **gold:** {"answer": "novelist"}
  **model:** Both are authors.  Nora Roberts is a romance novelist and Elizabeth Hardwick was a poet and essayist.  They are both authors.  They are both writers.  They are both authors.  They are both writers.  They are both authors.  They are both writers.  They are both authors.
  **judge:** INCORRECT — Says authors, contradicts by calling Hardwick essayist

- **Q:** Which animal-themed Disney classic film appeared first: That Darn Cat! or Perri?
  **gold:** {"answer": "That Darn Cat!"}
  **model:** Perri.  Perri was released in 1957, while That Darn Cat! was released in 1965.  Perri is an animated film, while That Darn Cat! is a live-action film.  Perri is based on a children's book of the same name, while That D
  **judge:** INCORRECT — Says Perri came first, contradicts reference

- **Q:** Which film was Oscar nominated, LaLee's Kin: The Legacy of Cotton or Gimme Shelter, the 1970 Rolling Stones documentary?
  **gold:** {"answer": "LaLee's Kin: The Legacy of Cotton"}
  **model:** Gimme Shelter, the 1970 Rolling Stones documentary.  LaLee's Kin: The Legacy of Cotton was nominated for an Oscar in 2007.  Gimme Shelter was nominated for an Oscar in 1971.  Gimme Shelter was nominated for Best Original Song for "Love Street" and Best Sound
  **judge:** INCORRECT — Model picked Gimme Shelter, reference says LaLee's Kin

- **Q:** Which genus contains more species, Crowea or Dracula?
  **gold:** {"answer": "Dracula"}
  **model:** Crowea.  #Crowea #Dracula #plants #botany #species #taxonomy

The final answer is: Crowea.  I hope it is correct.  Thank you for your understanding!  I'll be happy to help you with any other questions you might have.  Please let me know if
  **judge:** INCORRECT — Model says Crowea, reference is Dracula

- **Q:** What is the name of the largest shipping facility in the Northeastern United States, where the Boston Red Socks play?
  **gold:** {"answer": "Port of Boston"}
  **model:** The largest shipping facility in the Northeastern United States is the Port of Boston, but it is not where the Boston Red Sox play. The Boston Red Sox play at Fenway Park. The Port of Boston is a major shipping facility, but it is not the largest in the Northeastern United States. The largest shippi
  **judge:** INCORRECT — Self-contradicts, denies Port of Boston is largest

- **Q:** 12 Years a Slave starred what British actor born 10 July 1977)
  **gold:** {"answer": "Chiwetel Ejiofor"}
  **model:** Chiwetel Ejiofor was not the British actor born 10 July 1977 who starred in 12 Years a Slave. The correct answer is Benedict Cumberbatch was not in the film either. The correct answer is Chiwetel Ejiofor was not in the film either. The correct answer
  **judge:** INCORRECT — Denies Chiwetel Ejiofor, contradicts reference

- **Q:** Lenroy Thomas is the current commonwealth heavyweight champion after beating a British , professional boxer who has a record of 12 wins, 3 defeats, and how many draws?
  **gold:** {"answer": "1"}
  **model:** 2

Explanation: The question states that the boxer has a record of 12 wins, 3 defeats, and 2 draws. Therefore, the answer is 2. The question does not ask for any additional information, so there is no need to provide any further details. The answer is simply 2.
  **judge:** INCORRECT — Answered 2 draws, reference is 1

- **Q:** The world's greatest Super-Heroes anthology showcased one of four superheroes known for speaking the phrase "SHAZAM", what was their name?
  **gold:** {"answer": "Captain Marvel"}
  **model:** Billy Batson.  He is the alter ego of Captain Marvel.  He is a teenager who can transform into a powerful adult superhero by saying the magic word "SHAZAM".  The word is an acronym for the names of six ancient gods and heroes: Solomon, Hercules, Atlas, Zeus, Achilles, and
  **judge:** INCORRECT — Commits to Billy Batson, not Captain Marvel

