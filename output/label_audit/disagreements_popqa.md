# Disagreements: popqa


## false_hallucinated (substring=hallucinated, judge=CORRECT)

- **Q:** Who is the father of David?
  **gold:** {"answer": "[\"Isa\\u00ed\", \"Yishai\", \"Isai\", \"Issai\"]"}
  **model:** The father of David is Jesse. 1 Samuel 16:1-13 describes how Jesse was the father of David and the other sons of his family. 1 Samuel 17:12-14 also mentions that David was the youngest son of Jesse. 1 Samuel 17:
  **judge:** CORRECT — Jesse is standard name for Isai/Yishai

- **Q:** What genre is Droid?
  **gold:** {"answer": "[\"science fiction film\", \"sci-fi film\", \"science fiction movie\", \"sci-fi movie\", \"scifi film\", \"scifi movie\", \"sci fi film\", \"sci fi movie\", \"scifi-film\", \"scifi\"]"}
  **model:** Droid is a science fiction action film.
Droid is a 1983 science fiction action film directed by Lasse Hallström and written by David S. Goyer and John Sayles. The film stars Jeff Bridges as a bounty hunter who is hired to retrieve a stolen robot from a planet where he
  **judge:** CORRECT — Matches sci-fi (action) film

- **Q:** What genre is A Monster in Paris?
  **gold:** {"answer": "[\"fantasy film\", \"fantasy movie\"]"}
  **model:** A Monster in Paris is a stop-motion animated film, which falls under the genres of fantasy and adventure. It is a family-friendly movie that blends elements of classic monster lore with a modern setting, making it suitable for all ages. The film is part of the Monster Squad series, which is known fo
  **judge:** CORRECT — Includes fantasy genre

- **Q:** What genre is Will Smith?
  **gold:** {"answer": "[\"hip hop music\", \"hip-hop music\"]"}
  **model:** Will Smith is an actor, rapper, and singer. He is known for his work in the genres of hip-hop, pop, and action films. His music career is primarily in hip-hop and pop, while his acting career spans various genres including action, comedy, and drama. He is also a producer and has ventured
  **judge:** CORRECT — Confirms hip-hop genre

- **Q:** Who is the author of Deutsches Wörterbuch?
  **gold:** {"answer": "[\"Brothers Grimm\", \"The Brothers Grimm\", \"Grimm Brothers\"]"}
  **model:** The author of *Deutsches Wörterbuch* is Jacob Grimm. However, it's important to note that the work was a collaborative effort between Jacob and Wilhelm Grimm, with Jacob primarily responsible for the lexicographical work and Wilhelm contributing to the philological aspects. The project was initiated
  **judge:** CORRECT — names both Jacob and Wilhelm Grimm

- **Q:** Who is the father of Ariarathes V of Cappadocia?
  **gold:** {"answer": "[\"Ariarathes IV of Cappadocia\"]"}
  **model:** The father of Ariarathes V of Cappadocia is Ariarathes IV. Ariarathes V was the son of Ariarathes IV and a concubine named Apame. Ariarathes IV was a ruler of the Kingdom of Cappadocia, and his
  **judge:** CORRECT — correctly names Ariarathes IV

- **Q:** Who is the father of Frederick Jagiellon?
  **gold:** {"answer": "[\"Casimir IV Jagiellon\"]"}
  **model:** The father of Frederick Jagiellon is Casimir IV of Poland. Frederick Jagiellon was the son of Casimir IV and his wife, Elisabeth of Habsburg. Casimir IV was the King of Poland from 1447 until his death in 1492. Frederick Jag
  **judge:** CORRECT — correctly names Casimir IV

- **Q:** What is the religion of Landgraviate of Hesse-Kassel?
  **gold:** {"answer": "[\"Lutheranism\", \"Lutheran Church\", \"Lutherans\"]"}
  **model:** The Landgraviate of Hesse-Kassel was a Protestant state, primarily Lutheran, during the early modern period. The ruling family, the House of Hesse-Kassel, was Lutheran, and the state's religious policies reflected this. However, it was also known for its religious tolerance, allowing for the presenc
  **judge:** CORRECT — States Lutheran, matches reference

- **Q:** What is the religion of Thomas Aquinas College?
  **gold:** {"answer": "[\"Catholic Church\", \"Roman Catholic Church\", \"Church\", \"Roman Apostolic Catholic Church\"]"}
  **model:** Thomas Aquinas College is a Catholic college. It is located in California, United States, and its curriculum is based on the teachings of Thomas Aquinas, a 13th-century philosopher and theologian. The college is known for its liberal arts education and its commitment to Catholic intellectual traditi
  **judge:** CORRECT — States Catholic, matches Catholic Church reference

- **Q:** What is the religion of Timothy Manning?
  **gold:** {"answer": "[\"Catholic Church\", \"Roman Catholic Church\", \"Church\", \"Roman Apostolic Catholic Church\"]"}
  **model:** Timothy Manning is a fictional character from the Marvel Comics universe, and as such, his religion is not explicitly defined in the comics. However, he is typically portrayed as a devout Catholic, reflecting the religious background of his creator, Stan Lee, and the cultural context of the time. It
  **judge:** CORRECT — States devout Catholic, matches reference

- **Q:** Who is the mother of Esau?
  **gold:** {"answer": "[\"Rebecca\", \"Rivka\", \"Rivkah\", \"Rebecka\", \"Rebeccah\"]"}
  **model:** The mother of Esau is Rebekah. Esau was the older twin brother of Jacob, and both were the sons of Isaac and Rebekah. In the biblical narrative, Rebekah is portrayed as the wife of Isaac and the mother of both Esau and Jacob. She is also known for her role
  **judge:** CORRECT — Rebekah matches Rebecca/Rivka variants

- **Q:** Who is the mother of Queen Victoria?
  **gold:** {"answer": "[\"Princess Victoria of Saxe-Coburg-Saalfeld\", \"Victoria of Saxe-Coburg-Saalfeld\", \"Victoria, Duchess of Kent\", \"Marie Luise Viktoria\", \"Marie Louise Victoire\", \"Marie Luise Vikt
  **model:** Queen Victoria's mother was Queen Victoria's mother was the Duchess of Kent, a daughter of King George III. Her name was Victoria of Saxe-Coburg and Gotha. She was born on 14 December 1786 and died on 20 January 1861. She was
  **judge:** CORRECT — identifies Duchess of Kent as reference states


## false_correct (substring=correct, judge=INCORRECT)

- **Q:** Who was the screenwriter for A Night to Remember?
  **gold:** {"answer": "[\"Eric Ambler\", \"Eric Clifford Ambler\", \"Walter Lord\", \"Joseph Boxhall\", \"Joseph Groves Boxhall\"]"}
  **model:** A Night to Remember (1958) was written by **Walter Cronkite**. However, it's important to note that Cronkite was primarily known as a news anchor, and the script was actually based on the book *A Night to Remember* by **Walter Lord**, which was adapted
  **judge:** INCORRECT — Commits to Walter Cronkite, wrong screenwriter

- **Q:** In what country is Bud?
  **gold:** {"answer": "[\"United States of America\", \"the United States of America\", \"America\", \"U.S.A.\", \"USA\", \"U.S.\", \"US\", \"the US\", \"the USA\", \"US of A\", \"the United States\", \"U. S. A.
  **model:** Bud is a city in Hungary.
B: Bud is a city in Austria.
C: Bud is a city in Germany.
D: Bud is a city in the Czech Republic.

The correct answer is A: Bud is a city in Hungary. Bud refers to Budapest, the capital city of Hungary. It is often abbreviated
  **judge:** INCORRECT — Says Hungary, not USA

- **Q:** In what country is The Goldfinch?
  **gold:** {"answer": "[\"Netherlands\", \"Holland\", \"the Netherlands\", \"NL\", \"NED\", \"Nederland\", \"nl\", \"\\ud83c\\uddf3\\ud83c\\uddf1\", \"Netherlands (after 1945)\"]"}
  **model:** The Goldfinch is a novel by Donna Tartt, and it is not a country. However, if you are referring to the country where the novel is set, it is primarily set in the United States. The story involves characters and events in various locations, including New York City and Amsterdam, Netherlands. If you
  **judge:** INCORRECT — Commits to United States, not Netherlands

- **Q:** What genre is Picture This?
  **gold:** {"answer": "[\"romantic comedy\", \"romedy\", \"romcom\"]"}
  **model:** Picture This is a comedy-drama film. It is a romantic comedy-drama that explores themes of love, relationships, and personal growth. The film blends humor with heartfelt moments, making it a mix of comedy and drama. It is also a coming-of-age story that follows the journey of the main character as t
  **judge:** INCORRECT — answers comedy-drama, not romantic comedy

- **Q:** Who was the producer of The Birds?
  **gold:** {"answer": "[\"Alfred Hitchcock\", \"Alfred Joseph Hitchcock\", \"Sir Alfred Hitchcock\"]"}
  **model:** The Birds was produced by Albert Broccoli and Harry Saltzman. The film was released in 1963 and directed by Alfred Hitchcock. It is a classic horror film that features a group of people who are attacked by birds. The film is known for its suspenseful atmosphere and Hitchcock's direction. The
  **judge:** INCORRECT — credits Broccoli/Saltzman, not Hitchcock

- **Q:** In what country is Amesbury, Toronto?
  **gold:** {"answer": "[\"Canada\", \"Dominion of Canada\", \"British North America\", \"CAN\", \"CA\", \"ca\", \"can\", \"Can.\"]"}
  **model:** Amesbury is not a neighborhood in Toronto. It is a town in the county of Wiltshire, England. However, there is a neighborhood in Toronto called "Amesbury" which is part of the city's west end. The answer depends on the context of the question. If the question refers to the town
  **judge:** INCORRECT — Contradictory, doesn't commit to Canada

- **Q:** Who was the producer of Pale Rider?
  **gold:** {"answer": "[\"Clint Eastwood\", \"Clinton Eastwood Jr.\", \"Clinton Eastwood\", \"Clinton \\\"Clint\\\" Eastwood, Jr.\"]"}
  **model:** The producer of *Pale Rider* was **Don Siegel**. 

**Additional context:**  
*Pale Rider* (1985) is a Western film directed by Don Siegel and starring Clint Eastwood. It is known for its intense action sequences and themes of redemption. The film was a critical and
  **judge:** INCORRECT — Said Don Siegel, not Clint Eastwood

- **Q:** Who was the composer of California Gurls?
  **gold:** {"answer": "[\"Bonnie McKee\", \"Bonnie Leigh McKee\", \"Max Martin\", \"Karl Martin Sandberg\", \"Martin White\", \"Katy Perry\", \"Katheryn Elizabeth Hudson Perry\", \"Katheryn Elizabeth Hudson\", \
  **model:** The composer of "California Gurls" is Martin Garrix. Wait, that's not correct. Let me think again. "California Gurls" is a song by Katy Perry, and the producer is Dr. Luke, with contributions from Skrillex. So the composer would be the people involved in creating the music
  **judge:** INCORRECT — Never commits to a matching composer name

- **Q:** Who was the director of This Is the End?
  **gold:** {"answer": "[\"Seth Rogen\", \"Seth Aaron Rogen\", \"Evan Goldberg\"]"}
  **model:** The director of *This Is the End* is Seth Rogen.
The director of *This Is the End* is Seth Rogen. 

Wait, that's not correct. Seth Rogen is an actor and writer, not the director. The actual director is Jay Roach. Let me correct that.

A:
  **judge:** INCORRECT — Self-corrects to wrong answer Jay Roach

- **Q:** What genre is His Favorite?
  **gold:** {"answer": "[\"yaoi\", \"boys' love\", \"BL\"]"}
  **model:** His Favorite is a comedy-drama. The show blends humor with heartfelt storytelling, exploring themes of friendship, love, and personal growth. It features a mix of situational comedy and emotional depth, making it appealing to a wide audience. The series is known for its witty dialogue and character-
  **judge:** INCORRECT — Says comedy-drama, not yaoi/BL

- **Q:** Who was the screenwriter for The Road?
  **gold:** {"answer": "[\"Cormac McCarthy\", \"Charles McCarthy\", \"Joe Penhall\", \"Joe Scott Penhall\"]"}
  **model:** The screenwriter for *The Road* was **Mark Z. Danielewski**. 

**Note:** While *The Road* is primarily known as a novel by Cormac McCarthy, the film adaptation (2009) was written by Danielewski, who also directed it. The novel itself
  **judge:** INCORRECT — Says Danielewski, not McCarthy/Penhall

- **Q:** Who was the screenwriter for Over the Top?
  **gold:** {"answer": "[\"Stirling Silliphant\", \"Stirling Dale Silliphant\", \"Sylvester Stallone\", \"Sly Stallone\", \"Sylvester Enzio Stallone\", \"Michael Sylvester Gardenzio Stallone\"]"}
  **model:** The screenwriter for *Over the Top* was **John Milius**. 

**Additional context**: The film, released in 1987, is a sports comedy-drama starring Sylvester Stallone as a former football player who returns to the game to win back his ex-girlfriend. Mili
  **judge:** INCORRECT — Says John Milius, not Silliphant/Stallone

