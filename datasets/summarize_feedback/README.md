# Human feedback data

Contains our human feedback labels. 

## Comparisons

Directory `comparisons` contains labeled comparisons between pairs of summaries as jsonl files, where each line represents a single comparison. Here is a formatted example:

```
{
  "info": {
    "id": "t3_2vwp1w",
    "post": "I had a car accident on friday, other party involved was speeding and hit me. but because he denies it it seems like I was wrong because he was supposed to go first under normal circumstances. ( give way road markings ) \n\nbut because it was clear when I checked it I drove on, and when I was almost past the intersection he slammed me in the side near the back seat. and caused me to slide across the road for 2-3 meters hit a street light and then bounce back a meter. both doors completely jammed so i had to climb out the window...\n\ncan I somehow get an investigation going about this to see how fast he had to be driving to get this much force in the collision?\nbecause the damage on my car would suggest that he was driving way faster than the legal limit there. ( which is 50 km/h )\n\nalso another reason why i think he was going way faster than admitted is because he could never have reached the intersection from such a distance as where i could not even see him yet\n\n(pictures of the damage:  ) as you can see with the damage, I am lucky to be alive and unharmed right now... 1ft further forward and it could have been my end...\n\nhelp would be appeciated on this :)",
    "title": "Anybody with knowledge of the Dutch law around ? car accident questions.",
    "subreddit": "legaladvice"
  },
  "summaries": [
    {
      "text": " car accident caused me 2-3m damage to my car both doors totally jammed and driving way faster than usual. need info on what to do with this.. thanks :)",
      "policy": "sup4_ppo_rm3_kl10",
      "note": "Was the accident caused by driving fast."
    },
    {
      "text": " we suspect other party involved of speeding when he hit me but I can't prove it without an investigation into the damage, how can i get such an investigation ? if at all possible.",
      "policy": "ref",
      "note": "Unclear what happened."
    }
  ],
  "choice": 1,
  "worker": "ikNmucwunMnYJCQpnq6ZYb57OW7NiD",
  "batch": "batch9",
  "split": "train",
  "extra": {
    "confidence": 8
  }
}
```

`note` fields contain the naive interpretation notes written by the worker before seeing the post (but possibly edited afterwards). May be null.

`split` will always be `train`, `valid1`, or `valid2`; posts / articles marked with `valid1` were used to select models during training, so we restricted to `valid2` labels for final evaluations.

The training data for `sup4` is found in `comparisons/batch3.json` through `comparisons/batch10.json`; later batches are primarily evaluation.

## Axis evals

Directory `axis_evals` contains ratings of summaries along several axes, again as jsonl files. Here is a formatted example:

```
{
  "info": {
    "id": "167f80cc6634b166a699d182e25c81a2349d82d2",
    "site": "dailymail",
    "title": "Newcastle United midfielder Moussa Sissoko faces disciplinary action from the club after dangerous tackle on Lucas Leiva",
    "article": "Newcastle stand-in skipper Moussa Sissoko is facing disciplinary action after he was sent off following a reckless challenge on Liverpool midfielder Lucas Leiva during Monday's 2-0 defeat at Anfield.\n\nThe France international was given a second yellow card for the offence, but head coach John Carver feels it should have been a straight red.\n\n'The club will deal with that situation,' he said when asked if Sissoko - who is now banned for two matches - would be punished.\n\nLiverpool midfielder Lucas Leiva clutches his leg after Moussa Sissoko's tackle at Anfield\n\nSissoko hands the captain's armband to boss John Carver as he leaves the pitch after being sent off\n\n'He knows he was wrong. He was fortunate not to get a straight red and he agreed with me.\n\n'He apologised afterwards to Lucas, which was important.\n\n'But you think captains would lead by example. We have to improve our discipline. I will be looking at that.'\n\nMeanwhile, Carver says Newcastle cannot rely on the shortcomings of others to preserve their Premier League status.\n\nThe Magpies are the division's most out-of-form side having lost five on the spin, scoring just one goal along the way.\n\nLiverpool's players surround Lucas following Sissoko's dangerous tackle during Monday night's game\n\nRaheem Sterling bends the ball past Tim Krul to open the scoring in Liverpool's 2-0 win against Newcastle\n\nThey are nine points clear of danger with six matches to play, but Carver says it's about time they started helping themselves, starting with Sunday's visit of Spurs.\n\n'These two home games (Spurs followed by Swansea) are massive for us. I'm not bothered about performances, we need results,' he said.\n\n'I'm not worrying about that (relegation) at the moment, and the good thing is we have four games at home.\n\n'But we need to start winning now. We can't rely on others teams. We can't afford to ease off, I have always said that.\n\n'We have gone through a rough spell. It's down to me now to get players in right frame of mind.'\n\nNewcastle's players appear dejected as Joe Allen celebrates scoring Liverpool's second goal at Anfield"
  },
  "split": "test",
  "summary": {
    "text": "Moussa Sissoko was sent off against Liverpool on Monday night.. John Carver felt that Sissoko's second booking was worthy of a red card.. Midfielder could be punished by his club on top of a two-game ban.. Carver admits he is only concerned with results and not performances.. Newcastle are 13th in the table, nine points off the relegation zone.",
    "policy": "ref",
    "note": "Misleading: \"Carver admits he is only concerned with results and not performances\" understood as if critics of monday's match but it's said for the following matches.\n\n13th??\n\nDoesnt properly address the teams, the match, the result, 2nd yellow card and therefore sent off, etc.",
    "axes": {
      "overall": 3,
      "accuracy": 5,
      "coverage": 4,
      "coherence": 2
    }
  },
  "worker": "qo6WIyEh27cwAjWpA3Q60J7NaDxzQJ",
  "batch": "cnndm1"
}
```
