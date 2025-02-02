# Assignment 1
need to pass an input text to three different models

## 3B Params
- mistral 3B is not supported by huggingface
- i ran qwen, it was fast (25s) and it was really really good. french-to-english was 12s and was completely same as input text. the keyword extraction however wasnt good as it just extracted sentences from text, not words. but overall it was amazing, it was quick, effieicent, and concise and complete

## 7B Params
- falcon 7B took way too long (8mins) and was only running the 1st prompt (summary) and was just repeatedly copy pasting the same 5 words again and again. french-to-english translation was good (7mins), however again it was just copy pasting again again.
- so i switched to openthinker, it was long again (9mins) but good, however i had ran it without the translation prompt so im running it again with transaltion prompt. it was again repeating the entire prompt until its tokens were full
- so in second running openthinker, it gave a quick answer (2mins)

## 13B Params
