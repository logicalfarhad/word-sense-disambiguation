from word_sense.wordsense import WordSense as ws

if __name__ == "__main__":
    wordSense = ws(glove_file_path='C:/Users/logicalfarhad/Downloads/glove.6B/glove.6B.50d.txt',
                   sentence_to_disambiguate='quick brown fox jumps over the lazy dog', threshold=0.05)
    wordSense.get_successful_replaced_words()
