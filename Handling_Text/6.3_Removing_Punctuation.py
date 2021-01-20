
#load libraries
import unicodedata
import sys

#Create text
text_data = ['Hi!!!! I. Love. This. Song....',
             '10000% Agree!!!! #LoveIT',
             'Right?!?!']
# Create a dictionary of punctuation characters
punctuation = dict.fromkeys(i for i in range(sys.maxunicode)
 if unicodedata.category(chr(i)).startswith('P'))

#foreach string, remove any punctuation characters
print([string.translate(punctuation) for string in text_data])