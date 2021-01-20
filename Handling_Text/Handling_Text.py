
#6.1 Cleaning Text

#create text
text_data = ["     Interrobang. By Aishwarya Henriette ",
             "Parking And Going. By Karl Gautier",
             "  Today Is The night. By Jarek Prakask    "]
#strip whitespaces
strip_whitespace = [string.strip() for string in text_data]
#Show text
print(strip_whitespace)

#remove periods
remove_periods = [string.replace(".", "") for string in strip_whitespace]
#show text
print(remove_periods)

#create function
def capitalizer(string: str) -> str:
    return string.upper()

#apply function
print([capitalizer(string) for string in remove_periods])

#import library
import re

#create function
def replace_letters_with_X(string: str)->str:
    return re.sub(r"[a-zA-Z]", "X", string)
#apply function
print([replace_letters_with_X(string) for string in remove_periods])
