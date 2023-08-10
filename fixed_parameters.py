##### List of parameters used that are not subject to change #####

### FILES ###
# ES 
ES_dev_in = open("ES/dev.in","r",encoding="utf-8").read() 
ES_dev_out = open("ES/dev.out","r",encoding="utf-8").read() 
ES_train = open("ES/train","r",encoding="utf-8").read() 
ES_dev_p2_out = open("ES/dev.p2.out","w",encoding="utf-8")
# RU 
RU_dev_in = open("RU/dev.in","r",encoding="utf-8").read() 
RU_dev_out = open("RU/dev.out","r",encoding="utf-8").read() 
RU_train = open("RU/train","r",encoding="utf-8").read() 

### TAGS ###
NUMBER_OF_TAGS = 7

B_Positive = "B-positive"
B_Negative = "B-negative"
B_Neutral = "B-neutral"
I_Positive = "I-positive"
I_Negative = "I-negative"
I_Neutral = "I-neutral"
Outside = "O"

TAGS = (B_Positive, B_Negative, B_Neutral, I_Positive, I_Negative, I_Neutral, Outside)

### WORDS ###
Invalid_Word = "#UNK#"