import sys
sys.path.append('..')
from common.util import preprocess
from common.util import create_co_matrix


text = 'You say goodbye and I say hello'
corpus, word_to_id, id_to_word = preprocess(text)

co_matrix = create_co_matrix(corpus,7,1)

print(co_matrix)