import re
import time
import itertools
import numpy as np
from collections import Counter

# For pretty-printing
import pandas as pd
from IPython.display import display, HTML
import jinja2

import nltk
from nltk.corpus import stopwords

def flatten(list_of_lists):
    """Flatten a list-of-lists into a single list."""
    return list(itertools.chain.from_iterable(list_of_lists))

HIGHLIGHT_BUTTON_TMPL = jinja2.Template("""
<script>
colors_on = true;
function color_cells() {
  var ffunc = function(i,e) {return e.innerText {{ filter_cond }}; }
  var cells = $('table.dataframe').children('tbody')
                                  .children('tr')
                                  .children('td')
                                  .filter(ffunc);
  if (colors_on) {
    cells.css('background', 'white');
  } else {
    cells.css('background', '{{ highlight_color }}');
  }
  colors_on = !colors_on;
}
$( document ).ready(color_cells);
</script>
<form action="javascript:color_cells()">
<input type="submit" value="Toggle highlighting (val {{ filter_cond }})"></form>
""")

RESIZE_CELLS_TMPL = jinja2.Template("""
<script>
var df = $('table.dataframe');
var cells = df.children('tbody').children('tr')
                                .children('td');
cells.css("width", "{{ w }}px").css("height", "{{ h }}px");
</script>
""")

def render_matrix(M, rows=None, cols=None, dtype=float,
                        min_size=30, highlight=""):
    html = [pd.DataFrame(M, index=rows, columns=cols,
                         dtype=dtype)._repr_html_()]
    if min_size > 0:
        html.append(RESIZE_CELLS_TMPL.render(w=min_size, h=min_size))

    if highlight:
        html.append(HIGHLIGHT_BUTTON_TMPL.render(filter_cond=highlight,
                                             highlight_color="yellow"))

    return "\n".join(html)
    
def pretty_print_matrix(*args, **kwargs):
    """Pretty-print a matrix using Pandas.
    Optionally supports a highlight button, which is a very, very experimental
    piece of messy JavaScript. It seems to work for demonstration purposes.
    Args:
      M : 2D numpy array
      rows : list of row labels
      cols : list of column labels
      dtype : data type (float or int)
      min_size : minimum cell size, in pixels
      highlight (string): if non-empty, interpreted as a predicate on cell
      values, and will render a "Toggle highlighting" button.
    """
    html = render_matrix(*args, **kwargs)
    display(HTML(html))


def pretty_timedelta(fmt="%d:%02d:%02d", since=None, until=None):
    """Pretty-print a timedelta, using the given format string."""
    since = since or time.time()
    until = until or time.time()
    delta_s = until - since
    hours, remainder = divmod(delta_s, 3600)
    minutes, seconds = divmod(remainder, 60)
    return fmt % (hours, minutes, seconds)


##
# Word processing functions
def canonicalize_digits(word):
    if any([c.isalpha() for c in word]): return word
    word = re.sub("\d", "DG", word)
    if word.startswith("DG"):
        word = word.replace(",", "") # remove thousands separator
    return word

def canonicalize_word(word, wordset=None, digits=True):
    word = word.lower()
    if digits:
        if (wordset != None) and (word in wordset): return word
        word = canonicalize_digits(word) # try to canonicalize numbers
    if (wordset == None) or (word in wordset): return word
    else: return "<unk>" # unknown token

def canonicalize_words(words, **kw):
    return [canonicalize_word(word, **kw) for word in words]

##
# Data loading functions
import nltk
import vocabulary

def get_corpus(name="brown"):
    return nltk.corpus.__getattr__(name)

def sents_to_tokens(sents, vocab):
    """Returns an flattened list of the words in the sentences."""
    # This will canonicalize words, and replace anything not in vocab with <unk>
    return np.array([canonicalize_word(w, wordset=vocab.wordset)
                     for w in flatten(sents)], dtype=object)

# Returns a feature matrix that contains counts for each
# word in sentence.
def sents_to_counts(sents, vocab):
    feature_matrix = np.zeros(shape=(len(sents),vocab.size),
                             dtype=float)
    for i, s in enumerate(sents):
        # Canonicalize words first.
        canon_sent = [canonicalize_word(w) for w in s]
        unigram_counts = Counter(canon_sent)
        for k, v in unigram_counts.items():
            feature_matrix[i,vocab.word_to_id.get(k, vocab.UNK_ID)] = v
    return feature_matrix

# Converts words in each sentence to tokens.
def tokenize_sents(sents, vocab):
    tokenized_sents = []
    for s in sents:
        # Canonicalize words first.
        sent_tokens = [vocab.word_to_id.get(canonicalize_word(w), vocab.UNK_ID) for w in s]
        tokenized_sents.append(sent_tokens)
    return tokenized_sents

# Pads sentences where lenth is < max_length and truncates
# too long sentences to max_length so that all sentences
# are the same length.
def pad_sents(sents, max_length):
    sents_p = []
    for s in sents:
        if(len(s) <= max_length):
            sents_p.append(np.pad(s, pad_width=(0,max_length-len(s)),mode='constant',constant_values=0))
        else:
            sents_p.append(s[0:max_length])
    return np.array(sents_p)

def build_vocab(corpus, V=10000):
    token_feed = (canonicalize_word(w) for w in corpus.words())
    vocab = vocabulary.Vocabulary(token_feed, size=V)
    return vocab

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    return tokens

def get_train_test_sents(corpus, labels, norm_labels, split=0.7, shuffle=True):
    """Get train and test sentences.
    Args:
      corpus: nltk.corpus that supports sents() function
      split (double): fraction to use as training set
      shuffle (int or bool): seed for shuffle of input data, or False to just
      take the training data as the first xx% contiguously.
    Returns:
      train_sentences, test_sentences ( list(list(string)) ): the train and test
      splits
      train_labels, test_labels: the train and test splits
    """
    sentences = np.array([tokenize(s) for s in corpus], dtype=object)
    fmt = (len(sentences), sum(map(len, sentences)))
    print "Loaded %d sentences (%g tokens)" % fmt

    if shuffle:
        rng = np.random.RandomState(shuffle)
        rng.shuffle(sentences)  # in-place
    split_idx = int(split * len(sentences))
    train_sentences = sentences[:split_idx]
    test_sentences = sentences[split_idx:]
    train_labels = labels[:split_idx]
    test_labels = labels[split_idx:]
    train_norm_labels = norm_labels[:split_idx]
    test_norm_labels = norm_labels[split_idx:]

    fmt = (len(train_sentences), sum(map(len, train_sentences)))
    print "Training set: %d sentences (%d tokens)" % fmt
    fmt = (len(test_sentences), sum(map(len, test_sentences)))
    print "Test set: %d sentences (%d tokens)" % fmt

    return train_sentences, test_sentences, train_labels, test_labels, train_norm_labels, test_norm_labels

##
# Use this function
def batch_generator(ids, batch_size, max_time):
    """Convert ids to data-matrix form."""
    # Clip to multiple of max_time for convenience
    clip_len = ((len(ids)-1) / batch_size) * batch_size
    input_w = ids[:clip_len]     # current word
    target_y = ids[1:clip_len+1]  # next word
    # Reshape so we can select columns
    input_w = input_w.reshape([batch_size,-1])
    target_y = target_y.reshape([batch_size,-1])

    # Yield batches
    for i in xrange(0, input_w.shape[1], max_time):
	yield input_w[:,i:i+max_time], target_y[:,i:i+max_time]