from builtins import range
from builtins import object
import numpy as np

from deeplearning.layers import *
from deeplearning.rnn_layers import *


class CaptioningRNN(object):
    """
    A CaptioningRNN produces captions from image features using a recurrent
    neural network.

    The RNN receives input vectors of size D, has a vocab size of V, works on
    sequences of length T, has an RNN hidden dimension of H, uses word vectors
    of dimension W, and operates on minibatches of size N.

    Note that we don't use any regularization for the CaptioningRNN.
    """

    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128,
                 hidden_dim=128, cell_type='rnn', dtype=np.float32):
        """
        Construct a new CaptioningRNN instance.

        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries,
          and maps each string to a unique integer in the range [0, V).
        - input_dim: Dimension D of input image feature vectors.
        - wordvec_dim: Dimension W of word vectors.
        - hidden_dim: Dimension H for the hidden state of the RNN.
        - cell_type: What type of RNN to use; either 'rnn' or 'lstm'.
        - dtype: numpy datatype to use; use float32 for training and float64 for
          numeric gradient checking.
        """
        if cell_type not in {'rnn', 'lstm'}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.dtype = dtype
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.params = {}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx['<NULL>']
        self._start = word_to_idx.get('<START>', None)
        self._end = word_to_idx.get('<END>', None)

        # Initialize word vectors
        self.params['W_embed'] = np.random.randn(vocab_size, wordvec_dim)
        self.params['W_embed'] /= 100

        # Initialize CNN -> hidden state projection parameters
        self.params['W_proj'] = np.random.randn(input_dim, hidden_dim)
        self.params['W_proj'] /= np.sqrt(input_dim)
        self.params['b_proj'] = np.zeros(hidden_dim)

        # Initialize parameters for the RNN
        dim_mul = {'lstm': 4, 'rnn': 1}[cell_type]
        self.params['Wx'] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)
        self.params['Wx'] /= np.sqrt(wordvec_dim)
        self.params['Wh'] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
        self.params['Wh'] /= np.sqrt(hidden_dim)
        self.params['b'] = np.zeros(dim_mul * hidden_dim)

        # Initialize output to vocab weights
        self.params['W_vocab'] = np.random.randn(hidden_dim, vocab_size)
        self.params['W_vocab'] /= np.sqrt(hidden_dim)
        self.params['b_vocab'] = np.zeros(vocab_size)

        # Cast parameters to correct dtype
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)


    def loss(self, features, captions):
        """
        Compute training-time loss for the RNN. We input image features and
        ground-truth captions for those images, and use an RNN (or LSTM) to compute
        loss and gradients on all parameters.

        Inputs:
        - features: Input image features, of shape (N, D)
        - captions: Ground-truth captions; an integer array of shape (N, T) where
          each element is in the range 0 <= y[i, t] < V

        Returns a tuple of:
        - loss: Scalar loss
        - grads: Dictionary of gradients parallel to self.params
        """
        # Cut captions into two pieces: captions_in has everything but the last word
        # and will be input to the RNN; captions_out has everything but the first
        # word and this is what we will expect the RNN to generate. These are offset
        # by one relative to each other because the RNN should produce word (t+1)
        # after receiving word t. The first element of captions_in will be the START
        # token, and the first element of captions_out will be the first word.
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        # You'll need this
        mask = (captions_out != self._null)

        # Weight and bias for the affine transform from image features to initial
        # hidden state
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']

        # Word embedding matrix
        W_embed = self.params['W_embed']

        # Input-to-hidden, hidden-to-hidden, and biases for the RNN
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']

        # Weight and bias for the hidden-to-vocab transformation.
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        loss, grads = 0.0, {}

        h0 = np.dot(features,W_proj)
        h0+=b_proj
    
    
        captions_in_embedded, cache_embed = word_embedding_forward(captions_in, W_embed)
    
        if self.cell_type == 'rnn':
        
            h,cache_rnn = rnn_forward(captions_in_embedded, h0, Wx, Wh, b)
            scores, cache_aff = temporal_affine_forward(h, W_vocab, b_vocab)
            loss,dout = temporal_softmax_loss(scores, captions_out, mask, verbose=False)
        
        
            dh,dW_vocab,db_vocab = temporal_affine_backward(dout, cache_aff)
            dx, dh0, dWx, dWh, db = rnn_backward(dh, cache_rnn)
        elif self.cell_type == 'lstm':
        
            h,cache_lstm = lstm_forward(captions_in_embedded, h0, Wx, Wh, b)
            scores, cache_aff = temporal_affine_forward(h, W_vocab, b_vocab)
            loss,dout = temporal_softmax_loss(scores, captions_out, mask, verbose=False)
        
        
            dh,dW_vocab,db_vocab = temporal_affine_backward(dout, cache_aff)
            dx, dh0, dWx, dWh, db = lstm_backward(dh, cache_lstm)
        
        dW_embed = word_embedding_backward(dx, cache_embed)
        dW_proj = np.dot(features.T,dh0)
        
        grads['Wx'] = dWx
        grads['Wh'] = dWh
        grads['b'] = db
        grads['W_embed'] = dW_embed
        grads['W_vocab'] = dW_vocab
        grads['b_vocab'] = db_vocab
        grads['W_proj'] = dW_proj
        grads['b_proj'] = np.sum(dh0,axis=0)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


    def sample(self, features, max_length=30):
        """
        Run a test-time forward pass for the model, sampling captions for input
        feature vectors.

        At each timestep, we embed the current word, pass it and the previous hidden
        state to the RNN to get the next hidden state, use the hidden state to get
        scores for all vocab words, and choose the word with the highest score as
        the next word. The initial hidden state is computed by applying an affine
        transform to the input image features, and the initial word is the <START>
        token.

        For LSTMs you will also have to keep track of the cell state; in that case
        the initial cell state should be zero.

        Inputs:
        - features: Array of input image features of shape (N, D).
        - max_length: Maximum length T of generated captions.

        Returns:
        - captions: Array of shape (N, max_length) giving sampled captions,
          where each element is an integer in the range [0, V). The first element
          of captions should be the first sampled word, not the <START> token.
        """
        N = features.shape[0]
        captions = self._null * np.ones((N, max_length), dtype=np.int32)

        # Unpack parameters
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
        W_embed = self.params['W_embed']
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        ###########################################################################
        ###########################################################################
        h0 = np.dot(features, W_proj) + b_proj

        captions[:, 0] = self._start
        prev_h = h0  # Previous hidden state
        prev_c = np.zeros_like(h0)  # Previous cell state
        # Current word (start word)
        capt = self._start * np.ones((N, 1), dtype=np.int32)

        for t in range(max_length):  # Let's go over the sequence

            word_embed, _ = word_embedding_forward(
                capt, W_embed)  # Embedded current word
            if self.cell_type == 'rnn':
                # Run a step of rnn
                h, _ = rnn_step_forward(np.squeeze(
                    word_embed), prev_h, Wx, Wh, b)
            elif self.cell_type == 'lstm':
                # Run a step of lstm
                h, c, _ = lstm_step_forward(np.squeeze(
                    word_embed), prev_h, prev_c, Wx, Wh, b)
            else:
                raise ValueError('%s not implemented' % (self.cell_type))

            # Compute the score distrib over the dictionary
            scores, _ = temporal_affine_forward(
                h[:, np.newaxis, :], W_vocab, b_vocab)
            # Squeeze unecessari dimension and get the best word idx
            idx_best = np.squeeze(np.argmax(scores, axis=2))
            # Put it in the captions
            captions[:, t] = idx_best

            # Update the hidden state, the cell state (if lstm) and the current
            # word
            prev_h = h
            if self.cell_type == 'lstm':
                prev_c = c
            capt = captions[:, t]

        return captions
