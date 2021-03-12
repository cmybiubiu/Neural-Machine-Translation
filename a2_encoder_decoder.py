'''
This code is provided solely for the personal and private use of students
taking the CSC401H/2511H course at the University of Toronto. Copying for
purposes other than this use is expressly prohibited. All forms of
distribution of this code, including but not limited to public repositories on
GitHub, GitLab, Bitbucket, or any other online platform, whether as given or
with any changes, are expressly prohibited.

Authors: Sean Robertson, Jingcheng Niu, Zining Zhu, and Mohamed Abdall

All of the files in this directory and all subdirectories are:
Copyright (c) 2021 University of Toronto
'''

'''Concrete implementations of abstract base classes.

You don't need anything more than what's been imported here
'''

import torch

from a2_abcs import EncoderBase, DecoderBase, EncoderDecoderBase

# All docstrings are omitted in this file for simplicity. So please read
# a2_abcs.py carefully so that you can have a solid understanding of the
# structure of the assignment.

class Encoder(EncoderBase):


    def init_submodules(self):
        # Hints:
        # 1. You must initialize these submodules:
        #   self.rnn, self.embedding
        # 2. You will need these object attributes:
        #   self.source_vocab_size, self.word_embedding_size,
        #   self.pad_id, self.dropout, self.cell_type,
        #   self.hidden_state_size, self.num_hidden_layers.
        # 3. cell_type will be one of: ['lstm', 'gru', 'rnn']
        # 4. Relevant pytorch modules: torch.nn.{LSTM, GRU, RNN, Embedding}
        #######
        # if self.cell_type == 'gru':
        #     self.rnn = torch.nn.GRU(self.word_embedding_size, self.hidden_state_size, self.num_hidden_layers,
        #                             bidirectional=True, dropout=self.dropout)
        # elif self.cell_type == 'rnn':
        #     self.rnn = torch.nn.RNN(self.word_embedding_size, self.hidden_state_size, self.num_hidden_layers,
        #                             bidirectional=True, dropout=self.dropout)
        # elif self.cell_type == 'lstm':
        #     self.rnn = torch.nn.LSTM(self.word_embedding_size, self.hidden_state_size, self.num_hidden_layers,
        #                              bidirectional=True, dropout=self.dropout)
        #
        # self.embedding = torch.nn.Embedding(self.source_vocab_size, self.word_embedding_size, padding_idx=self.pad_id)
        #########
        self.embedding = torch.nn.Embedding(self.source_vocab_size,
                                            self.word_embedding_size,
                                            padding_idx=self.pad_id)
        init_packet = [self.word_embedding_size, self.hidden_state_size]
        init_kwargs = {'dropout': self.dropout,
                       'num_layers': self.num_hidden_layers,
                       'bidirectional': True}

        if self.cell_type == 'gru':
            initializer = torch.nn.GRU
        elif self.cell_type == 'rnn':
            initializer = torch.nn.RNN
        elif self.cell_type == 'lstm':
            initializer = torch.nn.LSTM
        else:
            raise ValueError(f"cell type: '{self.cell_type}' not valid.")
        self.rnn = initializer(*init_packet, **init_kwargs)




    def forward_pass(self, F, F_lens, h_pad=0.):
        # Recall:
        #   F is shape (S, M)
        #   F_lens is of shape (M,)
        #   h_pad is a float
        #
        # Hints:
        # 1. The structure of the encoder should be:
        #   input seq -> |embedding| -> embedded seq -> |rnn| -> seq hidden
        # 2. You will need to use these methods:
        #   self.get_all_rnn_inputs, self.get_all_hidden_states
        embedded = self.get_all_rnn_inputs(F)
        output = self.get_all_hidden_states(embedded, F_lens, h_pad)
        return output


    def get_all_rnn_inputs(self, F):
        # Recall:
        #   F is shape (S, M)
        #   x (output) is shape (S, M, I)
        return self.embedding(F)

    def get_all_hidden_states(self, x, F_lens, h_pad):
        # Recall:
        #   x is of shape (S, M, I)
        #   F_lens is of shape (M,)
        #   h_pad is a float
        #   h (output) is of shape (S, M, 2 * H)
        #
        # Hint:
        #   relevant pytorch modules:
        #   torch.nn.utils.rnn.{pad_packed,pack_padded}_sequence

        x = torch.nn.utils.rnn.pack_padded_sequence(x, F_lens, enforce_sorted=False)
        output, _ = self.rnn(x)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, padding_value=h_pad)
        return output

class DecoderWithoutAttention(DecoderBase):
    '''A recurrent decoder without attention'''

    def init_submodules(self):
        # Hints:
        # 1. You must initialize these submodules:
        #   self.embedding, self.cell, self.ff
        # 2. You will need these object attributes:
        #   self.target_vocab_size, self.word_embedding_size, self.pad_id
        #   self.hidden_state_size, self.cell_type.
        # 3. cell_type will be one of: ['lstm', 'gru', 'rnn']
        # 4. Relevant pytorch modules:
        #   torch.nn.{Embedding, Linear, LSTMCell, RNNCell, GRUCell}
        ################
        # if self.cell_type == 'gru':
        #     self.cell = torch.nn.GRUCell(self.word_embedding_size, self.hidden_state_size)
        # elif self.cell_type == 'rnn':
        #     self.cell = torch.nn.RNNCell(self.word_embedding_size, self.hidden_state_size)
        # elif self.cell_type == 'lstm':
        #     self.cell = torch.nn.LSTMCell(self.word_embedding_size, self.hidden_state_size)
        #
        # self.embedding = torch.nn.Embedding(self.target_vocab_size, self.word_embedding_size, padding_idx=self.pad_id)
        #
        # self.ff = torch.nn.Linear(self.hidden_state_size, self.target_vocab_size)

        init_packet = [self.word_embedding_size,
                       self.hidden_state_size]
        if self.cell_type == 'gru':
            initializer = torch.nn.GRUCell
        elif self.cell_type == 'rnn':
            initializer = torch.nn.RNNCell
        elif self.cell_type == 'lstm':
            initializer = torch.nn.LSTMCell
        else:
            raise ValueError(f"cell type: '{self.cell_type}' not valid.")
        self.cell = initializer(*init_packet)
        self.embedding = torch.nn.Embedding(self.target_vocab_size,
                                            self.word_embedding_size,
                                            padding_idx=self.pad_id)
        self.ff = torch.nn.Linear(self.hidden_state_size, self.target_vocab_size)

    def forward_pass(self, E_tm1, htilde_tm1, h, F_lens):
        # Recall:
        #   E_tm1 is of shape (M,)
        #   htilde_tm1 is of shape (M, 2 * H)
        #   h is of shape (S, M, 2 * H)
        #   F_lens is of shape (M,)
        #   logits_t (output) is of shape (M, V)
        #   htilde_t (output) is of same shape as htilde_tm1
        #
        # Hints:
        # 1. The structure of the encoder should be:
        #   encoded hidden -> |embedding| -> embedded hidden -> |rnn| ->
        #   decoded hidden -> |output layer| -> output logits
        # 2. You will need to use these methods:
        #   self.get_current_rnn_input, self.get_current_hidden_state,
        #   self.get_current_logits
        # 3. You can assume that htilde_tm1 is not empty. I.e., the hidden state
        #   is either initialized, or t > 1.
        # 4. The output of an LSTM cell is a tuple (h, c), but a GRU cell or an
        #   RNN cell will only output h.
        xtilde_t = self.get_current_rnn_input(E_tm1, htilde_tm1, h, F_lens)

        htilde_tm1 = self.get_first_hidden_state(h, F_lens)

        htile_t = self.get_current_hidden_state(xtilde_t, htilde_tm1)

        logits_t = self.get_current_logits(htile_t)

        return logits_t

    def get_first_hidden_state(self, h, F_lens):
        # Recall:
        #   h is of shape (S, M, 2 * H)
        #   F_lens is of shape (M,)
        #   htilde_tm1 (output) is of shape (M, 2 * H)
        #
        # Hint:
        # 1. Ensure it is derived from encoder hidden state that has processed
        # the entire sequence in each direction. You will need to:
        # - Populate indices [0: self.hidden_state_size // 2] with the hidden
        #   states of the encoder's forward direction at the highest index in
        #   time *before padding*
        # - Populate indices [self.hidden_state_size//2:self.hidden_state_size]
        #   with the hidden states of the encoder's backward direction at time
        #   t=0
        # 2. Relevant pytorch functions: torch.cat
        ###############
        # f = h[F_lens - 1, torch.arange(F_lens.size(0), device=h.device), :(self.hidden_state_size // 2)]  # forward hidden state
        # b = h[0, :, (self.hidden_state_size // 2):]  # backward hidden state
        #
        # htilde_tm1 = torch.cat([f.squeeze(), b.squeeze()], dim=1)
        # return htilde_tm1

        mid = self.hidden_state_size // 2
        f = h[F_lens - 1, torch.arange(F_lens.size(0), device=h.device), :mid]  # forward hidden state
        b = h[0, :, mid:]  # backward hidden state
        return torch.cat([f.squeeze(), b.squeeze()], dim=1)

    def get_current_rnn_input(self, E_tm1, htilde_tm1, h, F_lens):
        # Recall:
        #   E_tm1 is of shape (M,)
        #   htilde_tm1 is of shape (M, 2 * H) or a tuple of two of those (LSTM)
        #   h is of shape (S, M, 2 * H)
        #   F_lens is of shape (M,)
        #   xtilde_t (output) is of shape (M, Itilde)

        # mask = (E_tm1 != self.pad_id).float().unsqueeze(1)
        # xtilde_t = self.embedding(E_tm1) * mask
        # return xtilde_t

        device = h.device
        mask = torch.where(E_tm1 == torch.tensor([self.pad_id]).to(device),
                           torch.tensor([0.]).to(device), torch.tensor([1.]).to(device)).to(device)
        xtilde_t = self.embedding(E_tm1) * mask.view(-1, 1)
        return xtilde_t


    def get_current_hidden_state(self, xtilde_t, htilde_tm1):
        # Recall:
        #   xtilde_t is of shape (M, Itilde)
        #   htilde_tm1 is of shape (M, 2 * H) or a tuple of two of those (LSTM)
        #   htilde_t (output) is of same shape as htilde_tm1

        # htilde_t = self.cell(xtilde_t, htilde_tm1)
        # return htilde_t

        if self.cell_type == 'lstm':
            htilde_tm1 = (htilde_tm1[0][:, :self.hidden_state_size],
                          htilde_tm1[1][:, :self.hidden_state_size])
        else:
            htilde_tm1 = htilde_tm1[:, :self.hidden_state_size]
        return self.cell(xtilde_t, htilde_tm1)

    def get_current_logits(self, htilde_t):
        # Recall:
        #   htilde_t is of shape (M, 2 * H), even for LSTM (cell state discarded)
        #   logits_t (output) is of shape (M, V)

        # logits_t = self.ff(htilde_t)
        # return logits_t

        logits = self.ff.forward(htilde_t)
        return logits


class DecoderWithAttention(DecoderWithoutAttention):
    '''A decoder, this time with attention

    Inherits from DecoderWithoutAttention to avoid repeated code.
    '''

    def init_submodules(self):
        # Hints:
        # 1. Same as the case without attention, you must initialize the
        #   following submodules: self.embedding, self.cell, self.ff
        # 2. You will need these object attributes:
        #   self.target_vocab_size, self.word_embedding_size, self.pad_id
        #   self.hidden_state_size, self.cell_type.
        # 3. cell_type will be one of: ['lstm', 'gru', 'rnn']
        # 4. Relevant pytorch modules:
        #   torch.nn.{Embedding, Linear, LSTMCell, RNNCell, GRUCell}
        # 5. The implementation of this function should be different from
        #   DecoderWithoutAttention.init_submodules.

        # if self.cell_type == 'lstm':
        #     self.cell = torch.nn.LSTMCell(self.word_embedding_size + self.hidden_state_size, self.hidden_state_size)
        # elif self.cell_type == 'gru':
        #     self.cell = torch.nn.GRUCell(self.word_embedding_size + self.hidden_state_size, self.hidden_state_size)
        # else:
        #     self.cell = torch.nn.RNNCell(self.word_embedding_size + self.hidden_state_size, self.hidden_state_size)
        #
        # self.embedding = torch.nn.Embedding(self.target_vocab_size, self.word_embedding_size, padding_idx=self.pad_id)
        #
        # self.ff = torch.nn.Linear(self.hidden_state_size, self.target_vocab_size)

        init_packet = [self.word_embedding_size + self.hidden_state_size,
                       self.hidden_state_size]
        if self.cell_type == 'gru':
            initializer = torch.nn.GRUCell
        elif self.cell_type == 'rnn':
            initializer = torch.nn.RNNCell
        elif self.cell_type == 'lstm':
            initializer = torch.nn.LSTMCell
        else:
            raise ValueError(f"cell type: '{self.cell_type}' not valid.")
        self.cell = initializer(*init_packet)
        self.embedding = torch.nn.Embedding(self.target_vocab_size,
                                            self.word_embedding_size,
                                            padding_idx=self.pad_id)
        self.ff = torch.nn.Linear(self.hidden_state_size, self.target_vocab_size)


    def get_first_hidden_state(self, h, F_lens):
        # Hint: For this time, the hidden states should be initialized to zeros.
        return torch.zeros_like(h[0], device=h.device)

    def get_current_rnn_input(self, E_tm1, htilde_tm1, h, F_lens):
        # Hint: Use attend() for c_t
        # mask = (E_tm1 != self.pad_id).float().unsqueeze(1)  # (M,1)
        # E_t = self.embedding(E_tm1) * mask
        #
        # c_t = self.attend(htilde_tm1[0] if self.cell_type == 'lstm' else htilde_tm1, h, F_lens)  # (M, 2*H)
        #
        # return torch.cat([E_t, c_t], dim=1)

        device = h.device
        mask = torch.where(E_tm1 == torch.tensor([self.pad_id]).to(device),
                           torch.tensor([0.]).to(device), torch.tensor([1.]).to(device)).to(device)
        if self.cell_type == 'lstm':
            htilde_tm1 = htilde_tm1[0]  # take the hidden states
        prev_input = self.embedding(E_tm1) * mask.view(-1, 1)
        # prev input concatenated with the attention context
        return torch.cat([prev_input, self.attend(htilde_tm1, h, F_lens)], 1)


    def attend(self, htilde_t, h, F_lens):
        '''The attention mechanism. Calculate the context vector c_t.

        Parameters
        ----------
        htilde_t : torch.FloatTensor or tuple
            Like `htilde_tm1` (either a float tensor or a pair of float
            tensors), but matching the current hidden state.
        h : torch.FloatTensor
            A float tensor of shape ``(S, M, self.hidden_state_size)`` of
            hidden states of the encoder. ``h[s, m, i]`` is the
            ``i``-th index of the encoder RNN's last hidden state at time ``s``
            of the ``m``-th sequence in the batch. The states of the
            encoder have been right-padded such that ``h[F_lens[m]:, m]``
            should all be ignored.
        F_lens : torch.LongTensor
            An integer tensor of shape ``(M,)`` corresponding to the lengths
            of the encoded source sentences.

        Returns
        -------
        c_t : torch.FloatTensor
            A float tensor of shape ``(M, self.target_vocab_size)``. The
            context vectorc_t is the product of weights alpha_t and h.

        Hint: Use get_attention_weights() to calculate alpha_t.
        '''
        # alpha_t = self.get_attention_weights(htilde_t, h, F_lens) #(S,M)
        # c_t = torch.einsum('smh,sm->mh', h, alpha_t)
        #
        # return c_t

        alpha = self.get_attention_weights(htilde_t, h, F_lens)  # (S, N)
        alpha = alpha.transpose(0, 1)  # (N, S)
        alpha = alpha.unsqueeze(2)  # (N, S, 1)
        h = h.permute(1, 2, 0)  # (N, 2*H, S)
        c_t = torch.bmm(h, alpha).squeeze()  # (N, 2*H) as desired.
        return c_t


    def get_attention_weights(self, htilde_t, h, F_lens):
        # DO NOT MODIFY! Calculates attention weights, ensuring padded terms
        # in h have weight 0 and no gradient. You have to implement
        # get_energy_scores()
        # alpha_t (output) is of shape (S, M)
        e_t = self.get_energy_scores(htilde_t, h)
        pad_mask = torch.arange(h.shape[0], device=h.device)
        pad_mask = pad_mask.unsqueeze(-1) >= F_lens  # (S, M)
        e_t = e_t.masked_fill(pad_mask, -float('inf'))
        return torch.nn.functional.softmax(e_t, 0)

    def get_energy_scores(self, htilde_t, h):
        # Recall:
        #   htilde_t is of shape (M, 2 * H)
        #   h is of shape (S, M, 2 * H)
        #   e_t (output) is of shape (S, M)
        #
        # Hint:
        # Relevant pytorch functions: torch.nn.functional.cosine_similarity

        # htilde_t = htilde_t.unsqueeze(0)
        # return torch.nn.functional.cosine_similarity(htilde_t, h, dim=2)

        csim = torch.nn.CosineSimilarity(dim=2)
        htilde_t = htilde_t.unsqueeze(0)
        similarties = csim(htilde_t, h)
        return similarties



class DecoderWithMultiHeadAttention(DecoderWithAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.W is not None, 'initialize W!'
        assert self.Wtilde is not None, 'initialize Wtilde!'
        assert self.Q is not None, 'initialize Q!'

    def init_submodules(self):
        super().init_submodules()  # Do not modify this line

        # Hints:
        # 1. The above line should ensure self.ff, self.embedding, self.cell are
        #    initialized
        # 2. You need to initialize these submodules:
        #       self.W, self.Wtilde, self.Q
        # 3. You will need these object attributes:
        #       self.hidden_state_size
        # 4. self.W, self.Wtilde, and self.Q should process all heads at once. They
        #    should not be lists!
        # 5. You do *NOT* need self.heads at this point
        # 6. Relevant pytorch module: torch.nn.Linear (note: set bias=False!)
        assert False, "Fill me"

    def attend(self, htilde_t, h, F_lens):
        # Hints:
        # 1. You can use super().attend to call for the regular attention
        #   function.
        # 2. Relevant pytorch functions:
        #   tensor().repeat_interleave, tensor().view
        # 3. Fun fact:
        #   tensor([1,2,3,4]).repeat(2) will output tensor([1,2,3,4,1,2,3,4]).
        #   tensor([1,2,3,4]).repeat_interleave(2) will output
        #   tensor([1,1,2,2,3,3,4,4]), just like numpy.repeat.
        # 4. You *WILL* need self.heads at this point
        assert False, "Fill me"

class EncoderDecoder(EncoderDecoderBase):

    def init_submodules(self, encoder_class, decoder_class):
        # Hints:
        # 1. You must initialize these submodules:
        #   self.encoder, self.decoder
        # 2. encoder_class and decoder_class inherit from EncoderBase and
        #   DecoderBase, respectively.
        # 3. You will need these object attributes:
        #   self.source_vocab_size, self.source_pad_id,
        #   self.word_embedding_size, self.encoder_num_hidden_layers,
        #   self.encoder_hidden_size, self.encoder_dropout, self.cell_type,
        #   self.target_vocab_size, self.target_eos
        # 4. Recall that self.target_eos doubles as the decoder pad id since we
        #   never need an embedding for it
        ################
        # self.encoder = encoder_class(self.source_vocab_size, self.source_pad_id, self.word_embedding_size,
        #                              self.encoder_num_hidden_layers, self.encoder_hidden_size, self.encoder_dropout,
        #                              self.cell_type)
        # self.decoder = decoder_class(self.target_vocab_size, self.target_eos, self.word_embedding_size,
        #                              2 * self.encoder_hidden_size, self.cell_type)

        self.encoder = encoder_class(source_vocab_size=self.source_vocab_size,
                                     pad_id=self.source_pad_id,
                                     word_embedding_size=self.word_embedding_size,
                                     num_hidden_layers=self.encoder_num_hidden_layers,
                                     hidden_state_size=self.encoder_hidden_size,
                                     dropout=self.encoder_dropout,
                                     cell_type=self.cell_type)
        self.encoder.init_submodules()
        self.decoder = decoder_class(target_vocab_size=self.target_vocab_size,
                                     pad_id=self.target_eos,
                                     word_embedding_size=self.word_embedding_size,
                                     hidden_state_size=self.encoder_hidden_size * 2,
                                     cell_type=self.cell_type)
        self.decoder.init_submodules()

    def get_logits_for_teacher_forcing(self, h, F_lens, E):
        # Recall:
        #   h is of shape (S, M, 2 * H)
        #   F_lens is of shape (M,)
        #   E is of shape (T, M)
        #   logits (output) is of shape (T - 1, M, Vo)
        #
        # Hints:
        # 1. Relevant pytorch modules: torch.{zero_like, stack}
        # 2. Recall an LSTM's cell state is always initialized to zero.
        # 3. Note logits sequence dimension is one shorter than E (why?)
        ####################################
        # logits = []
        # h_tilde_tm1 = None
        # for t in range(E.size()[0] - 1):
        #     logit, h_tilde_tm1 = self.decoder.forward(E[t], h_tilde_tm1, h, F_lens)
        #     logits.append(logit)
        #
        # logits = torch.stack(logits, dim=0)
        # return logits
        ######################################
        # T, N = E.size()
        # htilde_tm1 = self.decoder.get_first_hidden_state(h, F_lens)
        # if self.cell_type == 'lstm':
        #     cell_state = torch.zeros_like(htilde_tm1).to(htilde_tm1.device)
        # logits = []
        # for i in range(T - 1):
        #     E_tm1 = E[i, :]
        #     xtilde_t = self.decoder.get_current_rnn_input(E_tm1, htilde_tm1, h, F_lens)
        #     if self.cell_type == 'lstm':
        #         htilde_tm1, cell_state = self.decoder.get_current_hidden_state(xtilde_t, (htilde_tm1, cell_state))
        #     else:
        #         htilde_tm1 = self.decoder.get_current_hidden_state(xtilde_t, htilde_tm1)
        #     logits.append(self.decoder.get_current_logits(htilde_tm1))
        # logits = torch.stack(logits, dim=0)
        # return logits

        logits = []  # for holding logits as we do all steps in time
        h_tilde_tm1 = None
        for t in range(E.size()[0] - 1):  # run all T, with first being the SOS
            l, h_tilde_tm1 = self.decoder.forward(E[t], h_tilde_tm1, h, F_lens)
            logits.append(l)
        logits = torch.stack(logits[:], 0)  # take all but the SOS one.
        return logits


    def update_beam(self, htilde_t, b_tm1_1, logpb_tm1, logpy_t):
        # perform the operations within the psuedo-code's loop in the
        # assignment.
        # You do not need to worry about which paths have finished, but DO NOT
        # re-normalize logpy_t.
        #
        # Recall
        #   htilde_t is of shape (M, K, 2 * H) or a tuple of two of those (LSTM)
        #   logpb_tm1 is of shape (M, K)
        #   b_tm1_1 is of shape (t, M, K)
        #   b_t_0 (first output) is of shape (M, K, 2 * H) or a tuple of two of
        #      those (LSTM)
        #   b_t_1 (second output) is of shape (t + 1, M, K)
        #   logpb_t (third output) is of shape (M, K)
        #
        # Hints:
        # 1. Relevant pytorch modules:
        #   torch.{flatten, topk, unsqueeze, expand_as, gather, cat}
        # 2. If you flatten a two-dimensional array of shape z of (A, B),
        #   then the element z[a, b] maps to z'[a*B + b]
        ####################################################
        # M, K, V = logpy_t.size()
        # logpb_tm1 = logpb_tm1.unsqueeze(-1).expand(-1, -1, V)
        # logpb_t = logpb_tm1 + logpy_t  # (M,K,V)
        # logpb_t, indices = logpb_t.view(M, -1).tpk(self.beam_width, dim=1)  # (M,K), (M,K)
        #
        # indices_k = indices // V  # (M, K)
        # indices_v = indices % V  # (M, K)
        #
        # if self.cell_type == 'lstm':
        #     b_t_0 = (htilde_t[0].gather(dim=1, index=indices_k.unsqueeze(-1).expand_as(htilde_t[0]))
        #              , htilde_t[1].gather(dim=1, index=indices_k.unsqueeze(-1).expand_as(htilde_t[1])))
        # else:
        #     b_t_0 = htilde_t.gather(dim=1, index=indices_k.unsqueeze(-1).expand_as(htilde_t))  # (M,K,2*H)
        #
        # b_tm1_1 = b_tm1_1.gather(dim=2, index=indices_k.unsqueeze(0).expand_as(b_tm1_1))  # (t,M,K)
        # b_t_1 = torch.cat([b_tm1_1, indices_v.unsqueeze(0)], dim=0)
        #
        # return b_t_0, b_t_1, logpb_t
        ########################################################
        V = logpy_t.size()[-1]
        all_paths = logpb_tm1.unsqueeze(-1) + logpy_t  # (N, K, V), add logprobs for new extensions
        all_paths = all_paths.view((all_paths.shape[0], -1))  # (N, K*V)
        logpb_t, v = all_paths.topk(self.beam_width,
                                    -1,
                                    largest=True,
                                    sorted=True)  # take beam_width best possible extensions
        logpb_t = logpb_t  # (N, K)
        # v is (N, K)
        # v are the indices of the maximal values.
        paths = torch.div(v, V)  # paths chosen to be kept
        v = torch.remainder(v, V)  # the indices of the extended words that are kept
        # choose the paths from b_tm1_1 that were kept in our next propogation
        b_tm1_1 = b_tm1_1.gather(2, paths.unsqueeze(0).expand_as(b_tm1_1))
        # choose the htdile that coorespond to the taken paths
        if self.cell_type == 'lstm':
            b_t_0 = (htilde_t[0].gather(1, paths.unsqueeze(-1).expand_as(htilde_t[0])),
                     htilde_t[1].gather(1, paths.unsqueeze(-1).expand_as(htilde_t[1])))
        else:
            b_t_0 = htilde_t.gather(1, paths.unsqueeze(-1).expand_as(htilde_t))
        v = v.unsqueeze(0)  # (1, N, K)
        b_t_1 = torch.cat([b_tm1_1, v], dim=0)
        return b_t_0, b_t_1, logpb_t
