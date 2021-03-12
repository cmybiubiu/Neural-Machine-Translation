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

'''Functions related to training and testing.

You don't need anything more than what's been imported here.
'''

import torch
import a2_bleu_score


from tqdm import tqdm


def train_for_epoch(model, dataloader, optimizer, device):
    '''Train an EncoderDecoder for an epoch

    An epoch is one full loop through the training data. This function:

    1. Defines a loss function using :class:`torch.nn.CrossEntropyLoss`,
       keeping track of what id the loss considers "padding"
    2. For every iteration of the `dataloader` (which yields triples
       ``F, F_lens, E``)
       1. Sends ``F`` to the appropriate device via ``F = F.to(device)``. Same
          for ``F_lens`` and ``E``.
       2. Zeros out the model's previous gradient with ``optimizer.zero_grad()``
       3. Calls ``logits = model(F, F_lens, E)`` to determine next-token
          probabilities.
       4. Modifies ``E`` for the loss function, getting rid of a token and
          replacing excess end-of-sequence tokens with padding using
        ``model.get_target_padding_mask()`` and ``torch.masked_fill``
       5. Flattens out the sequence dimension into the batch dimension of both
          ``logits`` and ``E``
       6. Calls ``loss = loss_fn(logits, E)`` to calculate the batch loss
       7. Calls ``loss.backward()`` to backpropagate gradients through
          ``model``
       8. Calls ``optim.step()`` to update model parameters
    3. Returns the average loss over sequences

    Parameters
    ----------
    model : EncoderDecoder
        The model we're training.
    dataloader : HansardDataLoader
        Serves up batches of data.
    device : torch.device
        A torch device, like 'cpu' or 'cuda'. Where to perform computations.
    optimizer : torch.optim.Optimizer
        Implements some algorithm for updating parameters using gradient
        calculations.

    Returns
    -------
    avg_loss : float
        The total loss divided by the total numer of sequence
    '''
    # If you want, instead of looping through your dataloader as
    # for ... in dataloader: ...
    # you can wrap dataloader with "tqdm":
    # for ... in tqdm(dataloader): ...
    # This will update a progress bar on every iteration that it prints
    # to stdout. It's a good gauge for how long the rest of the epoch
    # will take. This is entirely optional - we won't grade you differently
    # either way.
    # If you are running into CUDA memory errors part way through training,
    # try "del F, F_lens, E, logits, loss" at the end of each iteration of
    # the loop.

    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = 0
    nb_sequence = 0

    for F, F_lens, E in tqdm(dataloader):
        # 1. Sends ``F`` to the appropriate device via ``F = F.to(device)``. Same for ``F_lens`` and ``E``.
        nb_sequence += E.size()[1]
        F, F_lens, E = F.to(device), F_lens.to(device), E.to(device)

        # 2. Zeros out the model's previous gradient with ``optimizer.zero_grad()``
        optimizer.zero_grad()

        # 3. Calls ``logits = model(F, F_lens, E)`` to determine next-token
        #    probabilities.
        logits = model(F, F_lens, E)

        # 4. Modifies ``E`` for the loss function, getting rid of a token and
        #    replacing excess end-of-sequence tokens with padding using
        #  ``model.get_target_padding_mask()`` and ``torch.masked_fill``
        E = E[1:, :]  # (t-1, N)
        mask = model.get_target_padding_mask(E)

        # 5. Flattens out the sequence dimension into the batch dimension of both
        #    ``logits`` and ``E``
        logits = torch.masked_fill(logits, mask.unsqueeze(-1), 0).view(-1, logits.size()[-1])
        E = torch.masked_fill(E, mask, 0).flatten()

        # 6. Calls ``loss = loss_fn(logits, E)`` to calculate the batch loss
        loss = loss_fn(logits, E)

        # 7. Calls ``loss.backward()`` to backpropagate gradients through
        #    ``model``
        loss.backward()

        # 8. Calls ``optim.step()`` to update model parameters
        optimizer.step()
        total_loss += loss.item()
        # del F, F_lens, E, logits, loss

    avg_loss = total_loss / nb_sequence
    return avg_loss


def compute_batch_total_bleu(E_ref, E_cand, target_sos, target_eos):
    '''Calculate the total BLEU score over elements in a batch

    Parameters
    ----------
    E_ref : torch.LongTensor
        A batch of reference transcripts of shape ``(T, M)``, including
        start-of-sequence tags and right-padded with end-of-sequence tags.
    E_cand : torch.LongTensor
        A batch of candidate transcripts of shape ``(T', M)``, also including
        start-of-sequence and end-of-sequence tags.
    target_sos : int
        The id of the start-of-sequence tag in the target vocabulary.
    target_eos : int
        The id of the end-of-sequence tag in the target vocabulary.

    Returns
    -------
    total_bleu : float
        The sum total BLEU score for across all elements in the batch. Use
        n-gram precision 4.
    '''
    # you can use E_ref.tolist() to convert the LongTensor to a python list
    # of numbers
    #######################
    # total_bleu = 0
    # N = E_ref.shape[1]
    # for i in range(N):
    #     reference_list = [x for x in E_ref[:, i].tolist() if x != target_sos and x != target_eos]
    #     candidate_list = [x for x in E_cand[:, i].tolist() if x != target_sos and x != target_eos]
    #     total_bleu += a2_bleu_score.BLEU_score(reference_list, candidate_list, 4)
    #
    # print("total: %f", total_bleu)
    # return total_bleu
    ########################
    total_bleu = 0
    T, N = E_ref.size()
    for i in range(N):
        reference = E_ref[:, i].tolist()
        candidate = E_cand[:, i].tolist()
        reference = reference[1:]
        candidate = candidate[1:]
        if target_eos in reference:
            reference = reference[:reference.index(target_eos)]
        if target_eos in candidate:
            candidate = candidate[:candidate.index(target_eos)]
        total_bleu += a2_bleu_score.BLEU_score(reference,
                                               candidate, 4)
    print("total: %f", total_bleu)
    return total_bleu




def compute_average_bleu_over_dataset(
        model, dataloader, target_sos, target_eos, device):
    '''Determine the average BLEU score across sequences

    This function Calculates the average BLEU score across all sequences in
    a single loop through the `dataloader`.

    1. For every iteration of the `dataloader` (which yields triples
       ``F, F_lens, E_ref``):
       1. Sends ``F`` to the appropriate device via ``F = F.to(device)``. Same
          for ``F_lens``. No need for ``E_cand``, since it will always be
          compared on the CPU.
       2. Performs a beam search by calling ``b_1 = model(F, F_lens)``
       3. Extracts the top path per beam as ``E_cand = b_1[..., 0]``
       4. Calculates the total BLEU score of the batch using
          :func:`compute_batch_total_bleu`
    2. Returns the average per-sequence BLEU score

    Parameters
    ----------
    model : EncoderDecoder
        The model we're testing.
    dataloader : HansardDataLoader
        Serves up batches of data.
    target_sos : int
        The id of the start-of-sequence tag in the target vocabulary.
    target_eos : int
        The id of the end-of-sequence tag in the target vocabulary.

    Returns
    -------
    avg_bleu : float
        The total BLEU score summed over all sequences divided by the number of
        sequences
    '''
    # 1. For every iteration of the `dataloader` (which yields triples
    #    ``F, F_lens, E_ref``):
    #    1. Sends ``F`` to the appropriate device via ``F = F.to(device)``. Same
    #       for ``F_lens``. No need for ``E_cand``, since it will always be
    #       compared on the CPU.
    #    2. Performs a beam search by calling ``b_1 = model(F, F_lens)``
    #    3. Extracts the top path per beam as ``E_cand = b_1[..., 0]``
    #    4. Calculates the total BLEU score of the batch using
    #       :func:`compute_batch_total_bleu`
    # 2. Returns the average per-sequence BLEU score

    bleu_score = 0
    nb_sequences = 0

    for F, F_lens, E_ref in dataloader:
        nb_sequences += E_ref.size()[1]
        F = F.to(device)
        F_lens = F_lens.to(device)
        b_1 = model(F, F_lens)
        E_cand = b_1[..., 0]
        bleu_score += compute_batch_total_bleu(E_ref, E_cand, target_sos, target_eos)

    avg_bleu = bleu_score/nb_sequences
    print("avg: %f", avg_bleu)

    return avg_bleu
