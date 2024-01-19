import torch
import spacy
from torchtext.data.metrics import bleu_score

def translate_sentence(model, sentence, spacy_es, es_vocab, en_vocab, device, max_length=50):
    # Tokenize the sentence using spacy_es
    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_es(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Add <SOS> and <EOS> tokens
    tokens.insert(0, '<sos>')
    tokens.append('<eos>')

    # Convert tokens to indices
    text_to_indices = [es_vocab[token] if token in es_vocab else es_vocab['<unk>'] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    # Translation with the model
    with torch.no_grad():
        hidden, cell = model.encoder(sentence_tensor)

    outputs = [en_vocab['<sos>']]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, hidden, cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Break if the <EOS> token is reached
        if best_guess == en_vocab['<eos>']:
            break

    # Convert indices to words using the get_itos method of en_vocab
    en_itos = en_vocab.get_itos()
    translated_sentence = [en_itos[idx] for idx in outputs]

    # Remove the <SOS> token
    return translated_sentence[1:]

def bleu(data, model, spacy_es, es_vocab, en_vocab, device):
    targets = []
    outputs = []

    for example in data:
        src = example[0]  # Spanish sentence
        trg = example[1]  # English sentence

        prediction = translate_sentence(model, src, spacy_es, es_vocab, en_vocab, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg.split()])
        outputs.append(prediction)

    return bleu_score(outputs, targets)

def save_checkpoint(state, filename="seq2seq_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])