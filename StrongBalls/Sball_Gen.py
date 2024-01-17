
from parser_code import args, MAX_STRONG
from runStrong import test_data

def generate(model, vocab, nwords=100, temp=1.0):
    model.eval()
    ntokens = MAX_STRONG + 1
    model_input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)
    words = []
    with torch.no_grad():
        for i in range(nwords):
            output = model(model_input, None)
            output = torch.softmax(output, dim=-1)
            word_weights = output[-1].squeeze().div(temp).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            word_tensor = torch.Tensor([[word_idx]]).long().to(device)
            model_input = torch.cat([model_input, word_tensor], 0)
            word = itos[word_idx]
            words.append(word)
    return words