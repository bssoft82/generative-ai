import torch
import logging

from engine.GPTLanguageModel import GPTLanguageModel
from src.utils.config import Config
from src.utils.logger import setup_logging


# data loading
def get_batch(split, train_data, val_data, block_size, batch_size, device):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, eval_iters, train_data, val_data, block_size, batch_size, device):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data, block_size, batch_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def main():
    # Read configuration
    config_path = '../../config/model_config.yaml'
    config = Config(config_path)
    config.log_state()

    # Configure logging
    log_config_path = '../../config/log_config.yaml'
    log_file = '../../logs/app.log'
    setup_logging(log_file, log_config_path)

    logging.info("--------------------------------------------------------")
    logging.info("Starting the application.")

    #Business Logic Start
    torch.manual_seed(1337)

    # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    with open('../../res/input-shakespeare.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    model = GPTLanguageModel(config, vocab_size)
    m = model.to(config.device)
    # print the number of parameters in the model
    logging.info(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    for iter in range(config.max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
            #def estimate_loss(model, eval_iters, train_data, val_data, block_size, batch_size, device):
            losses = estimate_loss(model, config.eval_iters, train_data, val_data, config.block_size, config.batch_size, config.device)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train', train_data, val_data, config.block_size, config.batch_size, config.device)

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
    # open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
    # Business Logic End
    logging.info("Application finished.")
    logging.info("--------------------------------------------------------")


if __name__ == "__main__":
    main()
