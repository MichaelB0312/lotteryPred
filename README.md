# lotteryPred

first: building the VAE with Transformer encoder, canceling the using of nn.embedding because it's just numbers.
 - we selected an MSE reconstruction loss, and made round and clip for the sampling of new 6 balls.
 - running some betas values, \beta=2 is the best, visually looked, results.
 - we've examined 1 to 3 layers to the transformer encoder, 2 is the best.
 - now, lets consider another layer for the VAE Decoder.
