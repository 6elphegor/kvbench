# Attention
Standard attention does not scale. As context length grows, attention becomes increasingly diffuse and waters down the desired value signal. The solution is to square the attention scores before softmax. The power of 2 of attention scores is the critical point of stability. Less than 2, and attention grows diffuse. Greater than 2, and attention spikes as context grows.

This can be seen by analyzing the value variance as the context length grows. Assume the keys, queries, and values are distributed according to the standard normal distribution. The resulting attention scores are then standard normally distributed because of the scale factor used in attention. The following graph shows how the value variance changes as the context length grows depending on the operation applied to the attention scores. The value variance of None decays rapidly (diffuse), Cubed approaches 1 (spikes), but Squared remains stable between 0 and 1.
![Variance of softmax-weighted sum](softmax_weighted_sum_cubed.png)

I cannot claim complete credit for quadratic attention as I encountered it in an article linked in an X post. However, this attention correction on its own is only part of the solution to context length generalization. Without correctly handling the position embeddings, models still will not generalize to context lengths longer than what is encountered during training.
