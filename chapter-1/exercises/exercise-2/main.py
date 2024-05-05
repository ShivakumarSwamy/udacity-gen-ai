import transformers
import pandas
import torch


def main():
    # ----------------------
    # Step 1: Pre-processing
    # ----------------------
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

    # partial sentence
    text = "Udacity is the best place to learn about generative"
    inputs = tokenizer(text, return_tensors="pt")
    print(inputs)

    # {'input_ids': tensor([[  52,   67, 4355,  318,  262, 1266, 1295,  284, 2193,  546, 1152,  876]]),
    # 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
    # decode token

    id_text_list_tuple = []
    for token_id in inputs["input_ids"][0]:
        id_text_list_tuple.append((token_id, tokenizer.decode(token_id)))

    print(pandas.DataFrame(id_text_list_tuple, columns=["Token ID", "Token"]))

    # {'input_ids': tensor([[  52,   67, 4355,  318,  262, 1266, 1295,  284, 2193,  546, 1152,  876]]),
    # 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
    #         Token ID   Token
    # 0     tensor(52)       U
    # 1     tensor(67)       d
    # 2   tensor(4355)   acity
    # 3    tensor(318)      is
    # 4    tensor(262)     the
    # 5   tensor(1266)    best
    # 6   tensor(1295)   place
    # 7    tensor(284)      to
    # 8   tensor(2193)   learn
    # 9    tensor(546)   about
    # 10  tensor(1152)   gener
    # 11   tensor(876)   ative

    # ----------------------
    # Step 2:Model Inference
    # ----------------------

    # Causal Language Modeling (CLM) is a type of language modeling
    # where the model makes predictions based on the previous tokens in the sentence
    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")

    outputs = model(**inputs)
    print(outputs.logits.shape)
    # Why is the shape of the logits tensor (1, 12, 50257)?
    # The shape of the logits tensor (1, 12, 50257) indicates that the model has generated 12 tokens.
    # The first dimension represents the batch size, which is 1 in this case.
    # The second dimension represents the sequence length, which is 12 in this case.
    # The third dimension represents the number of tokens in the vocabulary, which is 50257 in this case.

    print(outputs.logits[:, -1, :])
    # What does .logits[:, -1, :] do?
    # The .logits[:, -1, :] expression selects the last token in the sequence.
    # The first colon [:] selects all the elements in the first dimension (batch size).
    # The -1 index selects the last element in the second dimension (sequence length).
    # The colon [:] selects all the elements in the third dimension (vocabulary size).
    # Therefore, .logits[:, -1, :] selects the logits for the last token in the sequence.

    # -----------------------
    # Step 3: Post-processing
    # -----------------------

    # The model outputs logits, which are raw non-normalized predictions
    # To get the probabilities, we apply the softmax function
    last_tokens = outputs.logits[:, -1, :]
    predictions = torch.nn.functional.softmax(last_tokens[0], dim=-1)

    id_token_probability_list_tuple = []
    for token_id, token_prob in enumerate(predictions):
        if token_prob and token_prob.item():
            id_token_probability_list_tuple.append((token_id, tokenizer.decode(token_id), token_prob.item()))

    print(pandas.DataFrame(id_token_probability_list_tuple, columns=["Token ID", "Token", "Probability"])
          .sort_values("Probability", ascending=False)[:5])

    #        Token ID         Token  Probability
    # 8300       8300   programming     0.157589
    # 4673       4673      learning     0.148416
    # 4981       4981        models     0.048505
    # 17219     17219       biology     0.046481
    # 16113     16113    algorithms     0.027796

    most_probable_next_token = torch.argmax(predictions).item()

    print(f"Next token id: {most_probable_next_token}")
    print(f"Next token: {tokenizer.decode(most_probable_next_token)}")

    # Next token id: 8300
    # Next token: programming
    text += tokenizer.decode(most_probable_next_token)
    print(text)


if __name__ == '__main__':
    main()
