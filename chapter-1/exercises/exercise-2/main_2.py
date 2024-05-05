import transformers


def main():
    # Step 1: Pre-processing
    text = "Once upon a time, generative models"
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    inputs = tokenizer(text, return_tensors="pt")
    print(inputs)

    # Step 2: Model Inference &  Post-processing - Decoder
    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    output = model.generate(**inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)
    # What does the pad_token_id argument do?
    # The pad_token_id argument specifies the token ID used for padding.
    # The model pads the output sequence until it reaches the max_length with the pad_token_id.
    # The model stops generating tokens when it encounters the pad_token_id.
    # The pad_token_id is the end-of-sequence token for the GPT-2 model.
    # The pad_token_id is the eos_token_id for the GPT-2 model.
    # What does the max_length argument do?
    # The max_length argument specifies the maximum length of the generated sequence.
    # The model stops generating tokens when the length of the generated sequence reaches max_length.
    # The max_length argument is set to 100 in this case.
    # The model generates a sequence of up to 100 tokens.

    print(tokenizer.decode(output[0]))


if __name__ == '__main__':
    main()
