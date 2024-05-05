import transformers
import torch


# pipeline method is a high-level API that allows you to use a model for a specific task
# pipeline has three steps - tokenization (pre-processing), model inference, and post-processing


def main():
    # Step 1: Tokenization
    # checkpoint is snapshot of model at a particular point in time
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)

    raw_inputs = [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
    # pt is pytorch tensor
    inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
    print(inputs)

    # {
    #     'input_ids': tensor([ 16 length array of 2 sequences or sentences of above raw_inputs
    #         [  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172, 2607,  2026,  2878,  2166,  1012,   102],
    #         [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,     0,     0,     0,     0,     0,     0]
    #     ]),
    #     'attention_mask': tensor([
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 1 - indicates padding
    #         [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0] 0 - indicates no padding
    #     ])
    # }

    # Step 2: Model Inference
    # checkpoint is snapshot of model at a particular point in time
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    model = transformers.AutoModelForSequenceClassification.from_pretrained(checkpoint)
    outputs = model(**inputs)
    print(outputs)
    print(outputs.logits.shape)

    # SequenceClassifierOutput(loss=None, logits=tensor([[-1.5607,  1.6123], - represents 1st sentence (negative,positive)
    #         [ 4.1692, -3.3464]], grad_fn=<AddmmBackward0>), hidden_states=None, attentions=None)
    # torch.Size([2, 2])

    # Step 3: Post-processing
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    print(predictions)

    # First sentence: NEGATIVE: 0.0402, POSITIVE: 0.9598
    # Second sentence: NEGATIVE: 0.9995, POSITIVE: 0.0005


if __name__ == '__main__':
    main()
