import transformers


def main():
    generator = transformers.pipeline("text-generation")
    result = generator("I like to", num_return_sequences=3, max_length=30, truncation=True)
    print(result)

    generator = transformers.pipeline("text-generation", model="distilgpt2")
    result = generator("I like to", num_return_sequences=3, max_length=30, truncation=True)
    print(result)


if __name__ == '__main__':
    main()
