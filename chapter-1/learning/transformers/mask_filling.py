import transformers


def main():
    un_masker = transformers.pipeline('fill-mask')
    result = un_masker("Bangalore has lot of <mask> companies", top_k=3)
    print(result)


if __name__ == '__main__':
    main()
